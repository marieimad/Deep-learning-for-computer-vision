"""
model.py
────────
Architecture deux streams :

  ┌─────────────────────────┐   ┌────────────────────────┐
  │  RGB Stream             │   │  Keypoint Stream        │
  │  ViT-B/16 (frame-wise)  │   │  MLP → Transformer      │
  │  → Temporal Transformer │   │  → Temporal Transformer │
  └────────────┬────────────┘   └───────────┬────────────┘
               └──────────┬────────────────┘
                     Cross-Attention Fusion
                           │
                     BART Decoder
                           │
                    English Translation
"""

import torch
import torch.nn as nn
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    ViTModel,
)

# ── Config ────────────────────────────────────────────────────────────────────
T          = 48      # frames
KP_DIM     = 225     # 75 landmarks × 3
D_MODEL    = 512     # dimension interne
N_HEADS    = 8
N_LAYERS   = 2
DROPOUT    = 0.1
BART_MODEL = 'facebook/bart-base'  # d_model = 768


# ── Stream RGB ────────────────────────────────────────────────────────────────
class RGBStream(nn.Module):
    """
    ViT-B/16 frame-wise (partagé entre toutes les frames) +
    Transformer temporel pour capturer la dynamique.
    """

    def __init__(self, d_out: int = D_MODEL, freeze_vit: bool = True):
        super().__init__()

        # ViT pré-entraîné (feature extractor)
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad_(False)

        vit_dim = self.vit.config.hidden_size  # 768

        # Projection vers d_out
        self.proj = nn.Linear(vit_dim, d_out)

        # Transformer temporel
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_out, nhead=N_HEADS,
            dim_feedforward=d_out * 4,
            dropout=DROPOUT, batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        # Encodage positionnel appris
        self.pos_embed = nn.Embedding(T, d_out)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames : (B, T, 3, 224, 224)
        Returns:
            (B, T, d_out)
        """
        B, T_actual, C, H, W = frames.shape

        # Passe chaque frame dans ViT (batch over time)
        frames_flat = frames.view(B * T_actual, C, H, W)   # (B*T, 3, 224, 224)
        with torch.set_grad_enabled(not self.vit.parameters().__next__().requires_grad == False):
            vit_out = self.vit(pixel_values=frames_flat).last_hidden_state[:, 0, :]  # (B*T, 768)

        feat = self.proj(vit_out).view(B, T_actual, -1)    # (B, T, d_out)

        # Ajout encodage positionnel
        positions = torch.arange(T_actual, device=frames.device).unsqueeze(0)  # (1, T)
        feat = feat + self.pos_embed(positions)

        return self.temporal_transformer(feat)   # (B, T, d_out)


# ── Stream Keypoints ──────────────────────────────────────────────────────────
class KeypointStream(nn.Module):
    """
    MLP par frame pour encoder les 225 keypoints,
    puis Transformer temporel.
    """

    def __init__(self, kp_dim: int = KP_DIM, d_out: int = D_MODEL):
        super().__init__()

        self.frame_encoder = nn.Sequential(
            nn.Linear(kp_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, d_out),
            nn.LayerNorm(d_out),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_out, nhead=N_HEADS,
            dim_feedforward=d_out * 4,
            dropout=DROPOUT, batch_first=True,
        )
        self.temporal_transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)

        self.pos_embed = nn.Embedding(T, d_out)

    def forward(self, keypoints: torch.Tensor) -> torch.Tensor:
        """
        Args:
            keypoints : (B, T, 225)
        Returns:
            (B, T, d_out)
        """
        B, T_actual, _ = keypoints.shape
        feat = self.frame_encoder(keypoints)   # (B, T, d_out)

        positions = torch.arange(T_actual, device=keypoints.device).unsqueeze(0)
        feat = feat + self.pos_embed(positions)

        return self.temporal_transformer(feat)   # (B, T, d_out)


# ── Fusion ────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention : RGB query sur Keypoint key/value.
    Puis concaténation + projection vers la dim BART (768).
    """

    def __init__(self, d_model: int = D_MODEL, bart_dim: int = 768):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=N_HEADS,
            dropout=DROPOUT, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        # Concatène les deux streams puis projette vers bart_dim
        self.proj = nn.Linear(d_model * 2, bart_dim)

    def forward(self, rgb_feat: torch.Tensor, kp_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb_feat : (B, T, D_MODEL)
            kp_feat  : (B, T, D_MODEL)
        Returns:
            (B, T, bart_dim)   ← encoder_hidden_states pour BART
        """
        # RGB attend sur Keypoints
        attended, _ = self.cross_attn(
            query=rgb_feat,
            key=kp_feat,
            value=kp_feat,
        )
        attended = self.norm(rgb_feat + attended)   # residual

        # Concat + projection
        fused = torch.cat([attended, kp_feat], dim=-1)   # (B, T, 2*D)
        return self.proj(fused)   # (B, T, 768)


# ── Modèle complet ────────────────────────────────────────────────────────────
class SignLanguageTranslator(nn.Module):
    """
    Full pipeline :
      frames + keypoints → encoder fusion → BART decoder → English text
    """

    def __init__(self, freeze_vit: bool = True, freeze_bart_encoder: bool = True):
        super().__init__()

        self.rgb_stream = RGBStream(d_out=D_MODEL, freeze_vit=freeze_vit)
        self.kp_stream  = KeypointStream(kp_dim=KP_DIM, d_out=D_MODEL)
        self.fusion     = CrossAttentionFusion(d_model=D_MODEL, bart_dim=768)

        # BART complet (on utilisera seulement le décodeur + embedding)
        self.bart = BartForConditionalGeneration.from_pretrained(BART_MODEL)

        # Geler l'encodeur BART (on le remplace par notre fusion)
        if freeze_bart_encoder:
            for p in self.bart.model.encoder.parameters():
                p.requires_grad_(False)

    def forward(
            self,
            frames         : torch.Tensor,   # (B, T, 3, 224, 224)
            keypoints      : torch.Tensor,   # (B, T, 225)
            labels         : torch.Tensor,   # (B, seq_len)
            attention_mask : torch.Tensor,   # (B, seq_len)
    ):
        # ── Encode ──────────────────────────────────────────
        rgb_feat   = self.rgb_stream(frames)              # (B, T, D)
        kp_feat    = self.kp_stream(keypoints)            # (B, T, D)
        enc_hidden = self.fusion(rgb_feat, kp_feat)       # (B, T, 768)

        # ── Decode avec BART ─────────────────────────────────
        # On injecte nos features comme encoder_hidden_states
        out = self.bart(
            encoder_outputs=(enc_hidden,),
            decoder_input_ids=self._shift_right(labels),
            labels=labels,
        )
        return out   # out.loss, out.logits

    def _shift_right(self, labels: torch.Tensor) -> torch.Tensor:
        """Décale les labels d'un token vers la droite (teacher forcing)."""
        pad = self.bart.config.pad_token_id
        bos = self.bart.config.decoder_start_token_id
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0]  = bos
        shifted[shifted == -100] = pad
        return shifted

    @torch.no_grad()
    def generate(
            self,
            frames    : torch.Tensor,
            keypoints : torch.Tensor,
            max_new_tokens: int = 50,
            num_beams : int = 4,
    ):
        """Génère une traduction (inférence)."""
        rgb_feat   = self.rgb_stream(frames)
        kp_feat    = self.kp_stream(keypoints)
        enc_hidden = self.fusion(rgb_feat, kp_feat)

        out = self.bart.generate(
            encoder_outputs=(enc_hidden,),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )
        return out   # token ids → décode avec le tokenizer


# ── Comptage paramètres ───────────────────────────────────────────────────────
def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Paramètres totaux    : {total:,}')
    print(f'Paramètres entraîn.  : {trainable:,}')


if __name__ == '__main__':
    model = SignLanguageTranslator()
    count_params(model)

    # Test forward pass
    B = 2
    frames    = torch.randn(B, T, 3, 224, 224)
    keypoints = torch.randn(B, T, KP_DIM)
    labels    = torch.randint(0, 50264, (B, 50))
    att_mask  = torch.ones(B, 50).long()

    out = model(frames, keypoints, labels, att_mask)
    print(f'\nLoss  : {out.loss.item():.4f}')
    print(f'Logits: {out.logits.shape}')   # (B, 50, vocab_size)
