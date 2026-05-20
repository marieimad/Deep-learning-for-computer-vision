"""
model.py
────────
Architecture deux streams :
  RGB Stream    : ViT-B/16 frame-wise → Temporal Transformer
  KP Stream     : OpenPose (201-dim)  → MLP → Temporal Transformer
  Fusion        : Cross-Attention → projection → BART dim (768)
  Decoder       : BART-base seq2seq
"""

import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, ViTModel

# ── Config ────────────────────────────────────────────────────────────────────
T          = 48
KP_DIM     = 201   # 75 pose + 63 left_hand + 63 right_hand (OpenPose)
D_MODEL    = 512
N_HEADS    = 8
N_LAYERS   = 2
DROPOUT    = 0.1
BART_MODEL = 'facebook/bart-base'


# ── Stream RGB ────────────────────────────────────────────────────────────────
class RGBStream(nn.Module):
    def __init__(self, d_out=D_MODEL, freeze_vit=True):
        super().__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad_(False)
        self.proj = nn.Linear(self.vit.config.hidden_size, d_out)
        enc = nn.TransformerEncoderLayer(d_out, N_HEADS, d_out*4, DROPOUT, batch_first=True)
        self.temporal = nn.TransformerEncoder(enc, N_LAYERS)
        self.pos_embed = nn.Embedding(T, d_out)

    def forward(self, frames):
        B, T_, C, H, W = frames.shape
        flat = frames.view(B*T_, C, H, W)
        with torch.no_grad():
            cls = self.vit(pixel_values=flat).last_hidden_state[:, 0, :]
        feat = self.proj(cls).view(B, T_, -1)
        pos  = torch.arange(T_, device=frames.device).unsqueeze(0)
        return self.temporal(feat + self.pos_embed(pos))


# ── Stream Keypoints ──────────────────────────────────────────────────────────
class KeypointStream(nn.Module):
    def __init__(self, kp_dim=KP_DIM, d_out=D_MODEL):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(kp_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(256, d_out),  nn.LayerNorm(d_out),
        )
        enc = nn.TransformerEncoderLayer(d_out, N_HEADS, d_out*4, DROPOUT, batch_first=True)
        self.temporal  = nn.TransformerEncoder(enc, N_LAYERS)
        self.pos_embed = nn.Embedding(T, d_out)

    def forward(self, kp):
        B, T_, _ = kp.shape
        feat = self.encoder(kp)
        pos  = torch.arange(T_, device=kp.device).unsqueeze(0)
        return self.temporal(feat + self.pos_embed(pos))


# ── Fusion ────────────────────────────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    def __init__(self, d_model=D_MODEL, bart_dim=768):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, N_HEADS, DROPOUT, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model * 2, bart_dim)

    def forward(self, rgb, kp):
        attended, _ = self.cross_attn(rgb, kp, kp)
        fused = torch.cat([self.norm(rgb + attended), kp], dim=-1)
        return self.proj(fused)


# ── Modèle complet ────────────────────────────────────────────────────────────
class SignLanguageTranslator(nn.Module):
    def __init__(self, freeze_vit=True, freeze_bart_encoder=True):
        super().__init__()
        self.rgb_stream = RGBStream(freeze_vit=freeze_vit)
        self.kp_stream  = KeypointStream()
        self.fusion     = CrossAttentionFusion()
        self.bart       = BartForConditionalGeneration.from_pretrained(BART_MODEL)
        if freeze_bart_encoder:
            for p in self.bart.model.encoder.parameters():
                p.requires_grad_(False)

    def _shift_right(self, labels):
        pad = self.bart.config.pad_token_id
        bos = self.bart.config.decoder_start_token_id
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0]  = bos
        shifted[shifted == -100] = pad
        return shifted

    def forward(self, frames, keypoints, labels, attention_mask):
        enc = self.fusion(self.rgb_stream(frames), self.kp_stream(keypoints))
        return self.bart(
            encoder_outputs=(enc,),
            decoder_input_ids=self._shift_right(labels),
            labels=labels,
        )

    @torch.no_grad()
    def generate(self, frames, keypoints, max_new_tokens=50, num_beams=4):
        enc = self.fusion(self.rgb_stream(frames), self.kp_stream(keypoints))
        return self.bart.generate(
            encoder_outputs=(enc,),
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            early_stopping=True,
        )


def count_params(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params     : {total:,}')
    print(f'Trainable params : {trainable:,}')
