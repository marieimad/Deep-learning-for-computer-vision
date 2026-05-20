"""
train.py
────────
Boucle d'entraînement complète :
  - LR = 1e-6, Adam, batch size = 3
  - Early stopping (patience = 5)
  - Sauvegarde du meilleur checkpoint
  - Évaluation BLEU sur val à chaque époque
  - Logging dans un CSV (pour plots)
"""

import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BartTokenizer
import evaluate   # pip install evaluate sacrebleu
import numpy as np

from model   import SignLanguageTranslator
from dataset import build_dataloaders

# ── Hyperparamètres ───────────────────────────────────────────────────────────
LR           = 1e-6
BATCH_SIZE   = 3
EPOCHS       = 30
PATIENCE     = 5        # early stopping
GRAD_CLIP    = 1.0
MAX_NEW_TOK  = 50
NUM_BEAMS    = 4
DEVICE       = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f'Device : {DEVICE}')


# ── Utilitaires ───────────────────────────────────────────────────────────────
def decode_batch(token_ids, tokenizer):
    """Décode un batch de token ids en liste de strings."""
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def compute_bleu(predictions, references):
    """Calcule le BLEU score (sacrebleu)."""
    bleu = evaluate.load('sacrebleu')
    refs_wrapped = [[r] for r in references]   # sacrebleu attend List[List[str]]
    result = bleu.compute(predictions=predictions, references=refs_wrapped)
    return result['score']   # 0-100


# ── Entraînement une époque ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, scheduler_warmup=None):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        frames     = batch['frames'].to(DEVICE)          # (B, T, 3, 224, 224)
        keypoints  = batch['keypoints'].to(DEVICE)       # (B, T, 225)
        labels     = batch['labels'].to(DEVICE)          # (B, 50)
        att_mask   = batch['attention_mask'].to(DEVICE)  # (B, 50)

        optimizer.zero_grad()
        out  = model(frames, keypoints, labels, att_mask)
        loss = out.loss

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

        if n_batches % 50 == 0:
            print(f'  step {n_batches} | loss {loss.item():.4f}')

    return total_loss / n_batches


# ── Évaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_epoch(model, loader, tokenizer):
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_refs   = []
    n_batches  = 0

    for batch in loader:
        frames    = batch['frames'].to(DEVICE)
        keypoints = batch['keypoints'].to(DEVICE)
        labels    = batch['labels'].to(DEVICE)
        att_mask  = batch['attention_mask'].to(DEVICE)

        # Loss
        out = model(frames, keypoints, labels, att_mask)
        total_loss += out.loss.item()
        n_batches  += 1

        # Générer les traductions
        gen_ids = model.generate(frames, keypoints,
                                  max_new_tokens=MAX_NEW_TOK,
                                  num_beams=NUM_BEAMS)
        preds = decode_batch(gen_ids, tokenizer)
        refs  = batch['caption']   # list of str

        all_preds.extend(preds)
        all_refs.extend(refs)

    bleu = compute_bleu(all_preds, all_refs)
    avg_loss = total_loss / n_batches
    return avg_loss, bleu, all_preds[:5], all_refs[:5]   # +5 exemples pour debug


# ── Main ──────────────────────────────────────────────────────────────────────
def main(root: str, checkpoint_dir: str):
    """
    Args:
        root           : MyDrive/cs231n_sign/
        checkpoint_dir : où sauvegarder les .pt
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Data ──────────────────────────────────────────────────
    train_loader, val_loader, test_loader = build_dataloaders(
        root=root, batch_size=BATCH_SIZE
    )
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    # ── Modèle ────────────────────────────────────────────────
    model = SignLanguageTranslator(
        freeze_vit=True,
        freeze_bart_encoder=True,
    ).to(DEVICE)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-2,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                   patience=2, verbose=True)

    # ── Log CSV ───────────────────────────────────────────────
    log_path = os.path.join(checkpoint_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_bleu', 'time_s'])

    # ── Early stopping ────────────────────────────────────────
    best_bleu      = -1.0
    patience_count = 0
    best_ckpt      = os.path.join(checkpoint_dir, 'best_model.pt')

    print('\n🚀 Début entraînement\n')
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss          = train_epoch(model, train_loader, optimizer)
        val_loss, val_bleu, preds, refs = evaluate_epoch(model, val_loader, tokenizer)
        elapsed = time.time() - t0

        scheduler.step(val_bleu)

        print(f'\nÉpoque {epoch:02d}/{EPOCHS}')
        print(f'  Train loss : {train_loss:.4f}')
        print(f'  Val   loss : {val_loss:.4f}  |  BLEU : {val_bleu:.2f}')
        print(f'  Temps      : {elapsed:.0f}s')
        print(f'  Exemple    : {preds[0]}')
        print(f'  Référence  : {refs[0]}')

        # Log
        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_bleu, f'{elapsed:.0f}'])

        # Sauvegarde meilleur modèle
        if val_bleu > best_bleu:
            best_bleu = val_bleu
            patience_count = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'val_bleu'   : val_bleu,
            }, best_ckpt)
            print(f'  ✅ Nouveau meilleur BLEU = {best_bleu:.2f} → sauvegardé')
        else:
            patience_count += 1
            print(f'  Patience : {patience_count}/{PATIENCE}')
            if patience_count >= PATIENCE:
                print('\n⏹  Early stopping déclenché.')
                break

    # ── Évaluation finale sur test set ────────────────────────
    print('\n📊 Chargement du meilleur modèle pour évaluation test...')
    ckpt = torch.load(best_ckpt, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])

    test_loss, test_bleu, test_preds, test_refs = evaluate_epoch(
        model, test_loader, tokenizer
    )
    print(f'\n🏆 Test BLEU : {test_bleu:.2f}')
    print(f'   Test loss : {test_loss:.4f}')
    print('\nQuelques exemples :')
    for p, r in zip(test_preds, test_refs):
        print(f'  Pred : {p}')
        print(f'  Ref  : {r}')
        print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ROOT           = '/content/drive/MyDrive/cs231n_sign'
    CHECKPOINT_DIR = '/content/drive/MyDrive/cs231n_sign/checkpoints'
    main(ROOT, CHECKPOINT_DIR)
