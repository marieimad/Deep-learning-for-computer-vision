# train.py
# full training loop for the two-stream sign language translation model
#   - lr = 1e-6 (very low because bart is pretrained — larger lr would destroy it)
#   - batch size = 3 (constrained by t4 gpu vram with 48 frames at 224x224)
#   - early stopping patience = 5
#   - saves best checkpoint + resume checkpoint (survives modal preemptions)
#   - evaluates bleu-4 on val set every epoch using sacrebleu

import os
import csv
import time
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BartTokenizer
import evaluate
import numpy as np

from model   import SignLanguageTranslator
from dataset import build_dataloaders

# hyperparams
LR          = 1e-6
BATCH_SIZE  = 3
EPOCHS      = 30
PATIENCE    = 5       # early stopping — stops after 5 epochs with no bleu improvement
GRAD_CLIP   = 1.0
MAX_NEW_TOK = 50
NUM_BEAMS   = 4
DEVICE = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')

print(f'device: {DEVICE}')


def decode_batch(token_ids, tokenizer):
    return tokenizer.batch_decode(token_ids, skip_special_tokens=True)


def compute_bleu(predictions, references):
    bleu = evaluate.load('sacrebleu')
    refs_wrapped = [[r] for r in references]
    result = bleu.compute(predictions=predictions, references=refs_wrapped)
    return result['score']


def train_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        frames    = batch['frames'].to(DEVICE)
        keypoints = batch['keypoints'].to(DEVICE)
        labels    = batch['labels'].to(DEVICE)
        att_mask  = batch['attention_mask'].to(DEVICE)

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

        out = model(frames, keypoints, labels, att_mask)
        total_loss += out.loss.item()
        n_batches  += 1

        gen_ids = model.generate(frames, keypoints,
                                 max_new_tokens=MAX_NEW_TOK,
                                 num_beams=NUM_BEAMS)
        preds = decode_batch(gen_ids, tokenizer)
        refs  = batch['caption']

        all_preds.extend(preds)
        all_refs.extend(refs)

    bleu     = compute_bleu(all_preds, all_refs)
    avg_loss = total_loss / n_batches
    return avg_loss, bleu, all_preds[:5], all_refs[:5]


def main(root, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(
        root=root, batch_size=BATCH_SIZE
    )
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    model = SignLanguageTranslator(
        freeze_vit=True,
        freeze_bart_encoder=True,
    ).to(DEVICE)

    # only optimize the params that are not frozen
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=1e-2,
    )
    # reduce lr when bleu plateaus
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    # csv log for plotting later
    log_path = os.path.join(checkpoint_dir, 'training_log.csv')
    with open(log_path, 'w', newline='') as f:
        csv.writer(f).writerow(['epoch', 'train_loss', 'val_loss', 'val_bleu', 'time_s'])

    best_bleu      = -1.0
    patience_count = 0
    best_ckpt      = os.path.join(checkpoint_dir, 'best_model.pt')
    resume_ckpt    = os.path.join(checkpoint_dir, 'resume_checkpoint.pt')

    # resume from checkpoint if it exists (modal can preempt runs)
    start_epoch = 1
    if os.path.exists(resume_ckpt):
        print('resuming from checkpoint...')
        ckpt = torch.load(resume_ckpt, map_location=DEVICE)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer'])
        best_bleu      = ckpt['best_bleu']
        patience_count = ckpt['patience_count']
        start_epoch    = ckpt['epoch'] + 1
        print(f'  resumed at epoch {start_epoch}, best bleu = {best_bleu:.2f}')

    print('\nstarting training\n')
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, val_bleu, preds, refs = evaluate_epoch(model, val_loader, tokenizer)
        elapsed = time.time() - t0

        scheduler.step(val_bleu)

        print(f'\nepoch {epoch:02d}/{EPOCHS}')
        print(f'  train loss : {train_loss:.4f}')
        print(f'  val loss   : {val_loss:.4f}  |  bleu : {val_bleu:.2f}')
        print(f'  time       : {elapsed:.0f}s')
        print(f'  example    : {preds[0]}')
        print(f'  reference  : {refs[0]}')

        with open(log_path, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, val_bleu, f'{elapsed:.0f}'])

        if val_bleu > best_bleu:
            best_bleu      = val_bleu
            patience_count = 0
            torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                        'optimizer': optimizer.state_dict(), 'val_bleu': val_bleu}, best_ckpt)
            print(f'  new best bleu = {best_bleu:.2f} -> saved')
        else:
            patience_count += 1
            print(f'  patience: {patience_count}/{PATIENCE}')
            if patience_count >= PATIENCE:
                print('\nearly stopping triggered.')
                break

        # save resume checkpoint after every epoch
        torch.save({'epoch': epoch, 'model_state': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_bleu': best_bleu, 'patience_count': patience_count}, resume_ckpt)
        import modal; modal.Volume.from_name("cs231n-data").commit()

    # final eval on test set
    print('\nloading best model for test evaluation...')
    torch.load(best_ckpt, map_location=DEVICE)
    if len(test_loader.dataset) > 0:
        test_loss, test_bleu, test_preds, test_refs = evaluate_epoch(model, test_loader, tokenizer)
        print(f'\ntest bleu : {test_bleu:.2f}')
        print(f'test loss : {test_loss:.4f}')
    else:
        print('test set empty — skipping.')


if __name__ == '__main__':
    ROOT           = '/root/cs231n_sign'
    CHECKPOINT_DIR = '/root/cs231n_sign/checkpoints'
    main(ROOT, CHECKPOINT_DIR)
