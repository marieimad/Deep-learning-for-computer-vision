"""
dataset.py
──────────
PyTorch Dataset pour How2Sign.
Charge les frames pré-extraites et les keypoints,
et tokenise les captions avec le BART tokenizer.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer

# ── Config (doit correspondre à preprocessing.py) ─────────────────────────────
MAX_FRAMES  = 48
MAX_SEQ_LEN = 50
TOKENIZER   = 'facebook/bart-base'


class How2SignDataset(Dataset):
    """
    Args:
        annotation_csv : chemin vers train.csv / val.csv / test.csv
        frames_dir     : dossier contenant les <clip_id>.npy de frames
        keypoints_dir  : dossier contenant les <clip_id>.npy de keypoints
        split          : 'train' | 'val' | 'test'
        augment        : si True, applique data augmentation (train only)
    """

    def __init__(
            self,
            annotation_csv : str,
            frames_dir     : str,
            keypoints_dir  : str,
            split          : str = 'train',
            augment        : bool = False,
    ):
        self.frames_dir    = frames_dir
        self.keypoints_dir = keypoints_dir
        self.augment       = augment and (split == 'train')

        # ── Tokenizer ─────────────────────────────────────────
        self.tokenizer = BartTokenizer.from_pretrained(TOKENIZER)

        # ── Annotations ───────────────────────────────────────
        df = pd.read_csv(annotation_csv, sep='\t')

        # Garde uniquement les clips pour lesquels on a les .npy
        valid = []
        for _, row in df.iterrows():
            clip_id = str(row['SENTENCE_NAME'])
            fp = os.path.join(frames_dir,    f'{clip_id}.npy')
            kp = os.path.join(keypoints_dir, f'{clip_id}.npy')
            if os.path.exists(fp) and os.path.exists(kp):
                valid.append(row)

        self.data = pd.DataFrame(valid).reset_index(drop=True)
        print(f'[{split}] {len(self.data)} / {len(df)} clips disponibles')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row     = self.data.iloc[idx]
        clip_id = str(row['SENTENCE_NAME'])
        caption = str(row['SENTENCE'])

        # ── Frames  (T, 224, 224, 3)  →  (T, 3, 224, 224) ────
        frames = np.load(os.path.join(self.frames_dir, f'{clip_id}.npy'))
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # (T,3,H,W)

        # Normalisation ImageNet
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        frames = (frames - mean) / std

        # Augmentation : flip horizontal aléatoire (+ flip keypoints)
        flip = self.augment and (torch.rand(1).item() > 0.5)
        if flip:
            frames = torch.flip(frames, dims=[-1])   # flip W

        # ── Keypoints  (T, 225) ───────────────────────────────
        kp = np.load(os.path.join(self.keypoints_dir, f'{clip_id}.npy'))
        kp = torch.from_numpy(kp).float()   # (T, 225)

        if flip:
            # Inverser x pour chaque landmark (x est en [0,1])
            # indices : x = 0, 3, 6, ... (tous les 3 depuis 0)
            kp = kp.clone()
            kp[:, 0::3] = 1.0 - kp[:, 0::3]

        # ── Caption tokenisée ─────────────────────────────────
        enc = self.tokenizer(
            caption,
            max_length=MAX_SEQ_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids      = enc['input_ids'].squeeze(0)       # (50,)
        attention_mask = enc['attention_mask'].squeeze(0)  # (50,)

        # Labels = input_ids avec padding remplacé par -100
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'frames'        : frames,          # (48, 3, 224, 224)
            'keypoints'     : kp,              # (48, 225)
            'input_ids'     : input_ids,       # (50,)
            'attention_mask': attention_mask,  # (50,)
            'labels'        : labels,          # (50,)
            'caption'       : caption,         # str (pour debug)
        }


def build_dataloaders(
        root          : str,
        batch_size    : int = 3,
        num_workers   : int = 2,
):
    """
    Construit les 3 DataLoaders (train / val / test).

    Args:
        root : dossier MyDrive/cs231n_sign/
    """
    splits = {}
    for split in ('train', 'val', 'test'):
        ds = How2SignDataset(
            annotation_csv = os.path.join(root, 'annotations', f'{split}.csv'),
            frames_dir     = os.path.join(root, 'frames'),
            keypoints_dir  = os.path.join(root, 'keypoints'),
            split          = split,
            augment        = (split == 'train'),
        )
        splits[split] = DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = (split == 'train'),
            num_workers = num_workers,
            pin_memory  = True,
            drop_last   = (split == 'train'),
        )

    return splits['train'], splits['val'], splits['test']
