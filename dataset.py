"""
dataset.py — How2Sign PyTorch Dataset
Charge frames .npy + keypoints .npy + tokenise avec BART.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer

MAX_FRAMES  = 48
MAX_SEQ_LEN = 50
KP_DIM      = 201


class How2SignDataset(Dataset):
    def __init__(self, annotation_csv, frames_dir, keypoints_dir,
                 split='train', augment=False):
        self.frames_dir    = frames_dir
        self.keypoints_dir = keypoints_dir
        self.augment       = augment and (split == 'train')
        self.tokenizer     = BartTokenizer.from_pretrained('facebook/bart-base')

        df = pd.read_csv(annotation_csv, sep='\t')
        valid = []
        for _, row in df.iterrows():
            cid = str(row['SENTENCE_NAME'])
            if os.path.exists(os.path.join(keypoints_dir, f'{cid}.npy')):
                valid.append(row)
        self.data = pd.DataFrame(valid).reset_index(drop=True)
        print(f'[{split}] {len(self.data)}/{len(df)} clips disponibles')

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        row     = self.data.iloc[idx]
        cid     = str(row['SENTENCE_NAME'])
        caption = str(row['SENTENCE'])

        # Frames (T,3,H,W) normalisé ImageNet
        frames_path = os.path.join(self.frames_dir, cid + ".npy")
        if os.path.exists(frames_path):
            frames = torch.from_numpy(np.load(frames_path)).permute(0,3,1,2).float()
            mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
            std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
            frames = (frames - mean) / std
        else:
            frames = torch.zeros(48, 3, 224, 224)

        # Keypoints (T, 201)
        kp = torch.from_numpy(
            np.load(os.path.join(self.keypoints_dir, f'{cid}.npy'))
        ).float()

        # Augmentation flip horizontal
        if self.augment and torch.rand(1).item() > 0.5:
            frames = torch.flip(frames, dims=[-1])
            kp = kp.clone(); kp[:, 0::3] = 1.0 - kp[:, 0::3]

        # Tokenisation
        enc = self.tokenizer(caption, max_length=MAX_SEQ_LEN,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        input_ids      = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)
        labels = input_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return dict(frames=frames, keypoints=kp,
                    input_ids=input_ids, attention_mask=attention_mask,
                    labels=labels, caption=caption)


def build_dataloaders(root, batch_size=3, num_workers=2):
    loaders = {}
    for split in ('train', 'val', 'test'):
        ds = How2SignDataset(
            annotation_csv = os.path.join(root, 'annotations', f'{split}.csv'),
            frames_dir     = os.path.join(root, 'frames'),
            keypoints_dir  = os.path.join(root, 'keypoints_npy'),
            split=split, augment=(split=='train'),
        )
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split=='train'),
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    drop_last=(split=='train'))
    return loaders['train'], loaders['val'], loaders['test']
