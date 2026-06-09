# dataset.py — how2sign pytorch dataset
# loads rgb frames (.npy) + mediapipe keypoints (.npy) and tokenizes with bart

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

        # only keep clips that have keypoint files
        df = pd.read_csv(annotation_csv, sep='\t')
        valid = []
        for _, row in df.iterrows():
            cid = str(row['SENTENCE_NAME'])
            if os.path.exists(os.path.join(keypoints_dir, f'{cid}.npy')):
                valid.append(row)
        self.data = pd.DataFrame(valid).reset_index(drop=True)
        print(f'[{split}] {len(self.data)}/{len(df)} clips available')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row     = self.data.iloc[idx]
        cid     = str(row['SENTENCE_NAME'])
        caption = str(row['SENTENCE'])

        # load rgb frames — fall back to zeros if not extracted yet
        # (only 55.8% of training clips have rgb due to disk constraints)
        frames_path = os.path.join(self.frames_dir, cid + '.npy')
        if os.path.exists(frames_path):
            frames = torch.from_numpy(np.load(frames_path)).permute(0, 3, 1, 2).float()
            # imagenet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            frames = (frames - mean) / std
        else:
            frames = torch.zeros(MAX_FRAMES, 3, 224, 224)

        # load mediapipe keypoints (T, 201) — stored in pixel space
        kp = torch.from_numpy(
            np.load(os.path.join(self.keypoints_dir, f'{cid}.npy'))
        ).float()

        # horizontal flip augmentation for training
        # note: we only flip the rgb frames here since keypoints are in pixel
        # space and we don't have image width to properly mirror the x coordinates
        if self.augment and torch.rand(1).item() > 0.5:
            frames = torch.flip(frames, dims=[-1])

        # tokenize caption with bart tokenizer
        enc = self.tokenizer(caption, max_length=MAX_SEQ_LEN,
                             padding='max_length', truncation=True,
                             return_tensors='pt')
        input_ids      = enc['input_ids'].squeeze(0)
        attention_mask = enc['attention_mask'].squeeze(0)

        # replace padding token ids with -100 so they're ignored in the loss
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
            split=split,
            augment=(split == 'train'),
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == 'train'),
        )
    return loaders['train'], loaders['val'], loaders['test']
