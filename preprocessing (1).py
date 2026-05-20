"""
preprocessing.py
────────────────
Extrait depuis chaque clip How2Sign :
  - Les frames RGB (resizées, normalisées) → .npy  shape (T, 224, 224, 3)
  - Les keypoints OpenPose (body + mains)   → .npy  shape (T, 201)

KP_DIM = 75 (pose) + 63 (left hand) + 63 (right hand) = 201
"""

import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MAX_FRAMES = 48
FRAME_SIZE = (224, 224)
KP_DIM     = 201   # 75 pose + 63 left_hand + 63 right_hand


# ── Keypoints OpenPose ────────────────────────────────────────────────────────
def read_openpose_json(json_path: str) -> np.ndarray:
    """Lit un fichier JSON OpenPose et retourne un vecteur (201,)."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        if not data['people']:
            return np.zeros(KP_DIM, dtype=np.float32)
        p = data['people'][0]
        pose  = np.array(p.get('pose_keypoints_2d',       [0]*75),  dtype=np.float32)
        lhand = np.array(p.get('hand_left_keypoints_2d',  [0]*63),  dtype=np.float32)
        rhand = np.array(p.get('hand_right_keypoints_2d', [0]*63),  dtype=np.float32)
        return np.concatenate([pose, lhand, rhand])  # (201,)
    except Exception:
        return np.zeros(KP_DIM, dtype=np.float32)


def extract_keypoints_clip(kp_json_dir: str) -> np.ndarray:
    """
    Lit tous les JSON d'un clip, sous-échantillonne à MAX_FRAMES.
    Args:
        kp_json_dir : dossier contenant les *_keypoints.json du clip
    Returns:
        (MAX_FRAMES, 201)
    """
    json_files = sorted(Path(kp_json_dir).glob('*_keypoints.json'))
    if not json_files:
        return np.zeros((MAX_FRAMES, KP_DIM), dtype=np.float32)

    total = len(json_files)
    indices = np.linspace(0, total - 1, MAX_FRAMES, dtype=int)
    kps = [read_openpose_json(str(json_files[i])) for i in indices]
    return np.stack(kps, axis=0)   # (MAX_FRAMES, 201)


# ── Frames RGB ────────────────────────────────────────────────────────────────
def extract_frames_clip(video_path: str) -> np.ndarray:
    """
    Sous-échantillonne MAX_FRAMES depuis une vidéo.
    Returns: (MAX_FRAMES, 224, 224, 3) float32 normalisé [0,1]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((MAX_FRAMES, *FRAME_SIZE, 3), dtype=np.float32)

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), MAX_FRAMES, dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.float32) / 255.0   # (T, 224, 224, 3)


# ── Pipeline complet ──────────────────────────────────────────────────────────
def process_clip(clip_id, video_path, kp_json_dir, frames_out, kp_out):
    """Traite un clip et sauvegarde frames + keypoints en .npy."""
    fp = os.path.join(frames_out, f'{clip_id}.npy')
    kp = os.path.join(kp_out,    f'{clip_id}.npy')

    if os.path.exists(fp) and os.path.exists(kp):
        return True

    # Frames
    if not os.path.exists(fp):
        if not os.path.exists(video_path):
            return False
        frames = extract_frames_clip(video_path)
        np.save(fp, frames)

    # Keypoints
    if not os.path.exists(kp):
        if not os.path.exists(kp_json_dir):
            return False
        keypoints = extract_keypoints_clip(kp_json_dir)
        np.save(kp, keypoints)

    return True


def preprocess_dataset(
        annotation_csv : str,
        videos_dir     : str,
        kp_root        : str,   # dossier openpose_output/json/
        frames_out     : str,
        kp_out         : str,
        split          : str = 'train',
        max_clips      : int = None,
):
    os.makedirs(frames_out, exist_ok=True)
    os.makedirs(kp_out,     exist_ok=True)

    df = pd.read_csv(annotation_csv, sep='\t')
    if max_clips:
        df = df.head(max_clips)

    print(f'\n📦 Preprocessing {split} — {len(df)} clips')
    ok, fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        clip_id     = str(row['SENTENCE_NAME'])
        video_path  = os.path.join(videos_dir, f'{clip_id}.mp4')
        kp_json_dir = os.path.join(kp_root, clip_id)

        success = process_clip(clip_id, video_path, kp_json_dir,
                               frames_out, kp_out)
        if success: ok += 1
        else:       fail += 1

    print(f'\n✅ Succès : {ok}  |  ❌ Échecs : {fail}')
