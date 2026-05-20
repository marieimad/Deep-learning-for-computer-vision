"""
preprocessing.py
────────────────
Extrait depuis chaque vidéo How2Sign :
  - Les frames RGB (resizées, normalisées) → .npy
  - Les keypoints MediaPipe (corps + mains) → .npy

Usage (dans Colab) :
    from preprocessing import preprocess_dataset
    preprocess_dataset(annotation_csv, videos_dir, frames_out, keypoints_out)
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
MAX_FRAMES   = 48        # on sous-échantillonne à MAX_FRAMES uniformément
FRAME_SIZE   = (224, 224)
# Keypoints : 33 pose + 21 main gauche + 21 main droite = 75 landmarks × (x,y,z)
KP_DIM       = 75 * 3   # = 225

# ── MediaPipe init ────────────────────────────────────────────────────────────
mp_holistic = mp.solutions.holistic

def extract_keypoints(results) -> np.ndarray:
    """Récupère les 75 landmarks (pose + mains) depuis un résultat MediaPipe."""
    # Pose : 33 points × 3 coords
    if results.pose_landmarks:
        pose = np.array([[lm.x, lm.y, lm.z]
                         for lm in results.pose_landmarks.landmark])   # (33, 3)
    else:
        pose = np.zeros((33, 3))

    # Main gauche : 21 points × 3
    if results.left_hand_landmarks:
        lh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.left_hand_landmarks.landmark])  # (21, 3)
    else:
        lh = np.zeros((21, 3))

    # Main droite : 21 points × 3
    if results.right_hand_landmarks:
        rh = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.right_hand_landmarks.landmark])  # (21, 3)
    else:
        rh = np.zeros((21, 3))

    return np.concatenate([pose, lh, rh], axis=0).flatten()  # (225,)


def sample_frames(cap, n: int):
    """Sous-échantillonne n frames uniformément depuis une vidéo ouverte."""
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        else:
            frame = cv2.resize(frame, FRAME_SIZE)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    return np.stack(frames, axis=0)   # (n, 224, 224, 3)


def process_video(video_path: str, frames_out: str, kp_out: str, clip_id: str):
    """
    Traite une vidéo et sauvegarde :
      - frames_out/<clip_id>.npy  → float32 (T, 224, 224, 3), normalisé [0,1]
      - kp_out/<clip_id>.npy      → float32 (T, 225)
    Retourne True si succès.
    """
    frames_path = os.path.join(frames_out, f'{clip_id}.npy')
    kp_path     = os.path.join(kp_out,     f'{clip_id}.npy')

    # Skip si déjà traité
    if os.path.exists(frames_path) and os.path.exists(kp_path):
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  ⚠️  Cannot open {video_path}')
        return False

    # ── Frames ────────────────────────────────────────────────
    frames_uint8 = sample_frames(cap, MAX_FRAMES)   # (T, 224, 224, 3)
    frames_float = frames_uint8.astype(np.float32) / 255.0
    np.save(frames_path, frames_float)

    # ── Keypoints ─────────────────────────────────────────────
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, MAX_FRAMES, dtype=int)

    keypoints = []
    with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5) as holistic:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                keypoints.append(np.zeros(KP_DIM, dtype=np.float32))
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(frame_rgb)
            keypoints.append(extract_keypoints(results).astype(np.float32))

    np.save(kp_path, np.stack(keypoints, axis=0))   # (T, 225)
    cap.release()
    return True


def preprocess_dataset(
        annotation_csv : str,
        videos_dir     : str,
        frames_out     : str,
        kp_out         : str,
        split          : str = 'train',
        max_videos     : int = None,
):
    """
    Lance le preprocessing sur tous les clips d'un split.

    Args:
        annotation_csv : chemin vers train.csv / val.csv / test.csv
        videos_dir     : dossier contenant les .mp4
        frames_out     : dossier de sortie pour les frames
        kp_out         : dossier de sortie pour les keypoints
        split          : 'train' | 'val' | 'test'
        max_videos     : limite (utile pour debug)
    """
    os.makedirs(frames_out, exist_ok=True)
    os.makedirs(kp_out,     exist_ok=True)

    df = pd.read_csv(annotation_csv, sep='\t')
    if max_videos:
        df = df.head(max_videos)

    print(f'\n📦 Preprocessing {split} — {len(df)} clips')
    ok, fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        # How2Sign CSV columns: SENTENCE_ID, SENTENCE_NAME, START, END, SENTENCE
        clip_id   = str(row['SENTENCE_NAME'])
        video_path = os.path.join(videos_dir, f'{clip_id}.mp4')

        if not os.path.exists(video_path):
            fail += 1
            continue

        success = process_video(video_path, frames_out, kp_out, clip_id)
        if success:
            ok += 1
        else:
            fail += 1

    print(f'\n✅ Succès : {ok}  |  ❌ Échecs : {fail}')


# ── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    # Test sur une vidéo quelconque
    import sys
    if len(sys.argv) > 1:
        video = sys.argv[1]
        process_video(video, '/tmp/frames', '/tmp/kp', 'test_clip')
        frames = np.load('/tmp/frames/test_clip.npy')
        kp     = np.load('/tmp/kp/test_clip.npy')
        print(f'Frames shape  : {frames.shape}')   # (48, 224, 224, 3)
        print(f'Keypts shape  : {kp.shape}')        # (48, 225)
