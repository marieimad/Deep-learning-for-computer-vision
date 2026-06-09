# preprocessing.py
# offline feature extraction for how2sign — run once before training
#   - extracts rgb frames from mp4 videos -> .npy (T, 224, 224, 3)
#   - extracts mediapipe holistic keypoints -> .npy (T, 201)
#     201 = 25 body pose + 21 left hand + 21 right hand landmarks
#     coordinates stored in pixel space (not normalized)
# skips clips that are already processed

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print('warning: mediapipe not installed — keypoint extraction will not work')

MAX_FRAMES = 48
FRAME_SIZE = (224, 224)
KP_DIM     = 201   # 25*3 + 21*3 + 21*3


def extract_keypoints_mediapipe(video_path):
    """
    extract mediapipe holistic keypoints from a video file.
    returns (MAX_FRAMES, 201) — pixel space coordinates.
    zero-fills frames where landmarks are not detected.
    """
    if not MEDIAPIPE_AVAILABLE:
        return np.zeros((MAX_FRAMES, KP_DIM), dtype=np.float32)

    mp_holistic = mp.solutions.holistic
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((MAX_FRAMES, KP_DIM), dtype=np.float32)

    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width   = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    indices = np.linspace(0, max(total - 1, 0), MAX_FRAMES, dtype=int)

    results_list = []
    with mp_holistic.Holistic(static_image_mode=False,
                               model_complexity=1,
                               min_detection_confidence=0.5) as holistic:
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                results_list.append(np.zeros(KP_DIM, dtype=np.float32))
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result    = holistic.process(frame_rgb)

            kp = np.zeros(KP_DIM, dtype=np.float32)

            # pose landmarks (25 used out of 33, pixel coords)
            if result.pose_landmarks:
                for i, lm in enumerate(list(result.pose_landmarks.landmark)[:25]):
                    kp[i*3]   = lm.x * width
                    kp[i*3+1] = lm.y * height
                    kp[i*3+2] = lm.z * width   # depth relative to hip

            # left hand landmarks (21 landmarks)
            if result.left_hand_landmarks:
                offset = 25 * 3
                for i, lm in enumerate(result.left_hand_landmarks.landmark):
                    kp[offset + i*3]   = lm.x * width
                    kp[offset + i*3+1] = lm.y * height
                    kp[offset + i*3+2] = lm.z * width

            # right hand landmarks (21 landmarks)
            if result.right_hand_landmarks:
                offset = 25 * 3 + 21 * 3
                for i, lm in enumerate(result.right_hand_landmarks.landmark):
                    kp[offset + i*3]   = lm.x * width
                    kp[offset + i*3+1] = lm.y * height
                    kp[offset + i*3+2] = lm.z * width

            results_list.append(kp)

    cap.release()
    return np.stack(results_list, axis=0)   # (MAX_FRAMES, 201)


def extract_frames_clip(video_path):
    """
    uniformly subsample MAX_FRAMES from a video.
    returns (MAX_FRAMES, 224, 224, 3) float32 in [0, 1].
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
    return np.stack(frames).astype(np.float32) / 255.0   # normalize to [0,1]


def process_clip(clip_id, video_path, frames_out, kp_out):
    """process one clip — skip if already done."""
    fp = os.path.join(frames_out, f'{clip_id}.npy')
    kp = os.path.join(kp_out,    f'{clip_id}.npy')

    if os.path.exists(fp) and os.path.exists(kp):
        return True

    if not os.path.exists(video_path):
        return False

    if not os.path.exists(fp):
        frames = extract_frames_clip(video_path)
        np.save(fp, frames)

    if not os.path.exists(kp):
        keypoints = extract_keypoints_mediapipe(video_path)
        np.save(kp, keypoints)

    return True


def preprocess_dataset(annotation_csv, videos_dir, frames_out, kp_out,
                       split='train', max_clips=None):
    os.makedirs(frames_out, exist_ok=True)
    os.makedirs(kp_out,     exist_ok=True)

    df = pd.read_csv(annotation_csv, sep='\t')
    if max_clips:
        df = df.head(max_clips)

    print(f'\npreprocessing {split} — {len(df)} clips')
    ok, fail = 0, 0

    for _, row in tqdm(df.iterrows(), total=len(df)):
        clip_id    = str(row['SENTENCE_NAME'])
        video_path = os.path.join(videos_dir, f'{clip_id}.mp4')

        success = process_clip(clip_id, video_path, frames_out, kp_out)
        if success: ok   += 1
        else:       fail += 1

    print(f'\ndone: {ok} ok | {fail} failed')
