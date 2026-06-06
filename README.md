# Two-Stream Sign Language Translation via DINOv2 RGB Features and Keypoint Transformers

**CS231N Final Project — Marie Imad & Aaron Sequeira, Stanford University**

## Overview

This project implements a gloss-free two-stream architecture for American Sign Language (ASL) to English translation on the [How2Sign](https://how2sign.github.io/) dataset. The model fuses a frozen DINOv2 visual encoder with a MediaPipe keypoint Transformer before decoding with a pretrained BART model.

---

## Architecture

```
ASL video
  ├── RGB stream:   48 frames → frozen DINOv2 ViT-S/14 → 2-layer LSTM → h_R (512-dim)
  └── KP stream:    48 × 201-dim MediaPipe landmarks → Linear → Transformer encoder → h_K (512-dim)
                                          ↓
                              Cross-Attention Fusion
                                          ↓
                              Linear 1024 → 768
                                          ↓
                              BART-base decoder → English text
```

The BART encoder is frozen; only the decoder cross-attention, self-attention, and output projection are fine-tuned.

---

## Repository Structure

```
├── model.py           # Two-stream architecture
├── dataset.py         # How2Sign PyTorch Dataset and DataLoaders
├── train.py           # Training loop with early stopping and checkpoint resumption
├── preprocessing.py   # Offline RGB frame and MediaPipe keypoint extraction
└── README.md
```

---

## File Descriptions

### `model.py`
Defines the full model architecture:
- **`RGBStream`**: processes 48 frames through frozen DINOv2 ViT-S/14 (CLS token), then a 2-layer LSTM to produce a 512-dim temporal summary `h_R`.
- **`KeypointStream`**: projects 201-dim MediaPipe vectors through an MLP, then a 2-layer Transformer encoder (8 heads, FFN dim 2048) with learned positional embeddings, mean-pooled to `h_K`.
- **`CrossAttentionFusion`**: RGB queries keypoint representations via multi-head cross-attention, followed by LayerNorm and a linear projection to BART's 768-dim space.
- **`SignLanguageTranslator`**: wraps the two streams, fusion, and BART decoder into a single model with a `forward()` (training) and `generate()` (inference) method.

### `dataset.py`
- **`How2SignDataset`**: loads precomputed `.npy` RGB frames (T×3×224×224, ImageNet-normalized) and `.npy` keypoints (T×201). Falls back to zero tensors for clips without extracted frames. Tokenizes captions with `BartTokenizer`. Applies horizontal flip augmentation during training.
- **`build_dataloaders`**: returns train/val/test DataLoaders with configurable batch size.

### `train.py`
Full training pipeline reflecting our design choices:
- We chose **AdamW** as our optimizer because of its strong performance on fine-tuning pretrained language models, with a low learning rate of 1e-6 to avoid catastrophic forgetting of BART's pretrained weights.
- We chose a **batch size of 3** based on T4 GPU memory constraints (16GB VRAM) with sequences of 48 frames at 224×224.
- We chose **gradient clipping at norm 1.0** to stabilize training with small batches.
- We chose **ReduceLROnPlateau on BLEU** (factor=0.5, patience=2) rather than a fixed schedule, so the learning rate adapts to translation quality directly.
- We chose **early stopping with patience=5** to prevent overfitting while allowing enough time for the model to learn sign-to-content mappings.
- We implemented **checkpoint resumption** (saving optimizer state, epoch, best BLEU, patience count) to handle cloud preemptions on Modal.

### `preprocessing.py`
Offline feature extraction (run once before training):
- Reads OpenPose JSON files per frame, assembles 201-dim vectors (75 pose + 63 left hand + 63 right hand landmarks), uniformly subsamples to 48 frames, saves as `.npy`.
- Extracts frames from MP4 videos with OpenCV, resizes to 224×224, normalizes to [0,1], saves as `.npy`.
- Skips clips that are already preprocessed.

---

## Design Decisions

**Why DINOv2?** We chose DINOv2 ViT-S/14 as our visual backbone because its self-supervised pretraining on 142M images produces rich patch-level features that transfer well to video understanding tasks. We froze it entirely to avoid catastrophic forgetting on the limited sign language data.

**Why MediaPipe?** Prior work (Tarrès et al.) showed that feeding raw MediaPipe coordinates as a flat 1D array without temporal encoding collapses to near-zero BLEU. We addressed this by pairing MediaPipe with a dedicated Transformer encoder that learns temporal structure across the 48-frame sequence.

**Why BART?** Müller et al. showed that BART-base provides strong initialization for low-resource translation decoders. We froze the encoder and only fine-tuned the decoder to preserve the pretrained language model while adapting it to our visual inputs.

**Why T=48 frames?** We chose 48 frames as a balance between temporal coverage (~4s clips at 24fps) and memory cost at batch size 3.

**Why cross-attention fusion?** Simple concatenation discards the interaction between modalities. Cross-attention lets the RGB stream query the keypoint representations, allowing the model to focus on the keypoint frames that are most relevant to the visual context.

---

## Novel Metric: rBLEU

We introduced **rBLEU**, which removes high-frequency instructional filler words ("and", "you", "to", "the", "going to", etc.) from both prediction and reference before computing BLEU-4. This was motivated by the repetitive instructional register of How2Sign, where a model can achieve moderate BLEU by generating plausible filler phrases without translating any sign content. The rBLEU/BLEU ratio directly quantifies filler-word dependence.

---

## Setup

```bash
pip install torch torchvision transformers evaluate sacrebleu mediapipe opencv-python pandas numpy
```

**Training:**
```bash
python train.py
```

**Preprocessing** (run once):
```bash
python preprocessing.py
```

---

## Hardware

All training runs used a single NVIDIA T4 GPU (16GB VRAM) via [Modal](https://modal.com/) cloud, ~4.25 hours per epoch on 31k clips.

---

## Note on AI Assistance

This project used Claude (Anthropic) as a coding assistant. The core ideas, research directions, and architectural decisions were entirely our own: we reviewed the relevant literature, designed the two-stream architecture, chose the components (DINOv2, MediaPipe, BART), and defined the experimental setup including all hyperparameters. AI assistance was used to help implement parts of the code and to assist with report writing. The model architecture, ablation design, novel rBLEU metric, and all result analysis were conceived and driven by the authors.
