<div align="center">

# CLIP-Based Person Re-Identification (ReID)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch ≥2.0](https://img.shields.io/badge/pytorch-≥2.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**v1.0 — Stable Release** 🎉

A CLIP-based person re-identification pipeline covering training, evaluation, and online inference.

[**Quick Start**](#quick-start) • [**Training**](#training) • [**Evaluation**](#testing--evaluation) • [**Online Inference**](#online-inference)

</div>

---

## ✨ Highlights

- **CLIP-based ReID pipeline**: Covers the full lifecycle from training a contrastive image encoder to online inference.
- **YOLOv8 + CLIP inference**: Combines YOLOv8 person detection with CLIP feature matching for end-to-end retrieval.
- **Modular & documented**: Fully documented codebase with standardized English comments and docstrings (Google style).
- **DDP training**: Supports both single-GPU and multi-GPU (DDP) training out of the box.
- **Deployable inference**: The `online/` package is self-contained and deployable without the training source code.

---

## 📋 Table of Contents

- [Highlights](#-highlights)
- [Updates](#-updates)
- [Overview](#-overview)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Download Pre-trained Weights](#download-pre-trained-weights)
  - [Training](#training)
  - [Testing / Evaluation](#testing--evaluation)
  - [Online Inference](#online-inference)
- [Results](#-results)
- [Model Zoo](#-model-zoo)
- [Feature Enhancement Modules](#-feature-enhancement-modules)
- [Similarity Formulations](#-similarity-formulations)
- [Evaluation Metrics](#-evaluation-metrics)
- [Citation](#-citation)
- [Acknowledgments](#-acknowledgments)
- [Contact](#-contact)

---

## 📣 Updates

- **[2025.08]** Initial release — complete training & evaluation pipeline.
- **[2026.08]** Re-organized project structure; added self-contained `online/` inference module with standardized English docstrings; added architecture diagrams and detailed README.
- **[2026.09]** Refined README to strictly match the actual codebase: corrected `model/cluster.py` description, removed non-existent CLI flags, fixed deployment paths, and expanded the API reference for `ReIDInfer`.
- **[2026.09]** Added `experiments/` directory to track training runs with configs, logs, and TensorBoard events.
- **[2026.09]** Added baseline training results (ViT-B/32, 50 epochs, **R1=98.4%**, **R5=100.0%**, **R10=100.0%**).

---

## 🎯 Overview

Person Re-Identification aims to match a target person image against a gallery of candidate images, typically captured from different camera views. This project provides two unified workflows:

1. **Training & Evaluation** — Fine-tune a CLIP visual encoder with an InfoNCE contrastive loss on paired person images. Evaluate with standard ReID metrics (Rank-1, Rank-5, Rank-10, RSum). Supports single-GPU and DDP multi-GPU training, automatic checkpointing, TensorBoard logging, and warm-start resumption.
2. **Online Inference** — Deploy the `ReIDInfer` class (inside `online/module_reid/`) that detects persons in a scene image with **YOLOv8**, encodes each cropped person with the fine-tuned CLIP encoder, and returns the best match annotated on the original image.

### Key Features

| Feature | Description |
|---------|-------------|
| **Backbone** | OpenAI CLIP (`VisionTransformer` or `ModifiedResNet`) with pretrained weight loading and positional-embedding interpolation for arbitrary input resolutions |
| **Loss** | InfoNCE contrastive loss with learnable temperature scaling (`logit_scale = 1 / temperature`) |
| **Detection** | YOLOv8 (`classes=[0]`, person-only) for real-time detection during inference |
| **Data Augmentation** | `RandomHorizontalFlip`, `RandomCrop`, `RandomErasing` (optional via `--img_aug`) |
| **Training** | Single-GPU or Multi-GPU (DDP via `torch.distributed.launch`) with automatic `best`/`last` checkpointing |
| **Evaluation** | CMC metrics computed on GPU-parallel embedding extraction |
| **Config** | YAML-based configuration snapshots saved alongside each checkpoint; all args loaded as `EasyDict` for attribute-style access |
| **Online Mode** | Standalone `module_reid` package: no dependency on the training codebase except PyTorch and Ultralytics |

---

## 🏗️ Architecture

### Training Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  RSTPReid Dataset                                                          │
│  ├── train_pairs.json    (id, tar_path, can_path)                         │
│  └── test_pairs.json     (id, tar_path, can_path)                         │
│       │                                                                    │
│       ▼                                                                    │
│  ImageTextDataset ──► build_dataloader() ──► DataLoader                   │
│       │                                                                    │
│  ┌────┴────────────────────────────────────────┐                          │
│  │  Transform (training)                        │                          │
│  │  Resize(384,128) → H-Flip → Pad(10)         │                          │
│  │  → RandomCrop → Normalize(CLIP mean/std)    │                          │
│  │  → RandomErasing (optional)                  │                          │
│  └─────────────────────────────────────────────┘                          │
│       │                                                                    │
│  ┌────┴────────────────────────────────────────┐                          │
│  │  per-batch forward                           │                          │
│  │                                              │                          │
│  │  tar_img ──► CLIP.visual ──► L2 norm ──┐    │                          │
│  │  can_img ──► CLIP.visual ──► L2 norm ──┼───►│───► cos_sim matrix [B,B]│
│  │                                        │    │                          │
│  │  Diagonal = positive pairs             │    │───► InfoNCE loss         │
│  │  Off-diagonal = negatives              │    │                          │
│  └─────────────────────────────────────────────┘                          │
│       │                                                                    │
│       ▼                                                                    │
│  Optimizer (Adam/AdamW/SGD) + LR Scheduler (Step/Cosine/...)              │
│       │                                                                    │
│       ▼                                                                    │
│  Checkpointer saves best.pth (by RSum) and last.pth                       │
│       │                                                                    │
│       ▼                                                                    │
│  Evaluator: extract all test embeddings → similarity matrix                │
│             → rank sort → R1 / R5 / R10 / RSum / MdR / MnR                │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Online Inference Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Online Inference Pipeline                          │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Target Image                                                              │
│       │                                                                    │
│       ▼                                                                    │
│  _load_target() ──► Image.open / PIL ──► _transform ──► model.encode_image│
│       │                                                                    │
│       └──► L2-normalized feature (cached for str/bytes paths)             │
│                                                                            │
│  Scene Image (candidate_image)                                             │
│       │                                                                    │
│       ▼                                                                    │
│  detect_model.predict(img_bgr, classes=[0]) ──► YOLO boxes                │
│       │                                                                    │
│       ▼                                                                    │
│  For each box with conf > conf_thresh:                                     │
│    crop ──► cv2.cvtColor(BGR→RGB) ──► Image.fromarray ──► _transform      │
│       │                                                                    │
│       ▼                                                                    │
│  Stack all crops ──► encode_image ──► L2 normalize                        │
│       │                                                                    │
│       ▼                                                                    │
│  sims = target_feat @ crop_feats.T  (cosine similarity)                    │
│       │                                                                    │
│       ▼                                                                    │
│  best_idx = argmax(sims)                                                   │
│       │                                                                    │
│       ├── best_sim > 0.8 ──► Draw blue box + label ──► status=True        │
│       └── best_sim ≤ 0.8 ──► No drawing           ──► status=False        │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Network Components

- **Visual Encoder** — `CLIP` class wraps either `VisionTransformer` (ViT-B, ViT-L) or `ModifiedResNet` (RN50, RN101, etc.). Pretrained weights are downloaded automatically from OpenAI on first use (cached in `~/.cache/clip/`). Supports positional-embedding bilinear interpolation when `img_size` differs from the pretrained resolution.
- **Re-ID Head** — The `[CLS]` token from the final ViT layer (or pooled ResNet features from `AttentionPool2d`) is L2-normalized and treated as the person embedding. No additional projection head is added; the encoder output itself serves as the embedding.
- **Loss** — `Objective` in `model/objectives.py` computes InfoNCE over a batch: diagonal entries are positive pairs (same identity), off-diagonal entries are negatives. A `logit_scale = 1 / temperature` scalar scales similarities before softmax.
- **Detector** — `YOLO(self.yolo_model)` from Ultralytics, called with `classes=[0]` to restrict to person class and `verbose=False` to suppress console output. Confidence threshold is configurable via `conf_thresh`.

---

## 📁 Project Structure

```
reid/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── bash.sh                             # Quick-launch training script (ViT-B/32, 10 epochs)
│
├── main.py                             # Training entry point
├── infer.py                            # Standalone evaluation script
│
├── model/                              # Neural network definitions
│   ├── __init__.py                     #   Re-exports build_model
│   ├── build.py                        #   ReID(nn.Module) wrapper: loads CLIP + logit_scale
│   ├── clip_model.py                   #   Complete CLIP implementation (ViT & ResNet)
│   ├── cluster.py                      #   Patch token compression & clustering modules
│   └── objectives.py                   #   InfoNCE contrastive loss with L2 normalization
│
├── datasets/                           # Data loading & preprocessing
│   ├── __init__.py                     #   Re-exports build_dataloader
│   ├── bases.py                        #   ImageTextDataset: PyTorch Dataset
│   ├── build.py                        #   build_dataloader(), build_transforms(), collate()
│   ├── rstpreid.py                     #   RSTPReid: JSON annotation parser
│   ├── sampler.py                      #   RandomIdentitySampler: P × K identity-based batching
│   ├── sampler_ddp.py                  #   RandomIdentitySampler_DDP: DDP-compatible version
│   └── preprocessing.py                #   Image preprocessing utilities
│
├── processor/                          # Training loop execution
│   └── processor.py                    #   do_train(), do_inference()
│
├── solver/                             # Optimization
│   ├── build.py                        #   Optimizer factory (SGD / Adam / AdamW)
│   └── lr_scheduler.py               #   Step / Cosine / Polynomial / Warmup schedulers
│
├── utils/                              # Shared utilities
│   ├── checkpoint.py                   #   Checkpointer: save/load/resume
│   ├── comm.py                         #   DDP communication helpers
│   ├── iotools.py                      #   YAML config read/write, image I/O
│   ├── logger.py                       #   setup_logger(): file + console logging
│   ├── meter.py                        #   AverageMeter: running average for loss tracking
│   ├── metrics.py                      #   CMC metrics: R1, R5, R10, R50, MdR, MnR, RSum
│   ├── options.py                      #   Full argument parser
│   └── simple_tokenizer.py           #   CLIP BPE tokenizer
│
├── online/                             # Online Inference (Deployable Package)
│   ├── main.py                         #   Demo script
│   └── module_reid/                    #   Self-contained package
│       ├── __init__.py                 #     Exports ReIDInfer
│       ├── reid_infer.py               #     ReIDInfer class
│       ├── model/                      #     Lightweight CLIP model
│       └── utils/                      #     Minimal checkpoint & config loaders
│
├── data/                               # Data Assets
│   └── bpe_simple_vocab_16e6.txt.gz    #   CLIP BPE tokenizer vocabulary
│
├── RSTPReid/                           # Example dataset directory
│   ├── gen_pairs.py                    #   Script to generate train/test JSON pair files
│   ├── train_pairs.json                #   Training annotations
│   ├── test_pairs.json                 #   Testing annotations
│   └── imgsXXXX/                       #   Image folders
│
└── experiments/                        # Training run outputs
    └── 20260908_164014_baseline_reid/  #   Example run: ViT-B/32 baseline
        ├── configs.yaml                #   YAML config snapshot
        ├── train_log.txt               #   Training logs (loss, lr, ETA)
        └── events.out.tfevents.*       #   TensorBoard event file
```

---

## 🚀 Quick Start

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/OPA067/ReID.git
cd ReID

# 2. Create a conda environment
conda create -n reid python=3.12
conda activate reid

# 3. Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `ultralytics>=8.0.0` (YOLOv8)
- `pyyaml>=6.0`, `easydict>=1.10` (config management)
- `tqdm`, `prettytable`, `tensorboard` (logging & tables)

### Dataset Preparation

The project uses an **image-to-image ReID dataset** with JSON annotations. Each entry is a triplet:

```json
{"id": 0, "tar_path": "imgs0000/person_a.jpg", "can_path": "imgs0001/person_a.jpg"}
```

The `RSTPReid` class (`datasets/rstpreid.py`) expects the following layout:

```
RSTPReid/
├── train_pairs.json        # List of (id, tar_path, can_path) for training
├── test_pairs.json         # List of (id, tar_path, can_path) for testing
└── imgsXXXX/               # Image folders containing the actual images
    ├── person_001.jpg
    └── ...
```

To create your own dataset, place images into folders and run:

```bash
cd RSTPReid
python gen_pairs.py         # Generates train_pairs.json + test_pairs.json
```

> **Note:** Image paths inside the JSON are **relative to the `RSTPReid/` directory**.

### Download Pre-trained Weights

CLIP pretrained weights are downloaded **automatically** on first use (cached in `~/.cache/clip/`). Alternatively, download manually:

| Model | Architecture | Download | Notes |
|:-----:|:-------------|:---------|:------|
| **RN50** | ResNet-50 | [RN50.pt](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) | Fast, `[CLS]` embedding |
| **RN101** | ResNet-101 | [RN101.pt](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt) | Fast, `[CLS]` embedding |
| **RN50x4** | ResNet-50×4 | [RN50x4.pt](https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt) | Higher capacity |
| **RN50x64** | ResNet-50×64 | [RN50x64.pt](https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt) | Highest ResNet capacity |
| **ViT-B/32** | ViT-Base, patch 32 | [ViT-B-32.pt](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) | **Recommended baseline** |
| **ViT-B/16** | ViT-Base, patch 16 | [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | Finer spatial patches |
| **ViT-L/14** | ViT-Large, patch 14 | [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | Best accuracy, largest |

For online inference, you also need:
- A **fine-tuned ReID checkpoint** (`best.pth`) from training
- A **YOLOv8 model** (`yolov8l.pt` or variant) for person detection

### Training

#### Single-GPU Quick Start

```bash
bash bash.sh
```

`bash.sh` runs with: ViT-B/32, batch size 16, RSTPReid dataset, 10 epochs.

#### Manual Training (Full Control)

```bash
CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --name baseline_reid \
    --batch_size 16 \
    --root_dir /path/to/reid \
    --output_dir experiments \
    --dataset_name RSTPReid \
    --loss_names reid \
    --pretrain_choice ViT-B/32 \
    --img_size '(384, 128)' \
    --stride_size 16 \
    --temperature 0.02 \
    --lr 1e-5 \
    --optimizer Adam \
    --lrscheduler cosine \
    --num_epoch 50 \
    --log_period 100 \
    --eval_period 1 \
    --img_aug
```

#### Multi-GPU DDP

```bash
# Using torchrun (recommended for PyTorch ≥1.9)
torchrun --nproc_per_node=4 \
    main.py \
    --batch_size 64 \
    --num_epoch 50 \
    --pretrain_choice ViT-B/32
```

> **Note:** DDP is **auto-detected** via the `WORLD_SIZE` environment variable.

#### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | `baseline` | Experiment name (output directory) |
| `--pretrain_choice` | `ViT-B/32` | CLIP variant |
| `--img_size` | `(384, 128)` | Input resolution `(H, W)` |
| `--stride_size` | `16` | Patch stride for ViT |
| `--temperature` | `0.02` | InfoNCE temperature |
| `--batch_size` | `32` | Training batch size per GPU |
| `--num_epoch` | `50` | Total training epochs |
| `--lr` | `1e-5` | Base learning rate |
| `--optimizer` | `Adam` | `SGD`, `Adam`, or `AdamW` |
| `--lrscheduler` | `cosine` | `step`, `exp`, `poly`, `cosine`, `linear` |
| `--img_aug` | `False` | Enable `RandomCrop` + `RandomErasing` |
| `--sampler` | `random` | `random` or `identity` (PK sampling) |
| `--num_instance` | `4` | Instances per identity for identity sampler |
| `--log_period` | `100` | Log interval (iterations) |
| `--eval_period` | `1` | Evaluate every N epochs |
| `--resume` | `False` | Resume from checkpoint |
| `--resume_ckpt_file` | `""` | Path to checkpoint for resumption |

#### Training Outputs

Each run creates a timestamped directory:

```
experiments/
└── 20260908_164014_baseline_reid/
    ├── configs.yaml          # Full config snapshot
    ├── best.pth              # Best checkpoint (highest RSum)
    ├── last.pth              # Final checkpoint
    ├── train_log.txt         # Training logs
    └── events.out.tfevents.* # TensorBoard events
```

### Testing / Evaluation

```bash
python infer.py \
    --config_file experiments/20260908_164014_baseline_reid/configs.yaml \
    --gpu 0
```

`infer.py` steps:
1. Parses CLI args and loads YAML config
2. Reconstructs the test dataloader
3. Instantiates model and loads `best.pth`
4. Calls `Evaluator.eval()` and prints CMC table

**Example output:**
```
+-------+------+------+------+-------+------+-------+
| item  |  R1  |  R5  |  R10 |  RSum |  MdR |  MnR  |
+-------+------+------+------+-------+------+-------+
| sims  | 98.4 | 100.0| 100.0| 298.4 |  1.0 |  1.0  |
+-------+------+------+------+-------+------+-------+
```

### Online Inference

The `online/` directory contains a **self-contained deployment package** with no dependency on the training codebase.

```python
from online.module_reid import ReIDInfer

# Initialize (loads CLIP + YOLOv8)
infer = ReIDInfer()

# Run inference
status, result, box, score = infer(
    target_image="path/to/target.jpg",
    candidate_image="path/to/scene.jpg"
)

# status: True if match found (score > 0.8)
# result: dict with result_img, best_sim, best_box, etc.
# box: [x1, y1, x2, y2] of best match
# score: similarity score (0-1)
```

**Package Layout:**
```
online/
├── main.py                         # Demo script
└── module_reid/                    # Self-contained package
    ├── __init__.py                 #   Exports ReIDInfer
    ├── reid_infer.py               #   Core inference engine
    ├── model/                      #   Lightweight CLIP model
    └── utils/                      #   Checkpoint & config loaders
```

**Return Value:**

| Position | Type | Description |
|----------|------|-------------|
| 0 | `bool` | `True` if `best_sim > 0.8` (confident match) |
| 1 | `dict` | `result_img`, `best_sim`, `best_box`, `boxes`, `scores` |
| 2 | `list` | `[x1, y1, x2, y2]` of best match |
| 3 | `float` | Best similarity score (0-1) |

---

## 📊 Results

### Baseline Training Results

Model: **ViT-B/32** | Dataset: **RSTPReid** | Epochs: **50**

| Epoch | R1 (%) | R5 (%) | R10 (%) | RSum | MdR | MnR |
|:-----:|:------:|:------:|:-------:|:----:|:---:|:---:|
| 0 (Zero-shot) | 3.7 | 6.1 | 7.6 | 17.4 | 781.0 | 1166.9 |
| 1 | 46.3 | 67.5 | 75.4 | 189.2 | 2.0 | 31.7 |
| 5 | 64.6 | 87.2 | 92.8 | 244.6 | 1.0 | 4.2 |
| 10 | 73.0 | 92.9 | 96.5 | 262.4 | 1.0 | 2.3 |
| 20 | 84.4 | 98.2 | 99.5 | 282.0 | 1.0 | 1.4 |
| 30 | 91.0 | 99.5 | 99.9 | 290.4 | 1.0 | 1.2 |
| 40 | 97.0 | 100.0 | 100.0 | 297.0 | 1.0 | 1.0 |
| **50 (Best)** | **98.4** | **100.0** | **100.0** | **298.4** | **1.0** | **1.0** |

**Training Config:**
- Backbone: ViT-B/32 (87M params)
- Batch size: 32, Learning rate: 1e-5 (Adam, cosine schedule)
- Image size: (384, 128), Temperature: 0.02
- Dataset: 41,010 training / 4,101 testing samples

---

## 🤖 Model Zoo

### CLIP Variants

| Model | Params | Output | Speed | Memory | Recommended Use |
|-------|--------|--------|-------|--------|-----------------|
| `RN50` | ~38M | `[CLS]` 1×D | Fast | Low | Edge / resource-constrained |
| `RN101` | ~56M | `[CLS]` 1×D | Fast | Low | Slightly better than RN50 |
| `RN50x4` | ~87M | `[CLS]` 1×D | Medium | Medium | Balanced capacity |
| `RN50x64` | ~303M | `[CLS]` 1×D | Slow | High | Best ResNet accuracy |
| `ViT-B/32` | ~87M | `[CLS]` + Patch tokens | Medium | Medium | **Recommended baseline** |
| `ViT-B/16` | ~86M | `[CLS]` + Patch tokens | Medium-Low | Medium | Finer spatial detail |
| `ViT-L/14` | ~303M | `[CLS]` + Patch tokens | Slow | High | Highest accuracy |

### Input Resolution

The default `--img_size '(384, 128)'` corresponds to the ReID-standard aspect ratio (3:1). For ViT:
- `num_x = (128 - patch_size) // stride_size + 1`
- `num_y = (384 - patch_size) // stride_size + 1`

Positional embeddings are bilinearly interpolated when loading weights with different native resolution.

---

## 🔧 Feature Enhancement Modules

The codebase includes `model/cluster.py`, a collection of **patch-token manipulation modules** for ViT variants:

### 1. PCM — Progressive Clustering Module
Compresses patch tokens via **DPC-KNN (Density Peak Clustering with K-Nearest Neighbors)**:
1. Token features → 1D conv + LayerNorm
2. Linear layer → per-token importance score
3. Exponential weights (masked for invalid tokens)
4. `cluster_dpc_knn()` → `ceil(N * sample_ratio)` clusters
5. `merge_tokens()` → aggregated representatives

### 2. Att_PCM — Attention with Spatial Reduction
Attention module with **key/value spatial reduction** (`sr_ratio > 1`):
- Conv2d + LayerNorm (`use_sr_layer=True`)
- Average pooling (`use_sr_layer=False`)

Token scores from PCM are injected as positional confidence bias.

### 3. Att_Block_Patch — Transformer Block
Standard transformer block with:
- Pre-LayerNorm
- Residual connection + `DropPath` (stochastic depth)

> These modules are **pluggable building blocks** and not wired into the default `ReID` model.

---

## 📐 Similarity Formulations

Person similarity is computed as **cosine similarity** between L2-normalized feature vectors:

$$\mathrm{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^{\top} \mathbf{v}}{\|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2} \in [-1, 1]$$

The codebase supports four feature-alignment strategies:

| Strategy | Description | Fine-tuning |
|----------|-------------|:-----------:|
| **Zero-shot `[CLS]`** | Single global token from pretrained CLIP | ✗ |
| **Fine-tuned `[CLS]`** | End-to-end fine-tuning with InfoNCE | ✓ |
| **Zero-shot `[CLS]` + `[Patch]`** | Global + mean patch pooling | ✗ |
| **Fine-tuned `[CLS]` + `[Patch]`** | Global + learnable aggregation (PCM) | ✓ |

For fine-tuned variants, the InfoNCE loss is:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \left[ \log \frac{e^{\tau S_{ii}}}{\sum_{j=1}^{B} e^{\tau S_{ij}}} + \log \frac{e^{\tau S_{ii}}}{\sum_{j=1}^{B} e^{\tau S_{ji}}} \right]$$

where $\tau = 1 / \text{temperature}$ is the learnable temperature, and diagonal entries $S_{ii}$ are positive pairs.

---

## 📈 Evaluation Metrics

| Metric | Symbol | Description |
|--------|:------:|-------------|
| **Rank-1** | R1 | % queries where ground truth is ranked 1st |
| **Rank-5** | R5 | % queries where ground truth is in top-5 |
| **Rank-10** | R10 | % queries where ground truth is in top-10 |
| **Rank-50** | R50 | % queries where ground truth is in top-50 |
| **RSum** | — | `R1 + R5 + R10`, validation criterion for best checkpoint |
| **MdR** | — | Median rank of correct match |
| **MnR** | — | Mean rank of correct match |

---

## 📚 Citation

If you find this work useful, please cite:

```bibtex
@software{reid2026,
  author       = {OPA067},
  title        = {CLIP-Based Person Re-Identification},
  year         = {2026},
  url          = {https://github.com/OPA067/ReID},
  version      = {1.0}
}
```

---

## 🙏 Acknowledgments

- **[OpenAI CLIP](https://github.com/openai/CLIP)** — Pretrained vision-language representations and tokenizer.
- **[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)** — Real-time object detection for the online inference pipeline.

---

## 📬 Contact

📧 **Email:** [xinl067@193.com](mailto:xinl067@193.com)

🐛 **Issues:** Please open a [GitHub Issue](https://github.com/OPA067/ReID/issues) with a minimal reproduction script and the output of `python -m torch.utils.collect_env`.

---

<div align="center">
  <sub>Built for the person Re-ID community. ⭐ Star this repo if you find it helpful!</sub>
</div>
