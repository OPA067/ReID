<div align="center">

# CLIP-Based Person Re-Identification (ReID)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch ≥2.0](https://img.shields.io/badge/pytorch-≥2.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**v1.0 — Stable Release**

</div>

- **CLIP-based ReID pipeline**: Covers the full lifecycle from training a contrastive image encoder to online inference.
- **YOLOv8 + CLIP inference**: Combines YOLOv8 person detection with CLIP feature matching for end-to-end retrieval.
- **Modular & documented**: Fully documented codebase with standardized English comments and docstrings (Google style).
- **DDP training**: Supports both single-GPU and multi-GPU (DDP) training out of the box.
- **Deployable inference**: The `online/` package is self-contained and deployable without the training source code.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Dataset Preparation](#dataset-preparation)
  - [Download Pre-trained Weights](#download-pre-trained-weights)
  - [Training](#training)
  - [Testing / Evaluation](#testing--evaluation)
  - [Online Inference](#online-inference)
- [Model Zoo](#model-zoo)
- [Feature Enhancement Modules](#feature-enhancement-modules)
- [Evaluation Metrics](#evaluation-metrics)
- [Updates](#updates)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## Overview

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

## Updates

- **[2025.08]** Initial release — complete training & evaluation pipeline.
- **[2026.08]** Re-organized project structure; added self-contained `online/` inference module with standardized English docstrings; added architecture diagrams and detailed README.
- **[2026.09]** Refined README to strictly match the actual codebase: corrected `model/cluster.py` description, removed non-existent CLI flags, fixed deployment paths, and expanded the API reference for `ReIDInfer`.

---

## Architecture

### Training Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Training Pipeline                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RSTPReid Dataset                                                            │
│  ├── train_pairs.json    (id, tar_path, can_path)                           │
│  └── test_pairs.json     (id, tar_path, can_path)                           │
│       │                                                                      │
│       ▼                                                                      │
│  ImageTextDataset ──► build_dataloader() ──► DataLoader                     │
│       │                                                                      │
│  ┌────┴────────────────────────────────────────┐                            │
│  │  Transform (training)                        │                            │
│  │  Resize(384,128) → H-Flip → Pad(10)         │                            │
│  │  → RandomCrop → Normalize(CLIP mean/std)    │                            │
│  │  → RandomErasing (optional)                  │                            │
│  └─────────────────────────────────────────────┘                            │
│       │                                                                      │
│  ┌────┴────────────────────────────────────────┐                            │
│  │  per-batch forward                           │                            │
│  │                                              │                            │
│  │  tar_img ──► CLIP.visual ──► L2 norm ──┐    │                            │
│  │  can_img ──► CLIP.visual ──► L2 norm ──┼───►│───► cos_sim matrix [B,B]  │
│  │                                        │    │                            │
│  │  Diagonal = positive pairs             │    │───► InfoNCE loss           │
│  │  Off-diagonal = negatives              │    │                            │
│  └─────────────────────────────────────────────┘                            │
│       │                                                                      │
│       ▼                                                                      │
│  Optimizer (Adam/AdamW/SGD) + LR Scheduler (Step/Cosine/...)                │
│       │                                                                      │
│       ▼                                                                      │
│  Checkpointer saves best.pth (by RSum) and last.pth                         │
│       │                                                                      │
│       ▼                                                                      │
│  Evaluator: extract all test embeddings → similarity matrix                  │
│             → rank sort → R1 / R5 / R10 / RSum / MdR / MnR                  │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### Online Inference Pipeline

```
┌────────────────────────────────────────────────────────────────────────────┐
│                         Online Inference Pipeline                            │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Target Image                                                                │
│       │                                                                      │
│       ▼                                                                      │
│  _load_target() ──► Image.open / PIL ──► _transform ──► model.encode_image │
│       │                                                                      │
│       └──► L2-normalized feature (cached for str/bytes paths)               │
│                                                                              │
│  Scene Image (candidate_image)                                               │
│       │                                                                      │
│       ▼                                                                      │
│  detect_model.predict(img_bgr, classes=[0]) ──► YOLO boxes                  │
│       │                                                                      │
│       ▼                                                                      │
│  For each box with conf > conf_thresh:                                       │
│    crop ──► cv2.cvtColor(BGR→RGB) ──► Image.fromarray ──► _transform        │
│       │                                                                      │
│       ▼                                                                      │
│  Stack all crops ──► encode_image ──► L2 normalize                          │
│       │                                                                      │
│       ▼                                                                      │
│  sims = target_feat @ crop_feats.T  (cosine similarity)                      │
│       │                                                                      │
│       ▼                                                                      │
│  best_idx = argmax(sims)                                                     │
│       │                                                                      │
│       ├── best_sim > 0.8 ──► Draw blue box + label ──► status=True          │
│       └── best_sim ≤ 0.8 ──► No drawing           ──► status=False          │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```

### Network Components

- **Visual Encoder** — `CLIP` class wraps either `VisionTransformer` (ViT-B, ViT-L) or `ModifiedResNet` (RN50, RN101, etc.). Pretrained weights are downloaded automatically from OpenAI on first use (cached in `~/.cache/clip/`). Supports positional-embedding bilinear interpolation when `img_size` differs from the pretrained resolution.
- **Re-ID Head** — The `[CLS]` token from the final ViT layer (or pooled ResNet features from `AttentionPool2d`) is L2-normalized and treated as the person embedding. No additional projection head is added; the encoder output itself serves as the embedding.
- **Loss** — `Objective` in `model/objectives.py` computes InfoNCE over a batch: diagonal entries are positive pairs (same identity), off-diagonal entries are negatives. A `logit_scale = 1 / temperature` scalar scales similarities before softmax.
- **Detector** — `YOLO(self.yolo_model)` from Ultralytics, called with `classes=[0]` to restrict to person class and `verbose=False` to suppress console output. Confidence threshold is configurable via `conf_thresh`.

---

## Project Structure

```
reid/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── bash.sh                             # Quick-launch training script (ViT-B/32, 10 epochs)
│
# ─── Core Training & Evaluation ──────────────────────────────────────────
├── main.py                             # Training entry point
│                                       #   - Parses args, sets up DDP
│                                       #   - Builds dataloaders, model, optimizer
│                                       #   - Runs do_train() loop
│                                       #   - Loads best.pth and runs final do_inference()
│
├── infer.py                            # Standalone evaluation script
│                                       #   - Loads YAML config from a training run
│                                       #   - Reconstructs test loader and model
│                                       #   - Loads best.pth and prints CMC table
│
├── model/                              # Neural network definitions
│   ├── __init__.py                     #   Re-exports build_model
│   ├── build.py                        #   ReID(nn.Module) wrapper: loads CLIP + logit_scale
│   ├── clip_model.py                   #   Complete CLIP implementation:
│   │                                   #     - Bottleneck, AttentionPool2d, ModifiedResNet
│   │                                   #     - LayerNorm, QuickGELU, ResidualAttentionBlock
│   │                                   #     - Transformer, VisionTransformer
│   │                                   #     - CLIP class (image + text encoders)
│   │                                   #     - Pretrained weight download & position embed interpolation
│   ├── cluster.py                      #   Patch token compression & clustering:
│   │                                   #     - PCM (Progressive Clustering Module) via DPC-KNN
│   │                                   #     - Att_PCM (attention with spatial reduction)
│   │                                   #     - Att_Block_Patch (transformer block using Att_PCM)
│   │                                   #     - TokenConv, token2map / map2token helpers
│   └── objectives.py                   #   InfoNCE contrastive loss with L2 normalization
│
├── datasets/                           # Data loading & preprocessing
│   ├── __init__.py                     #   Re-exports build_dataloader
│   ├── bases.py                        #   ImageTextDataset: PyTorch Dataset for (id, tar_img, can_img)
│   ├── build.py                        #   build_dataloader(), build_transforms(), collate()
│   │                                   #   Augmentation pipeline, identity/random sampler selection
│   ├── rstpreid.py                     #   RSTPReid: JSON annotation parser, generates train/test splits
│   ├── sampler.py                      #   RandomIdentitySampler: P × K identity-based batching
│   ├── sampler_ddp.py                  #   RandomIdentitySampler_DDP: DDP-compatible version
│   │                                   #   with shared random seed and per-rank index partitioning
│   └── preprocessing.py                #   Image preprocessing utilities
│
├── processor/                          # Training loop execution
│   └── processor.py                    #   do_train(): epoch loop with loss, logging, TensorBoard
│                                       #   do_inference(): evaluator wrapper for final test
│
├── solver/                             # Optimization
│   ├── build.py                        #   Optimizer factory (SGD / Adam / AdamW)
│   └── lr_scheduler.py               #   Step / Cosine / Polynomial / Warmup schedulers
│
├── utils/                              # Shared utilities
│   ├── checkpoint.py                   #   Checkpointer: save/load/resume model, optimizer, scheduler
│   │                                   #   + state-dict alignment (strip "module." prefix, fuzzy key matching)
│   ├── comm.py                         #   DDP communication helpers: get_rank, synchronize, get_world_size
│   ├── iotools.py                      #   YAML config read/write, image I/O (read_image, read_json)
│   ├── logger.py                       #   setup_logger(): file + console logging for train/test
│   ├── meter.py                        #   AverageMeter: running average for loss tracking
│   ├── metrics.py                      #   CMC metrics: R1, R5, R10, R50, MdR, MnR, RSum
│   │                                   #   + Evaluator class for batched embedding extraction
│   ├── options.py                      #   Full argument parser with defaults for all hyperparameters
│   └── simple_tokenizer.py           #   CLIP BPE tokenizer (text-side compatibility)
│
# ─── Online Inference (Deployable Package) ──────────────────────────────
├── online/
│   ├── main.py                         #   Demo script: runs ReIDInfer on a single image pair
│   │                                   #   Saves annotated output to output.jpg
│   └── module_reid/                    #   Self-contained package (no dependency on training code)
│       ├── __init__.py                 #     Exports ReIDInfer
│       ├── reid_infer.py               #     ReIDInfer class:
│       │                               #       - Loads YAML config + best.pth + YOLOv8 weights
│       │                               #       - YOLO detection → CLIP encoding → cosine similarity
│       │                               #       - Caches target features (path-based key)
│       │                               #       - Draws bounding box + score label on BGR image
│       ├── model/
│       │   ├── __init__.py             #       Re-exports build_model
│       │   ├── build.py                #       Lightweight ReID wrapper (identical API to root model/)
│       │   └── clip_model.py           #       Pruned CLIP (image encoder only, same as root)
│       └── utils/
│           ├── checkpoint.py           #       Minimal checkpoint loader (load file → load model)
│           └── iotools.py              #       YAML config loader returning EasyDict
│
# ─── Data Assets ─────────────────────────────────────────────────────────
├── data/
│   └── bpe_simple_vocab_16e6.txt.gz    # CLIP BPE tokenizer vocabulary (used by simple_tokenizer.py)
│
└── RSTPReid/                           # Example dataset directory
    ├── gen_pairs.py                    #   Script to generate train/test JSON pair files
    ├── train_pairs.json                #   Training annotations: [{"id", "tar_path", "can_path"}]
    ├── test_pairs.json                 #   Testing annotations
    └── imgsXXXX/                       #   Image folders (0000 … 0049+)
```

---

## Quick Start

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

The `requirements.txt` pins core packages:
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

Image paths inside the JSON are **relative to the `RSTPReid/` directory**. The `RSTPReid` class joins `root_dir + dataset_dir + tar_path/can_path` to construct absolute paths.

### Download Pre-trained Weights

CLIP pretrained weights are downloaded **automatically** on first use by `build_CLIP_from_openai_pretrained()` (cached in `~/.cache/clip/`). Alternatively, download manually:

| Model | Architecture | Download URL | Notes |
|:-----:|:-------------|:-------------|:------|
| **RN50** | ResNet-50 | [RN50.pt](https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt) | Fast, `[CLS]` embedding |
| **RN101** | ResNet-101 | [RN101.pt](https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt) | Fast, `[CLS]` embedding |
| **RN50x4** | ResNet-50×4 | [RN50x4.pt](https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt) | Higher capacity |
| **RN50x64** | ResNet-50×64 | [RN50x64.pt](https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt) | Highest ResNet capacity |
| **ViT-B/32** | ViT-Base, patch 32 | [ViT-B-32.pt](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt) | Recommended baseline |
| **ViT-B/16** | ViT-Base, patch 16 | [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt) | Finer spatial patches |
| **ViT-L/14** | ViT-Large, patch 14 | [ViT-L-14.pt](https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt) | Best accuracy, largest |

In addition, the inference module requires:
- A **fine-tuned ReID checkpoint** (`best.pth`) from training.
- A **YOLOv8 model** (`yolov8l.pt` or variant) for person detection.

The online module locates these via hardcoded paths inside `online/module_reid/reid_infer.py`:

```python
_PACKAGE_DIR    = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CONFIG = os.path.join(_PACKAGE_DIR, "experiments/.../configs.yaml")
_DEFAULT_WEIGHT = os.path.join(_PACKAGE_DIR, "experiments/.../best.pth")
_DEFAULT_YOLO   = os.path.join(_PACKAGE_DIR, "experiments/.../yolov8l.pt")
```

To deploy with custom paths, copy your `configs.yaml`, `best.pth`, and `yolov8l.pt` into an `experiments/` folder under `module_reid/` or modify the `_DEFAULT_*` constants.

### Training

#### Single-GPU Quick Start

```bash
bash bash.sh
```

`bash.sh` runs the following configuration:
- CLIP: `ViT-B/32`
- Batch size: `16`
- Dataset: `RSTPReid`
- Epochs: `10`
- Log every: `100` iterations

#### Manual Training (Full Control)

```bash
CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --name baseline_reid \
    --batch_size 16 \
    --root_dir /home/mytasks/reid \
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

# Or legacy launch
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    main.py \
    --batch_size 64
```

> DDP is **auto-detected**: `main.py` checks the `WORLD_SIZE` environment variable. If `> 1`, it initializes `torch.distributed.init_process_group(backend="nccl")` and wraps the model with `DistributedDataParallel`. There is no `--distributed` CLI flag.

#### Important CLI Arguments

All arguments are defined in `utils/options.py`:

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | `baseline` | Experiment name (used in output directory) |
| `--pretrain_choice` | `ViT-B/32` | CLIP variant: `RN50`, `RN101`, `ViT-B/32`, `ViT-B/16`, `ViT-L/14` |
| `--img_size` | `(384, 128)` | Input resolution as `(height, width)` |
| `--stride_size` | `16` | Patch stride for ViT patch embedding |
| `--temperature` | `0.02` | InfoNCE temperature (`logit_scale = 1 / temperature`) |
| `--batch_size` | `32` | Training batch size per GPU |
| `--num_epoch` | `50` | Total training epochs |
| `--lr` | `1e-5` | Base learning rate |
| `--optimizer` | `Adam` | `SGD`, `Adam`, or `AdamW` |
| `--lrscheduler` | `cosine` | Decay schedule: `step`, `exp`, `poly`, `cosine`, `linear` |
| `--img_aug` | `False` | Enable image augmentation (`RandomCrop` + `RandomErasing`) |
| `--sampler` | `random` | Batch sampler: `random` or `identity` (PK sampling) |
| `--num_instance` | `4` | Instances per identity for identity sampler |
| `--log_period` | `100` | Log interval (iterations) |
| `--eval_period` | `1` | Evaluate every N epochs |
| `--resume` | `False` | Resume from `--resume_ckpt_file` |
| `--resume_ckpt_file` | `""` | Path to checkpoint for resumption |

#### Training Outputs

Each run creates a timestamped directory:

```
experiments/
└── 20260908_152404_baseline_reid/
    ├── configs.yaml          # Full config snapshot (YAML with all args)
    ├── best.pth              # Best checkpoint (selected by highest RSum)
    ├── last.pth              # Final checkpoint after last epoch
    ├── train_log.txt         # Training logs (loss, lr, memory, ETA)
    └── test_log.txt          # Final evaluation logs
```

The `Checkpointer` class (`utils/checkpoint.py`) saves a dict containing:
- `model`: `model.state_dict()`
- `optimizer`: `optimizer.state_dict()` (optional)
- `scheduler`: `scheduler.state_dict()` (optional)
- `epoch`: current epoch number

### Testing / Evaluation

```bash
python infer.py \
    --config_file experiments/20260908_152404_baseline_reid/configs.yaml \
    --gpu 0
```

`infer.py` performs the following steps:
1. Parses CLI args (`--config_file`, `--gpu`).
2. Loads the YAML config via `load_train_configs()` into an `EasyDict`.
3. Forces `args.training = False`.
4. Reconstructs the test dataloader using the same transforms.
5. Instantiates the model and loads `best.pth` via `Checkpointer.load()`.
6. Calls `do_inference()` which delegates to `Evaluator.eval()`.

Example output:

```
+-------+------+------+------+-------+------+-------+
| item  |  R1  |  R5  |  R10 |  RSum |  MdR |  MnR  |
+-------+------+------+------+-------+------+-------+
| sims  | 45.3 | 72.1 | 84.5 | 201.9 |  8.0 | 12.5  |
+-------+------+------+------+-------+------+-------+
```

### Online Inference

The `online/` directory contains a **self-contained deployment package** (`module_reid`) with no dependency on the training codebase except PyTorch and Ultralytics. The design goal is decoupling: you can copy `online/module_reid/` to a production environment without dragging in the entire training framework.

#### Package Layout

```
online/
├── main.py                             # Demo script
│                                       #   Initialize ReIDInfer and run inference on
│                                       #   a single image pair; saves annotated output.
└── module_reid/                        # Self-contained deployable package
    ├── __init__.py                     #   Exports ReIDInfer
    ├── reid_infer.py                   #   Core inference engine (see below)
    ├── model/
    │   ├── __init__.py                 #       Re-exports build_model
    │   ├── build.py                    #       Lightweight wrapper: ReID(nn.Module)
    │   │                               #       Loads CLIP + sets logit_scale = 1/temperature
    │   └── clip_model.py               #       Image-only CLIP (same implementation as root)
    │                                   #       VisionTransformer and ModifiedResNet
    │                                   #       with positional-embedding interpolation
    └── utils/
        ├── checkpoint.py               #       Minimal checkpoint loader
        │                               #       - _load_file(): torch.load(map_location="cpu")
        │                               #       - _load_model(): fuzzy key alignment
        │                               #       - strip_prefix_if_present(): removes "module."
        └── iotools.py                  #       YAML config loader returning EasyDict
```

#### Core: `ReIDInfer` Class (`reid_infer.py`)

`ReIDInfer` implements the **detect-then-compare** pipeline with these stages:

```
Target Image ──► _load_target() ──► CLIP encode ──► L2-normalized feature
                                                         │
                                                         ▼
                                                   Cached (for str/bytes paths)
                                                         │
Scene Image ──► YOLO predict(classes=[0]) ──► Detected boxes
       │                                          │
       │                                          ▼
       │                              For each box: crop → BGR→RGB → Resize(384,128)
       │                                          │
       │                                          ▼
       │                              Stack all crops ──► CLIP encode ──► L2 norm
       │                                          │
       └───────────────────────────────────► sims = target_feat @ crop_feats.T
                                                   │
                                                   ▼
                                          best_idx = argmax(sims)
                                                   │
                               ┌───────────────────┴───────────────────┐
                               ▼                                       ▼
                    best_sim > 0.8                            best_sim ≤ 0.8
                               │                                       │
                               ▼                                       ▼
                    Draw blue box + label                    No drawing
                    status = True                              status = False
```

**Initialization** (`__init__`):
- Reads three hardcoded paths (`_DEFAULT_CONFIG`, `_DEFAULT_WEIGHT`, `_DEFAULT_YOLO`) relative to `module_reid/`.
- Loads YAML config via `load_train_configs()` into an `EasyDict`.
- Builds the CLIP model with `build_model(args)`, restores weights via `Checkpointer.load()`, moves to GPU/CPU, and sets `eval()` mode.
- Loads the YOLOv8 detector via `ultralytics.YOLO()`.

**Target Feature Caching** (`_load_target`):
- Accepts `str`, `bytes`, or `PIL.Image`.
- For **path inputs**, the resolved absolute path is the cache key; if the same target is requested twice, the cached tensor is returned directly, skipping CLIP inference.
- For **PIL.Image inputs**, the cache key is `None`, so caching is bypassed (appropriate for dynamic inputs).

**Path Resolution** (`_resolve_data_path`):
1. If absolute or exists as-is, return as-is.
2. Otherwise, prepend `_PACKAGE_DIR` and check again.
3. If still not found, return the original path (let downstream fail with a clear error).

**Scene Processing** (`__call__`):
1. Loads the candidate image (`cv2.imread` for paths, `candidate_image.copy()` for arrays).
2. Runs YOLO with `classes=[0]` (person-only) and `verbose=False`.
3. Filters by `conf_thresh` (default 0.5).
4. Crops each valid detection, converts BGR→RGB, applies the CLIP transform (`Resize(384,128)`, `ToTensor`, `Normalize`), and stacks into a batch.
5. Extracts features, L2-normalizes, computes cosine similarities (`tar_feat @ can_feat.T`).
6. Finds the best match; if `best_sim > 0.8`, draws a **blue rectangle** (`(255, 0, 0)`) and a label `sims_{score:.4f}` at the top-left of the box.

**Return Value:**

| Position | Type | Description |
|----------|------|-------------|
| 0 | `bool` | `True` if `best_sim > 0.8` (confident match) |
| 1 | `dict` | `result_img`, `best_sim`, `best_box`, `boxes`, `scores` |
| 2 | `list` | `[x1, y1, x2, y2]` of best match, or `[None, ...]` |
| 3 | `float` | Best similarity score (range `[-1, 1]`, typically `[0, 1]`) |

**Edge cases handled:**
- Empty crop (`crop.size == 0`): silently skipped.
- No detections above `conf_thresh`: early return `(False, out, [None,...], 0)`.
- Image load failure: raises `ValueError` with the offending path.

---

## Model Zoo

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

For ViT variants, the default ReID head uses only the `[CLS]` token (`x[:, 0, :]` in `VisionTransformer.forward()`). The patch tokens are available in the intermediate tensor and can be aggregated via the modules in `model/cluster.py` for richer representations.

### Input Resolution

The default `--img_size '(384, 128)'` corresponds to the ReID-standard aspect ratio (3:1). For ViT:
- `num_x = (128 - patch_size) // stride_size + 1`
- `num_y = (384 - patch_size) // stride_size + 1`

When loading pretrained weights with a different native resolution, the positional embeddings are bilinearly interpolated to the new grid size (`resize_pos_embed()` in `model/clip_model.py`).

---

## Feature Enhancement Modules

The codebase includes `model/cluster.py`, a collection of **patch-token manipulation modules** that can be integrated into ViT variants for richer person representations:

### 1. PCM — Progressive Clustering Module
Compresses patch tokens via **Density Peak Clustering with K-Nearest Neighbors (DPC-KNN)**:
1. Token features pass through a 1D conv + LayerNorm.
2. A linear layer predicts a per-token importance score.
3. Scores are converted to exponential weights (masked for invalid tokens).
4. `cluster_dpc_knn()` clusters tokens into `ceil(N * sample_ratio)` groups.
5. `merge_tokens()` aggregates tokens within each cluster into a single representative.

Useful for reducing the spatial resolution of intermediate ViT features while preserving semantic information.

### 2. Att_PCM — Attention with Spatial Reduction
An attention module that supports **key/value spatial reduction** (`sr_ratio > 1`) via either:
- Conv2d + LayerNorm (`use_sr_layer=True`)
- Average pooling (`use_sr_layer=False`)

Token scores from PCM are injected as positional confidence bias into the attention logits.

### 3. Att_Block_Patch — Transformer Block
Wraps `Att_PCM` inside a standard transformer block with:
- Pre-LayerNorm
- Residual connection + `DropPath` (stochastic depth)

These modules are provided as **pluggable building blocks** and are not wired into the default `ReID` model (`model/build.py`). To use them, modify `VisionTransformer.forward()` or add a custom head that consumes patch tokens.

### Similarity Formulations

Person similarity is computed as the **cosine similarity** between feature vectors. Let us define the similarity operator uniformly:

$$\mathrm{sim}(\mathbf{u}, \mathbf{v}) \triangleq \frac{\mathbf{u}^{\top} \mathbf{v}}{\|\mathbf{u}\|_2 \, \|\mathbf{v}\|_2} \in [-1, 1]$$

where $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{D}$ are L2-normalized embeddings (unit vectors). When $\mathbf{u}$ and $\mathbf{v}$ are already L2-normalized, the expression simplifies to the dot product $\mathbf{u}^{\top}\mathbf{v}$. The codebase supports four feature-alignment strategies:

---

#### 1. Zero-shot `[CLS]`<br><sub>(no fine-tuning, single global token)</sub>

Extract the `[CLS]` token $\mathbf{I} \in \mathbb{R}^{1 \times D}$ from a pretrained CLIP visual encoder and L2-normalize:

$$S(\mathbf{I}^{(1)}, \mathbf{I}^{(2)}) = \mathrm{sim}\bigl(\mathbf{I}^{(1)}, \mathbf{I}^{(2)}\bigr)$$

where $\mathbf{I}^{(1)}$ and $\mathbf{I}^{(2)}$ are the target and candidate person embeddings, respectively. No gradient update is applied to the encoder.

---

#### 2. Fine-tuned `[CLS]`<br><sub>(end-to-end fine-tuning, single global token)</sub>

Same formulation as above, but the encoder is fine-tuned end-to-end with **InfoNCE** (a symmetric cross-entropy over a batch). For a batch of size $B$, let $\mathbf{I}^{\text{tar}}_i$ and $\mathbf{I}^{\text{can}}_i$ denote the target and candidate embeddings of the $i$-th sample. Define the similarity matrix:

$$\mathbf{S} \in \mathbb{R}^{B \times B}, \quad S_{ij} = \mathrm{sim}\bigl(\mathbf{I}^{\text{tar}}_i, \mathbf{I}^{\text{can}}_j\bigr)$$

The InfoNCE loss (with learnable temperature $\tau = 1 / \text{temperature}$) is:

$$\mathcal{L} = -\frac{1}{B} \sum_{i=1}^{B} \Biggl[ \underbrace{\log \frac{e^{\tau S_{ii}}}{\sum_{j=1}^{B} e^{\tau S_{ij}}}}_{\text{target} \to \text{candidate}} \;+\; \underbrace{\log \frac{e^{\tau S_{ii}}}{\sum_{j=1}^{B} e^{\tau S_{ji}}}}_{\text{candidate} \to \text{target}} \Biggr]$$

The diagonal $S_{ii}$ are **positive pairs** (same identity), while off-diagonal entries are **negatives**. The symmetric form ensures mutual consistency between both directions.

---

#### 3. Zero-shot `[CLS]` + `[Patch]`<br><sub>(global token + mean patch token, no fine-tuning)</sub>

Aggregate patch tokens (all spatial tokens excluding `[CLS]`) via **mean pooling** and combine with the `[CLS]` token:

$$\mathbf{P} = \frac{1}{N} \sum_{k=1}^{N} \mathbf{P}_k \in \mathbb{R}^{D}$$

$$S = \frac{1}{2} \Bigl( \mathrm{sim}\bigl(\mathbf{I}^{(1)}, \mathbf{I}^{(2)}\bigr) \;+\; \mathrm{sim}\bigl(\mathbf{P}^{(1)}, \mathbf{P}^{(2)}\bigr) \Bigr)$$

where $\mathbf{P}_k \in \mathbb{R}^{D}$ is the $k$-th patch token and $N$ is the number of patches ($N = \text{num}_x \times \text{num}_y$ for ViT).

---

#### 4. Fine-tuned `[CLS]` + `[Patch]`<br><sub>(global token + learnable patch aggregation)</sub>

Replace mean pooling with a **learnable aggregation head** $\mathcal{M}_{\theta}$ (from `model/cluster.py`, e.g. PCM, Att_PCM, or Att_Block_Patch):

$$\mathbf{P}^{*} = \mathcal{M}_{\theta}\bigl(\{\mathbf{P}_k\}_{k=1}^{N}\bigr) \in \mathbb{R}^{D}$$

$$S = \frac{1}{2} \Bigl( \mathrm{sim}\bigl(\mathbf{I}^{(1)}, \mathbf{I}^{(2)}\bigr) \;+\; \mathrm{sim}\bigl(\mathbf{P}^{*(1)}, \mathbf{P}^{*(2)}\bigr) \Bigr)$$

The aggregation head $\mathcal{M}_{\theta}$ is jointly trained with the CLIP encoder via InfoNCE. During inference, both $\mathbf{I}$ and $\mathbf{P}^{*}$ are L2-normalized before computing similarity.

---

## Evaluation Metrics

Standard CMC (Cumulative Matching Characteristics) metrics computed by `utils/metrics.py`:

| Metric | Symbol | Description |
|--------|--------|-------------|
| **Rank-1** | R1 | % queries where the ground truth is ranked 1st |
| **Rank-5** | R5 | % queries where the ground truth is in top-5 |
| **Rank-10** | R10 | % queries where the ground truth is in top-10 |
| **Rank-50** | R50 | % queries where the ground truth is in top-50 |
| **RSum** | — | `R1 + R5 + R10`, used as the validation criterion for best checkpoint |
| **MdR** | — | Median rank of the correct match |
| **MnR** | — | Mean rank of the correct match |

**Computation pipeline** (in `Evaluator.eval()`):
1. Extract L2-normalized embeddings for all `(tar_img, can_img)` pairs in the test set.
2. Compute the full cosine similarity matrix: `sims = tar_feat @ can_feat.T`.
3. For each query row, sort gallery similarities in descending order.
4. Find the rank of the diagonal entry (ground-truth pair) in the sorted list.
5. Aggregate ranks across all queries to compute the metrics above.

---

## Acknowledgments

- **[OpenAI CLIP](https://github.com/openai/CLIP)** — Pretrained vision-language representations and tokenizer.
- **[Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/)** — Real-time object detection for the online inference pipeline.

---

## Contact

📧 **Email:** [xinl067@193.com](mailto:xinl067@193.com)

🐛 **Issues:** Please open a GitHub Issue with a minimal reproduction script and the output of `python -m torch.utils.collect_env`.

---

<div align="center">
  <sub>Built for the person Re-ID community.</sub>
</div>
