"""
Configuration argument parser for the ReID project.

Contains default hyperparameters covering:
    - General settings (GPU, logging, checkpointing).
    - Model architecture (CLIP variant, input size, temperature).
    - Loss functions (multi-loss composition).
    - Vision transformer settings.
    - Optimizer / scheduler hyperparameters.
    - Dataset and batching parameters.
"""

import argparse


def get_args():
    """
    Build and return the argument parser with all default configurations.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Person ReID Training")

    # ===================== General Settings =====================
    parser.add_argument("--local_rank", default=0, type=int, help="Local rank for DDP (set automatically by torchrun/launch)")
    parser.add_argument("--name", default="baseline", help="Experiment name for checkpoint directory")
    parser.add_argument("--output_dir", default="experiments", help="Root output directory")
    parser.add_argument("--log_period", default=100, type=int, help="Number of iterations between training log prints")
    parser.add_argument("--eval_period", default=1, type=int, help="Evaluate every N epochs")
    parser.add_argument("--val_dataset", default="test", help="Use validation or test set during evaluation")
    parser.add_argument("--resume", default=False, action='store_true', help="Resume training from a checkpoint")
    parser.add_argument("--resume_ckpt_file", default="", help="Checkpoint file for resuming")

    # ===================== Model Architecture =====================
    parser.add_argument("--pretrain_choice", default='ViT-B/32', help="CLIP pre-trained model variant to load")
    parser.add_argument("--temperature", type=float, default=0.02, help="Temperature for InfoNCE scaling (0 disables temperature scaling)")
    parser.add_argument("--img_aug", default=False, action='store_true', help="Enable image data augmentation")
    parser.add_argument("--txt_aug", default=False, action='store_true', help="Enable text data augmentation (placeholder)")

    # Cross-modal transformer settings (placeholder for future extension).
    parser.add_argument("--cmt_depth", type=int, default=4, help="Cross-modal transformer self-attention layers")
    parser.add_argument("--masked_token_rate", type=float, default=0.8, help="Mask ratio for MLM task")
    parser.add_argument("--masked_token_unchanged_rate", type=float, default=0.1, help="Ratio of masked tokens left unchanged")
    parser.add_argument("--lr_factor", type=float, default=5.0, help="LR multiplier for randomly initialized modules")

    # ===================== Loss Settings =====================
    parser.add_argument("--loss_names", default='sdm+id+mlm', help="Loss composition string, e.g., 'sdm+id+mlm'")

    # ===================== Vision Transformer =====================
    parser.add_argument("--img_size", type=tuple, default=(384, 128), help="Input image size as (height, width)")
    parser.add_argument("--stride_size", type=int, default=16, help="Patch stride for ViT patch embedding")

    # ===================== Text Settings (placeholder) =====================
    parser.add_argument("--text_length", type=int, default=77, help="Maximum text token length")
    parser.add_argument("--vocab_size", type=int, default=49408, help="Vocabulary size for BPE tokenizer")

    # ===================== Optimizer =====================
    parser.add_argument("--optimizer", type=str, default="Adam", help="Optimizer type: [SGD, Adam, AdamW]")
    parser.add_argument("--lr", type=float, default=1e-5, help="Base learning rate")
    parser.add_argument("--bias_lr_factor", type=float, default=2.0, help="LR multiplier for bias parameters")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD")
    parser.add_argument("--weight_decay", type=float, default=4e-5, help="Weight decay (L2 regularization)")
    parser.add_argument("--weight_decay_bias", type=float, default=0.0, help="Weight decay for bias terms")
    parser.add_argument("--alpha", type=float, default=0.9, help="Beta1 for Adam / AdamW")
    parser.add_argument("--beta", type=float, default=0.999, help="Beta2 for Adam / AdamW")

    # ===================== LR Scheduler =====================
    parser.add_argument("--num_epoch", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--milestones", type=int, nargs='+', default=(20, 50), help="Epoch indices for step LR decay")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR decay factor at milestones")
    parser.add_argument("--warmup_factor", type=float, default=0.1, help="Initial LR multiplier during warmup")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of warmup epochs")
    parser.add_argument("--warmup_method", type=str, default="linear", help="Warmup strategy: 'linear' or 'constant'")
    parser.add_argument("--lrscheduler", type=str, default="cosine", help="Decay mode: step, exp, poly, cosine, linear")
    parser.add_argument("--target_lr", type=float, default=0, help="Minimum LR for polynomial / cosine decay")
    parser.add_argument("--power", type=float, default=0.9, help="Polynomial decay exponent")

    # ===================== Dataset =====================
    parser.add_argument("--dataset_name", default="RSTP-Reid", help="Dataset name [CUHK-PEDES, ICFG-PEDES-train, RSTP-Reid]")
    parser.add_argument("--sampler", default="random", help="Batch sampler: 'identity' or 'random'")
    parser.add_argument("--num_instance", type=int, default=4, help="Instances per identity in an identity sampler batch")
    parser.add_argument("--root_dir", default="/home/user/", help="Root directory of the dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size")
    parser.add_argument("--test_batch_size", type=int, default=32, help="Testing batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of data loading worker processes")
    parser.add_argument("--test", dest='training', default=True, action='store_false', help="Set to evaluation mode (disables training data loading)")

    args = parser.parse_args()
    return args
