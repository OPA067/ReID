"""
Optimizer and learning-rate scheduler builders for the ReID model.

This module provides helper functions that construct PyTorch optimizers and
schedulers with per-parameter-group learning-rate customization. Different
parameter groups receive different learning rates based on whether they are:
    - Randomly initialized modules (e.g., cross-modal attention layers).
    - Bias terms (typically use 2x base LR and 0 weight decay).
    - Classification / MLM heads (typically use 5x base LR).
"""

import torch

from .lr_scheduler import LRSchedulerWithWarmup


# Default LR multipliers for specific parameter groups.
_LR_FACTOR_CROSS_MODAL = 5.0    # Multiplier for randomly initialized cross-modal layers.
_LR_FACTOR_CLASSIFIER = 5.0    # Multiplier for new classification heads.
_LR_FACTOR_VISUAL_LAYER = 0.001 # Absolute LR for visual embedding adjustment layers.
_LR_FACTOR_TEXTUAL_LAYER = 0.001 # Absolute LR for textual embedding adjustment layers.


def build_optimizer(args, model):
    """
    Build a PyTorch optimizer with per-parameter-group learning rates.

    Args:
        args: Parsed configuration namespace (lr, weight_decay, etc.).
        model (nn.Module): The ReID model whose parameters will be optimized.

    Returns:
        torch.optim.Optimizer: Configured optimizer instance.
    """
    params = []

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.lr
        weight_decay = args.weight_decay

        # Use higher LR for randomly initialized cross-modal transformer layers.
        if "cross" in key:
            lr = args.lr * args.lr_factor  # Default multiplier is 5.0.

        # Bias parameters: typically no weight decay and slightly higher LR.
        if "bias" in key:
            lr = args.lr * args.bias_lr_factor
            weight_decay = args.weight_decay_bias

        # Classification / MLM heads: higher LR since they are newly trained.
        if "classifier" in key or "mlm_head" in key:
            lr = args.lr * args.lr_factor

        # Custom embedding adjustment layers (if present).
        if "visul_emb_layer" in key:
            lr = _LR_FACTOR_VISUAL_LAYER
        if "texual_emb_layer" in key:
            lr = _LR_FACTOR_TEXTUAL_LAYER

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum
        )
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-3,
        )
    elif args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(
            params,
            lr=args.lr,
            betas=(args.alpha, args.beta),
            eps=1e-8,
        )
    else:
        raise NotImplementedError(f"Optimizer {args.optimizer} is not supported.")

    return optimizer


def build_lr_scheduler(args, optimizer):
    """
    Build a learning-rate scheduler with warmup support.

    Args:
        args: Parsed configuration namespace (milestones, warmup settings, etc.).
        optimizer (Optimizer): The optimizer whose LR will be scheduled.

    Returns:
        LRSchedulerWithWarmup: Configured scheduler instance.
    """
    return LRSchedulerWithWarmup(
        optimizer,
        milestones=args.milestones,
        gamma=args.gamma,
        warmup_factor=args.warmup_factor,
        warmup_epochs=args.warmup_epochs,
        warmup_method=args.warmup_method,
        total_epochs=args.num_epoch,
        mode=args.lrscheduler,
        target_lr=args.target_lr,
        power=args.power,
    )
