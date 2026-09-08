"""
Person Re-Identification (ReID) Training Script.

Entry point for training a CLIP-based image ReID model. Supports single-GPU
and multi-GPU (DDP) training with automatic checkpoint management.

Usage::

    # Single-GPU
    python reid_train.py

    # Multi-GPU (DDP)
    python -m torch.distributed.launch --nproc_per_node=4 reid_train.py
"""

import os
import os.path as op
import torch
import numpy as np
import random
import time

from datasets import build_dataloader
from processor.processor import do_train, do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from utils.logger import setup_logger
from solver import build_optimizer, build_lr_scheduler
from model import build_model
from utils.metrics import Evaluator
from utils.options import get_args
from utils.comm import get_rank, synchronize

import warnings
warnings.filterwarnings("ignore")


def set_seed(seed=0):
    """Fix random seeds across torch, numpy and Python for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # deterministic=True ensures reproducible results; benchmark=True trades
    # reproducibility for speed by auto-tuning convolution algorithms.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    # Step 1 -- Parse arguments and set per-rank random seed.
    # Each rank gets a different seed so that the DDP workers load different
    # data samples instead of identical shards.
    args = get_args()
    set_seed(1 + get_rank())

    # Step 2 -- Detect distributed environment.
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step 3 -- Create output directory and configure logger.
    cur_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    args.output_dir = op.join(args.output_dir, f'{cur_time}_{args.name}_{args.loss_names}')
    logger = setup_logger(
        'reid',
        save_dir=args.output_dir,
        if_train=args.training,
        distributed_rank=get_rank()
    )
    logger.info("reid training using {} GPU(s)".format(num_gpus))
    logger.info(str(args))
    save_train_configs(args.output_dir, args)

    # Step 4 -- Build data loaders and model, then move to device.
    train_loader, test_loader = build_dataloader(args)
    model = build_model(args)
    total_params = sum(p.numel() for p in model.parameters()) / (1024 * 2024)
    logger.info('total parameters: %.3fM' % total_params)
    model.to(device)

    # Wrap with DistributedDataParallel when multiple GPUs are used.
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    optimizer = build_optimizer(args, model)
    scheduler = build_lr_scheduler(args, optimizer)

    # Step 5 -- Initialize checkpoint manager and evaluator.
    is_master = get_rank() == 0
    checkpointer = Checkpointer(model, optimizer, scheduler, args.output_dir, is_master)
    evaluator = Evaluator(test_loader)

    # Step 6 -- Resume from a previous checkpoint if requested.
    start_epoch = 1
    if args.resume:
        checkpoint = checkpointer.resume(args.resume_ckpt_file)
        start_epoch = checkpoint['epoch']

    # Step 7 -- Run the main training loop.
    do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer)

    # Step 8 -- Final evaluation on the best checkpoint.
    # Reconstruct the model and test loader in evaluation mode, load the best
    # saved weights, and run inference.
    logger.info("start final test 🚀")
    args.training = False
    test_loader = build_dataloader(args)
    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model = model.cuda()
    do_inference(model, test_loader)
