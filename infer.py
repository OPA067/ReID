"""
Person Re-Identification (ReID) Inference / Testing Script.

Evaluates a trained ReID model on the test set by loading the saved
configuration, reconstructing the data loader, restoring model weights,
and computing ranking metrics (R1, R5, R10, Rsum, MdR, MnR).

Usage::

    python infer.py --config_file experiments/YYYYMMDD_HHMMSS_baseline/configs.yaml
"""

import os
import os.path as op
import torch

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.iotools import load_train_configs
import argparse

import warnings
warnings.filterwarnings("ignore")

# Restrict PyTorch to the selected GPU.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':

    # Step 1 -- Parse arguments.
    parser = argparse.ArgumentParser(description="person reid inference")
    parser.add_argument("--config_file", default='experiments/20260908_161319_baseline_reid/configs.yaml', help="Path to the YAML config saved during training.")
    parser.add_argument("--gpu", default='0', help="GPU device ID(s) to use. Default: '0'.")
    args_cli = parser.parse_args()

    # Override GPU selection if provided via CLI.
    os.environ['CUDA_VISIBLE_DEVICES'] = args_cli.gpu

    # Step 2 -- Load training configuration and force evaluation mode.
    args = load_train_configs(args_cli.config_file)
    args.training = False

    # Step 3 -- Setup logger.
    logger = setup_logger('reid', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)

    # Step 4 -- Build test data loader and model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = build_dataloader(args)

    # Step 5 -- Load model weights from the best checkpoint.
    model = build_model(args)
    checkpointer = Checkpointer(model)
    best_ckpt = op.join(args.output_dir, 'best.pth')
    checkpointer.load(f=best_ckpt)
    model = model.to(device)

    # Step 6 -- Run inference.
    logger.info("start inference 🚀")
    do_inference(model, test_loader)
