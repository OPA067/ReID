import os
import os.path as op
from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
import argparse
from utils.iotools import load_train_configs

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ReID")
    sub = '{Model save dir instead of .pth}'
    parser.add_argument("--config_file", default=f'{sub}/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)
    args.training = False
    logger = setup_logger('ReID', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"
    args.output_dir = sub
    test_loader = build_dataloader(args)
    model = build_model(args)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    model = model.cuda()
    do_inference(model, test_loader)