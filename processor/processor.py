"""
Training and inference processor for the ReID pipeline.
"""

import logging
import time
import torch

from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):
    """Run the main training loop with periodic evaluation and TensorBoard logging."""
    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("reid")

    # Baseline zero-shot evaluation before any training begins.
    logger.info('reid model zero-shot performance: ')
    _ = evaluator.eval(model.eval())

    logger.info('start training 🚀')
    meters = {"loss": AverageMeter()}
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_Rsum = 0.0

    # Track epoch-level time for ETA estimation.
    start_time = time.time()

    for epoch in range(start_epoch, num_epoch + 1):
        epoch_start = time.time()
        for meter in meters.values():
            meter.reset()

        model.epoch = epoch
        model.train()

        # Iterate over all training batches in the current epoch.
        for n_iter, batch in enumerate(train_loader):
            t0 = time.time()
            batch_id = batch['id'].to(device)
            tar_img = batch['tar_img'].to(device)
            can_img = batch['can_img'].to(device)

            ret = model(tar_img, can_img)
            total_loss = sum([v for k, v in ret.items() if "loss" in k])

            batch_size = batch['tar_img'].shape[0]
            meters['loss'].update(total_loss.item(), batch_size)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            synchronize()

            # Log training metrics, GPU memory, and ETA at the specified interval.
            if (n_iter + 1) % log_period == 0:
                info_str = f"epoch[{epoch}/{num_epoch}], iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.10f}"
                info_str += f", base lr: {scheduler.get_lr()[0]:.2e}"

                # GPU memory usage: allocated + reserved (in MB).
                if torch.cuda.is_available():
                    mem_alloc = torch.cuda.memory_allocated() / (1024 ** 2)
                    mem_reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                    info_str += f", mem: {mem_alloc:.0f}/{mem_reserved:.0f}MB"

                # ETA to finish the current epoch (based on average time per iteration).
                elapsed = time.time() - epoch_start
                iters_done = n_iter + 1
                iters_left = len(train_loader) - iters_done
                eta_sec = int(elapsed / iters_done * iters_left)
                eta_str = f"{eta_sec // 3600:02d}:{(eta_sec % 3600) // 60:02d}:{eta_sec % 60:02d}"
                info_str += f", eta: {eta_str}"

                logger.info(info_str)

        # Log metrics to TensorBoard after each epoch.
        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()

        if get_rank() == 0:
            end_time = time.time()
            epoch_time = end_time - epoch_start
            time_per_batch = epoch_time / (n_iter + 1)
            logger.info(
                "epoch {} done. time: {:.1f}m, per batch: {:.3f}s, speed: {:.1f} samples/s"
                .format(epoch, epoch_time / 60, time_per_batch, train_loader.batch_size / time_per_batch)
            )

        # Validate every eval_period epochs and save the best checkpoint.
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("validation results - epoch: {}".format(epoch))
                model_to_eval = model.module if args.distributed else model
                Rsum = evaluator.eval(model_to_eval.eval())

                torch.cuda.empty_cache()
                if best_Rsum < Rsum:
                    best_Rsum = Rsum
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

            arguments["epoch"] = epoch

    if get_rank() == 0:
        logger.info(f"best RSum: {best_Rsum} at epoch {arguments['epoch']}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)


def do_inference(model, test_loader):
    """Evaluate a trained model on the test set and return the RSum metric."""
    evaluator = Evaluator(test_loader)
    Rsum = evaluator.eval(model.eval())
    return Rsum
