import logging
import time
import torch

from utils.meter import AverageMeter
from utils.metrics import Evaluator
from utils.comm import get_rank, synchronize
from torch.utils.tensorboard import SummaryWriter

def do_train(start_epoch, args, model, train_loader, evaluator, optimizer, scheduler, checkpointer):

    log_period = args.log_period
    eval_period = args.eval_period
    device = "cuda"
    num_epoch = args.num_epoch
    arguments = {}
    arguments["num_epoch"] = num_epoch
    arguments["iteration"] = 0

    logger = logging.getLogger("ReID")
    logger.info('zero-shot of ReID model')
    _ = evaluator.eval(model.eval())

    logger.info('start training')
    meters = {
        "loss": AverageMeter(),
    }
    tb_writer = SummaryWriter(log_dir=args.output_dir)
    best_Rsum = 0.0

    for epoch in range(start_epoch, num_epoch + 1):
        start_time = time.time()
        for meter in meters.values():
            meter.reset()

        model.epoch = epoch

        model.train()
        for n_iter, batch in enumerate(train_loader):
            # id = batch['id'].to(device)
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

            if (n_iter + 1) % log_period == 0:
                info_str = f"Epoch[{epoch}] Iteration[{n_iter + 1}/{len(train_loader)}]"
                for k, v in meters.items():
                    if v.avg > 0:
                        info_str += f", {k}: {v.avg:.10f}"
                info_str += f", Base Lr: {scheduler.get_lr()[0]:.2e}"
                logger.info(info_str)

        tb_writer.add_scalar('lr', scheduler.get_lr()[0], epoch)
        tb_writer.add_scalar('temperature', ret['temperature'], epoch)
        for k, v in meters.items():
            if v.avg > 0:
                tb_writer.add_scalar(k, v.avg, epoch)

        scheduler.step()
        if get_rank() == 0:
            end_time = time.time()
            time_per_batch = (end_time - start_time) / (n_iter + 1)
            logger.info(
                "Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        if epoch % eval_period == 0:
            if get_rank() == 0:
                logger.info("Validation Results - Epoch: {}".format(epoch))
                if args.distributed:
                    Rsum = evaluator.eval(model.module.eval())
                else:
                    Rsum = evaluator.eval(model.eval())

                torch.cuda.empty_cache()
                if best_Rsum < Rsum:
                    best_Rsum = Rsum
                    arguments["epoch"] = epoch
                    checkpointer.save("best", **arguments)

            arguments["epoch"] = epoch

    if get_rank() == 0:
        logger.info(f"best Rsum: {best_Rsum} at epoch {arguments['epoch']}")

    arguments["epoch"] = epoch
    checkpointer.save("last", **arguments)
                    
def do_inference(model, test_loader):

    logger = logging.getLogger("ReID")
    logger.info("Enter inferencing")

    evaluator = Evaluator(test_loader)
    Rsum = evaluator.eval(model.eval())
    return Rsum
