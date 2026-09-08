"""
Logging utility for the ReID project.

Sets up a dual-handler logger that writes to both the console (stdout)
and a file (train_log.txt or test_log.txt). In distributed training,
only the master process (rank 0) creates the file handler to avoid
concurrent write conflicts.
"""

import logging
import os
import sys
import os.path as op


def setup_logger(name, save_dir, if_train, distributed_rank=0):
    """
    Configure a logger with console and file handlers.

    Args:
        name (str): Logger name (typically 'reid').
        save_dir (str): Directory where log files are saved.
        if_train (bool): True -> train_log.txt, False -> test_log.txt.
        distributed_rank (int): DDP rank (0 = master). Non-master ranks
                                only get a console handler.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Skip file logging for non-master processes in DDP.
    if distributed_rank > 0:
        return logger

    # Console handler.
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler.
    if not op.exists(save_dir):
        print(f"{save_dir} does not exist, creating directory")
        os.makedirs(save_dir)
    if if_train:
        fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
    else:
        fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='a')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
