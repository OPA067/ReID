"""
Simple I/O helpers for configuration management.

Reads YAML training configs and returns them as ``EasyDict`` for
attribute-style access (e.g. ``args.img_size`` instead of ``args["img_size"]``).
"""

import os
import os.path as osp

import yaml
from easydict import EasyDict as edict


def load_train_configs(path):
    """
    Load a YAML config file and return an EasyDict.

    Args:
        path: path to the YAML configuration file.

    Returns:
        EasyDict with attribute access for all top-level keys.
    """
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)
