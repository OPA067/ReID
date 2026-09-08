"""
I/O and file system utilities for the ReID project.

This module provides helper functions for:
    - Robust image reading (retry on IO failure).
    - JSON / YAML serialization.
    - Directory management.
"""

from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict

# Enable loading of truncated images (common in large-scale datasets).
ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_image(img_path):
    """
    Keep reading the image until successful.

    This retry loop mitigates transient IOErrors caused by heavy disk
    contention or NFS latency in shared cluster environments.

    Args:
        img_path (str): Path to the image file.

    Returns:
        PIL.Image: RGB image.

    Raises:
        IOError: If the file does not exist.
    """
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will retry.".format(img_path))
            pass
    return img


def mkdir_if_missing(directory):
    """Create a directory (and its parents) if it does not already exist."""
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def check_isfile(path):
    """Check whether a file exists and print a warning if not."""
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    """Load a JSON file."""
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """Save an object to a JSON file with pretty formatting."""
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def save_train_configs(path, args):
    """
    Persist training arguments to a YAML configuration file.

    Args:
        path (str): Destination directory.
        args: argparse.Namespace or object with __dict__.
    """
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)


def load_train_configs(path):
    """
    Load training configuration from a YAML file.

    Args:
        path (str): Path to the config YAML file.

    Returns:
        EasyDict(edict): Hierarchical dict-like object for attribute-style access.
    """
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)
