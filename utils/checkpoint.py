"""
Model checkpoint manager.

Provides save/load/resume logic for model weights, optimizer states, and
scheduler states. Supports both strict loading and alignment of loaded state
dicts against the current model architecture.
"""

import logging
import os
from collections import OrderedDict

import torch


class Checkpointer:
    """
    Handles saving and loading of model checkpoints.

    In training mode, saves model, optimizer, and scheduler states.
    In inference mode, only loads model weights.
    """

    def __init__(
        self,
        model,
        optimizer=None,
        scheduler=None,
        save_dir="",
        save_to_disk=None,
        logger=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        self.save_to_disk = save_to_disk
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        """
        Save the current training state to a checkpoint file.

        Args:
            name (str): Checkpoint identifier (e.g., 'best', 'last').
            **kwargs: Additional metadata to store (e.g., epoch, iteration).
        """
        if not self.save_dir:
            return

        if not self.save_to_disk:
            return

        data = {}
        data["model"] = self.model.state_dict()
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
        if self.scheduler is not None:
            data["scheduler"] = self.scheduler.state_dict()
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        """
        Load model weights from a checkpoint file.

        Args:
            f (str, optional): Path to the checkpoint file. If None, a no-op.
        """
        if not f:
            self.logger.info("No checkpoint found.")
            return {}
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def resume(self, f=None):
        """
        Resume training from a checkpoint, restoring optimizer and scheduler states.

        Args:
            f (str, optional): Path to the checkpoint file.

        Returns:
            dict: Remaining checkpoint data (e.g., epoch, arguments).
        """
        if not f:
            self.logger.info("No checkpoint found.")
            raise IOError(f"No checkpoint file found at {f}")
        self.logger.info("Loading checkpoint from {}".format(f))
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        return checkpoint

    def _load_file(self, f):
        """Deserialize a checkpoint file."""
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, except_keys=None):
        """Load the model weights from the checkpoint dict."""
        load_state_dict(self.model, checkpoint.pop("model"), except_keys)


def check_key(key, except_keys):
    """Check whether a state-dict key matches any exclusion patterns."""
    if except_keys is None:
        return False
    else:
        for except_key in except_keys:
            if except_key in key:
                return True
        return False


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    """
    Align loaded state dict keys to the current model's keys.

    Uses a suffix-matching strategy to handle subtle naming differences
    (e.g., prefix changes after DDP wrapping).
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))

    # Build a string-matching (suffix) matrix between current and loaded keys.
    match_matrix = [
        len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys
    ]
    match_matrix = torch.as_tensor(match_matrix).view(
        len(current_keys), len(loaded_keys)
    )
    max_match_size, idxs = match_matrix.max(1)
    idxs[max_match_size == 0] = -1

    max_size = max([len(key) for key in current_keys]) if current_keys else 1
    max_size_loaded = max([len(key) for key in loaded_keys]) if loaded_keys else 1
    log_str_template = "{: <{}} loaded from {: <{}} of shape {}"
    logger = logging.getLogger("PersonSearch.checkpoint")
    for idx_new, idx_old in enumerate(idxs.tolist()):
        if idx_old == -1:
            continue
        key = current_keys[idx_new]
        key_old = loaded_keys[idx_old]
        if check_key(key, except_keys):
            continue
        model_state_dict[key] = loaded_state_dict[key_old]
        logger.info(
            log_str_template.format(
                key,
                max_size,
                key_old,
                max_size_loaded,
                tuple(loaded_state_dict[key_old].shape),
            )
        )


def strip_prefix_if_present(state_dict, prefix):
    """
    Remove a prefix (commonly 'module.') from state dict keys.

    This handles checkpoints saved from DataParallel or DistributedDataParallel.
    """
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict


def load_state_dict(model, loaded_state_dict, except_keys=None):
    """
    Load a state dictionary into the model with alignment and prefix stripping.

    Args:
        model (nn.Module): Model to load weights into.
        loaded_state_dict (dict): State dict from a checkpoint.
        except_keys (list[str], optional): Patterns of keys to skip loading.
    """
    model_state_dict = model.state_dict()
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys)

    # Strict loading: all aligned keys must match in shape.
    model.load_state_dict(model_state_dict)
