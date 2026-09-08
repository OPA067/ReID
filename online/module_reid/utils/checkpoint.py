"""
Checkpoint save / load utilities for PyTorch models.

Provides ``Checkpointer`` class for persisting model, optimizer, and scheduler
states, plus helper functions for robust state-dict alignment when keys differ
(e.g. due to ``DataParallel`` prefix stripping).
"""

import logging
import os
from collections import OrderedDict

import torch


class Checkpointer:
    """
    Manages saving and loading of model checkpoints.

    Attributes:
        model  : nn.Module – model whose state_dict is saved / loaded.
        optimizer: Optimizer – optional, saved / resumed together.
        scheduler: Scheduler – optional, saved / resumed together.
        save_dir : str      – directory for checkpoint output.
        logger   : Logger   – diagnostic logger instance.
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
        Save current model (and optimizer / scheduler if present) to disk.

        Args:
            name : checkpoint filename without extension.
            **kwargs: extra items to store in the checkpoint dict.
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

        save_file = os.path.join(self.save_dir, f"{name}.pth")
        self.logger.info(f"Saving checkpoint to {save_file}")
        torch.save(data, save_file)

    def load(self, f=None):
        """
        Load model weights from ``f`` (path to .pth file).

        Args:
            f: checkpoint file path. If None, silently returns empty dict.
        """
        if not f:
            self.logger.info("No checkpoint found.")
            return {}
        self.logger.info(f"Loading checkpoint from {f}")
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)

    def resume(self, f=None):
        """
        Resume full training state: model + optimizer + scheduler.

        Args:
            f: checkpoint file path. Raises IOError if None or not found.

        Returns:
            Remaining checkpoint dict items (hyper-parameters, training step, etc.).
        """
        if not f:
            self.logger.info("No checkpoint found.")
            raise IOError(f"No checkpoint file found at {f}")
        self.logger.info(f"Loading checkpoint from {f}")
        checkpoint = self._load_file(f)
        self._load_model(checkpoint)
        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info(f"Loading optimizer from {f}")
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info(f"Loading scheduler from {f}")
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        return checkpoint

    def _load_file(self, f):
        """Deserialize a checkpoint file (auto-maps to CPU to avoid GPU memory spikes)."""
        return torch.load(f, map_location=torch.device("cpu"))

    def _load_model(self, checkpoint, except_keys=None):
        """Delegate to the module-level ``load_state_dict`` with optional key exclusion."""
        load_state_dict(self.model, checkpoint.pop("model"), except_keys)


def check_key(key, except_keys):
    """Return True if ``key`` contains any substring in ``except_keys``."""
    if except_keys is None:
        return False
    for except_key in except_keys:
        if except_key in key:
            return True
    return False


def align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys=None):
    """
    Fuzzy-match keys between ``model_state_dict`` and ``loaded_state_dict``.

    Uses suffix matching (longest match wins) so that minor naming differences
    (e.g. ``features.layer1.weight`` vs ``layer1.weight``) are tolerated.
    """
    current_keys = sorted(list(model_state_dict.keys()))
    loaded_keys = sorted(list(loaded_state_dict.keys()))

    # Build match matrix: each (i, j) entry is len(loaded_key) if current_key ends with it.
    match_matrix = [len(j) if i.endswith(j) else 0 for i in current_keys for j in loaded_keys]
    match_matrix = torch.as_tensor(match_matrix).view(len(current_keys), len(loaded_keys))
    max_match_size, idxs = match_matrix.max(1)
    idxs[max_match_size == 0] = -1  # Mark no-match entries.

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
    Strip ``prefix`` (e.g. ``module.`` from DataParallel) from all keys.

    Returns the original dict if not all keys start with the prefix.
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
    Load ``loaded_state_dict`` into ``model`` with prefix stripping and fuzzy alignment.

    Raises RuntimeError on strict mismatch (PyTorch default behavior).
    """
    model_state_dict = model.state_dict()
    # Remove DP / DDP "module." prefix if present.
    loaded_state_dict = strip_prefix_if_present(loaded_state_dict, prefix="module.")
    align_and_update_state_dicts(model_state_dict, loaded_state_dict, except_keys)
    model.load_state_dict(model_state_dict)
