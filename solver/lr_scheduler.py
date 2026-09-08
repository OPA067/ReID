"""
Learning rate scheduler with warmup support.

Implements a composite scheduler that first warms up the learning rate
over a number of epochs, then applies a chosen decay strategy:
    step, exp, poly, cosine, or linear.
"""

from bisect import bisect_right
from math import cos, pi

from torch.optim.lr_scheduler import _LRScheduler


class LRSchedulerWithWarmup(_LRScheduler):
    """
    Multi-mode learning-rate schedule with warmup.

    Warmup Phase (first warmup_epochs):
        - 'constant':  Keep LR at base_lr * warmup_factor.
        - 'linear':    Linearly interpolate from warmup_factor to 1.0.

    Post-warmup Phase:
        - 'step':      Step-wise decay at specified milestones.
        - 'exp':       Exponential decay based on epoch ratio.
        - 'linear':    Linear decay to zero.
        - 'poly':      Polynomial decay toward target_lr.
        - 'cosine':    Cosine annealing toward target_lr.

    Args:
        optimizer:              Target PyTorch optimizer.
        milestones (list):      Epochs at which to decay (used in 'step' mode).
        gamma (float):          Decay factor (used in 'step' mode).
        mode (str):             Decay mode ('step', 'exp', 'poly', 'cosine', 'linear').
        warmup_factor (float):  Initial LR multiplier during warmup.
        warmup_epochs (int):    Number of warmup epochs.
        warmup_method (str):    'constant' or 'linear' warmup method.
        total_epochs (int):     Total number of training epochs.
        target_lr (float):      Final LR for polynomial / cosine modes.
        power (float):          Polynomial exponent (used in 'poly' mode).
        last_epoch (int):       Index of the last epoch (resumes from a checkpoint).
    """

    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        mode="step",
        warmup_factor=1.0 / 3,
        warmup_epochs=10,
        warmup_method="linear",
        total_epochs=100,
        target_lr=0,
        power=0.9,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones),
            )
        if mode not in ("step", "exp", "poly", "cosine", "linear"):
            raise ValueError(
                "Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted"
                "got {}".format(mode)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.mode = mode
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """
        Compute the learning rate for each parameter group.

        Returns:
            list[float]: Learning rate for each param group.
        """
        if self.last_epoch < self.warmup_epochs:
            # --------------------------------------------------------------
            # Warmup phase: gradually increase LR from warmup_factor to 1.0.
            # --------------------------------------------------------------
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        if self.mode == "step":
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]

        epoch_ratio = (self.last_epoch - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )

        if self.mode == "exp":
            factor = epoch_ratio
            return [base_lr * self.power ** factor for base_lr in self.base_lrs]

        if self.mode == "linear":
            factor = 1 - epoch_ratio
            return [base_lr * factor for base_lr in self.base_lrs]

        if self.mode == "poly":
            factor = 1 - epoch_ratio
            return [
                self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                for base_lr in self.base_lrs
            ]

        if self.mode == "cosine":
            # Cosine annealing: smoothly decay toward target_lr.
            factor = 0.5 * (1 + cos(pi * epoch_ratio))
            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]

        raise NotImplementedError
