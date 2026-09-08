"""
Average meter for tracking running training statistics.

Computes the average, current value, sum, and count of a metric (e.g., loss)
over an epoch of training.
"""


class AverageMeter(object):
    """
    Computes and stores the average, current value, sum, and count.
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset all counters."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the meter with a new value.

        Args:
            val (float): The new value to record.
            n (int): Number of samples this value represents (for weighted averaging).
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
