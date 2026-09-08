"""
Random identity sampler for person ReID data loading.

Ensures that each training batch contains a fixed number of identities
with a fixed number of samples per identity. This sampling strategy
improves triplet / contrastive learning by guaranteeing diverse positive
and negative pairs within every batch.
"""

from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np


class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity, randomly sample
    K instances. The resulting batch size is N * K.

    This guarantees that each batch contains multiple distinct identities,
    which is beneficial for contrastive metric learning.

    Args:
        data_source (list): Dataset list of (pid, tar_path, can_path) tuples.
        batch_size (int): Total number of examples in each batch.
        num_instances (int): Number of instances (images) sampled per identity.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances

        # Mapping: person_id -> list of dataset indices.
        self.index_dic = defaultdict(list)
        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # Estimate the number of examples yielded in one epoch.
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        """
        Generate an iterator over the sampled indices.

        Groups indices by PID, shuffles within-group, and then randomly
        selects PIDs to form batches of size P * K.
        """
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        """Return the estimated number of samples yielded per epoch."""
        return self.length
