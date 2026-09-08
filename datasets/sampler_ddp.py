"""
Distributed Random Identity Sampler (DDP compatible).

Extends the single-GPU RandomIdentitySampler to work correctly under
DistributedDataParallel (DDP). All workers share a common random seed,
ensuring consistent state across processes, while partitioning batches
so that each process consumes a non-overlapping subset.
"""

from torch.utils.data.sampler import Sampler
from collections import defaultdict
import copy
import random
import numpy as np
import math
import torch.distributed as dist

_LOCAL_PROCESS_GROUP = None


def _get_global_gloo_group():
    """
    Return a process group based on the gloo backend, containing all ranks.
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    """
    Serialize an arbitrary picklable object into a byte tensor for all_gather.
    """
    backend = dist.get_backend(group)
    assert backend in ["gloo", "nccl"]
    device = torch.device("cpu" if backend == "gloo" else "cuda")

    import pickle
    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        print(
            "Rank {} trying to all-gather {:.2f} GB of data on device {}".format(
                dist.get_rank(), len(buffer) / (1024 ** 3), device
            )
        )
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    """
    Pad a tensor to the maximum size across all ranks for all_gather.

    Returns:
        list[int]: Size of the tensor on each rank.
        Tensor: The padded tensor.
    """
    world_size = dist.get_world_size(group=group)
    assert (
            world_size >= 1
    ), "comm.gather/all_gather must be called from ranks within the given group!"
    local_size = torch.tensor([tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [
        torch.zeros([1], dtype=torch.int64, device=tensor.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]

    max_size = max(size_list)
    if local_size != max_size:
        padding = torch.zeros((max_size - local_size,), dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: Any picklable object.
        group: A torch process group. Defaults to a gloo group of all ranks.

    Returns:
        list[data]: List of data gathered from each rank.
    """
    if dist.get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]

    tensor = _serialize_to_tensor(data, group)

    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    tensor_list = [
        torch.empty((max_size,), dtype=torch.uint8, device=tensor.device) for _ in size_list
    ]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def shared_random_seed():
    """
    Generate a random seed that is identical across all DDP workers.

    All workers must call this function; otherwise it will deadlock.

    Returns:
        int: A shared random seed.
    """
    ints = np.random.randint(2 ** 31)
    all_ints = all_gather(ints)
    return all_ints[0]


class RandomIdentitySampler_DDP(Sampler):
    """
    DDP-compatible Random Identity Sampler.

    Similar to RandomIdentitySampler, but splits the generated batch indices
    across all DDP processes so that each rank consumes a disjoint subset
    of the global batch.

    Args:
        data_source (list): Dataset list of (pid, tar_path, can_path) tuples.
        batch_size (int): Total global batch size across all GPUs.
        num_instances (int): Number of instances sampled per identity.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.world_size = dist.get_world_size()
        self.num_instances = num_instances
        self.mini_batch_size = self.batch_size // self.world_size
        self.num_pids_per_batch = self.mini_batch_size // self.num_instances
        self.index_dic = defaultdict(list)

        for index, (pid, _, _, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

        # Estimate total number of examples in an epoch (before DDP splitting).
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

        self.rank = dist.get_rank()
        self.length //= self.world_size

    def __iter__(self):
        """
        Generate an iterator of indices for the current DDP rank.

        Uses a shared random seed to ensure all ranks generate the same
        global index ordering, then extracts the rank-specific slice.
        """
        seed = shared_random_seed()
        np.random.seed(seed)
        self._seed = int(seed)
        final_idxs = self.sample_list()
        final_idxs = self.__fetch_current_node_idxs(
            final_idxs, int(math.ceil(len(final_idxs) * 1.0 / self.world_size))
        )
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __fetch_current_node_idxs(self, final_idxs, length):
        """
        Partition the global index list for the current DDP rank.

        Args:
            final_idxs (list): Global list of sampled dataset indices.
            length (int): Target number of indices per rank.

        Returns:
            list: Indices assigned to the current rank.
        """
        total_num = len(final_idxs)
        block_num = length // self.mini_batch_size
        index_target = []
        for i in range(0, block_num * self.world_size, self.world_size):
            index = range(
                self.mini_batch_size * self.rank + self.mini_batch_size * i,
                min(self.mini_batch_size * self.rank + self.mini_batch_size * (i + 1), total_num)
            )
            index_target.extend(index)
        return list(np.array(final_idxs)[np.array(index_target)])

    def sample_list(self):
        """
        Generate the full global index list using random identity sampling.

        Returns:
            list: Ordered list of indices covering the entire dataset.
        """
        avai_pids = copy.deepcopy(self.pids)
        batch_idxs_dict = {}

        batch_indices = []
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = np.random.choice(
                avai_pids, self.num_pids_per_batch, replace=False
            ).tolist()
            for pid in selected_pids:
                if pid not in batch_idxs_dict:
                    idxs = copy.deepcopy(self.index_dic[pid])
                    if len(idxs) < self.num_instances:
                        idxs = np.random.choice(idxs, size=self.num_instances, replace=True).tolist()
                    np.random.shuffle(idxs)
                    batch_idxs_dict[pid] = idxs

                avai_idxs = batch_idxs_dict[pid]
                for _ in range(self.num_instances):
                    batch_indices.append(avai_idxs.pop(0))

                if len(avai_idxs) < self.num_instances:
                    avai_pids.remove(pid)

        return batch_indices

    def __len__(self):
        """Return the number of indices consumed by the current rank per epoch."""
        return self.length
