"""
Multi-GPU communication primitives for distributed training.

This module provides thin wrappers around torch.distributed for:
    - Determining world size and process rank.
    - Barrier synchronization across processes.
    - All-gathering arbitrary Python objects or tensors.
    - Reducing dictionaries across processes.
"""

import pickle

import torch
import torch.distributed as dist


def get_world_size():
    """Return the number of DDP processes (1 if not distributed)."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Return the current process rank (0 if not distributed)."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    """Check if the current process is the master (rank 0)."""
    return get_rank() == 0


def synchronize():
    """
    Synchronize (barrier) among all DDP processes.

    No-op if torch.distributed is not available or initialized.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Each process must participate; otherwise it will deadlock.

    Args:
        data: Any picklable Python object.

    Returns:
        list[data]: Data gathered from all ranks.
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize the object into a byte tensor.
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # Determine the maximum serialized size across all ranks.
    local_size = torch.IntTensor([tensor.numel()]).to("cuda")
    size_list = [torch.IntTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # Pad and gather byte tensors from every rank.
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    # Deserialize each rank's data back into Python objects.
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True):
    """
    Reduce (sum or average) dictionary values across all DDP processes.

    Args:
        input_dict (dict): Dictionary with tensor values to reduce.
        average (bool): If True, average the values; otherwise, sum them.

    Returns:
        dict: Reduced dictionary (only valid on rank 0 if average=True).
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # Sort keys to ensure consistent ordering across processes.
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.reduce(values, dst=0)
        if dist.get_rank() == 0 and average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict
