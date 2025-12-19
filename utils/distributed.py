"""
Distributed Data Parallel (DDP) utilities for multi-GPU training.

This module provides helper functions for setting up and managing
distributed training with PyTorch's DDP.

Usage:
    # With torchrun (recommended, 1 process per GPU):
    torchrun --nproc_per_node=8 main.py --model itransformer ...

    # Multi-process per GPU (4 processes per GPU on 8 GPUs = 32 workers):
    torchrun --nproc_per_node=32 main.py --model itransformer --processes_per_gpu 4 ...

    # Single-GPU fallback (no torchrun):
    python main.py --model itransformer ...
"""

import os
import torch
import torch.distributed as dist

# Global variable to store processes_per_gpu after setup
_PROCESSES_PER_GPU = 1


def is_dist_available_and_initialized():
    """Check if distributed training is available and initialized."""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    """Get total number of processes in distributed training."""
    if not is_dist_available_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    """Get the rank of the current process."""
    if not is_dist_available_and_initialized():
        return 0
    return dist.get_rank()


def get_local_rank():
    """Get the local rank (process index within the node)."""
    if not is_dist_available_and_initialized():
        return 0
    return int(os.environ.get("LOCAL_RANK", 0))


def get_gpu_id(processes_per_gpu=None):
    """
    Get the physical GPU ID for this process.

    When running multiple processes per GPU, LOCAL_RANK no longer maps
    directly to the GPU ID. This function computes the correct GPU ID.

    Args:
        processes_per_gpu: Number of processes per GPU. If None, uses the
                          global value set during setup_distributed().

    Returns:
        Physical GPU ID (0 to num_gpus-1)

    Example with 4 processes per GPU on 8 GPUs:
        LOCAL_RANK 0-3   -> GPU 0
        LOCAL_RANK 4-7   -> GPU 1
        LOCAL_RANK 8-11  -> GPU 2
        ...
        LOCAL_RANK 28-31 -> GPU 7
    """
    if not is_dist_available_and_initialized():
        return 0

    if processes_per_gpu is None:
        processes_per_gpu = _PROCESSES_PER_GPU

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    gpu_id = local_rank // processes_per_gpu
    return gpu_id


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return get_rank() == 0


def setup_distributed(backend="nccl", processes_per_gpu=1):
    """
    Initialize distributed training environment.

    Called automatically when torchrun sets environment variables.
    Falls back to single-GPU mode when env vars are not set.

    Args:
        backend: Communication backend ("nccl" for GPU, "gloo" for CPU)
        processes_per_gpu: Number of processes per GPU (default: 1)
                          Note: NCCL backend does NOT support multiple processes
                          per GPU. Use gloo backend or gradient accumulation instead.

    Returns:
        tuple: (rank, local_rank, world_size, gpu_id)
    """
    global _PROCESSES_PER_GPU
    _PROCESSES_PER_GPU = processes_per_gpu

    # Check if launched with torchrun (sets RANK, LOCAL_RANK, WORLD_SIZE)
    if "RANK" not in os.environ:
        # Single-GPU fallback
        return 0, 0, 1, 0

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Calculate physical GPU ID when multiple processes per GPU
    gpu_id = local_rank // processes_per_gpu

    # Warn about NCCL limitation with multiple processes per GPU
    if processes_per_gpu > 1 and backend == "nccl":
        if rank == 0:
            print("WARNING: NCCL backend does not support multiple processes per GPU!")
            print("         This will cause 'Duplicate GPU detected' errors.")
            print("         Consider using:")
            print("           1. Single process per GPU (recommended)")
            print("           2. Gradient accumulation for larger effective batch")
            print("           3. gloo backend (much slower for GPU training)")
            print()

    # Set the device before initializing process group
    torch.cuda.set_device(gpu_id)

    # Initialize process group
    try:
        # New API (PyTorch >= 2.0)
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{gpu_id}")
        )
    except TypeError:
        # Old API (PyTorch < 2.0) - device_id not supported
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            world_size=world_size,
            rank=rank
        )

    # Synchronize all processes
    dist.barrier()

    if is_main_process():
        print(f"Distributed training initialized:")
        print(f"  World size: {world_size}")
        print(f"  Backend: {backend}")

    return rank, local_rank, world_size, gpu_id


def cleanup_distributed():
    """Clean up distributed training resources."""
    if is_dist_available_and_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor, average=True):
    """
    Reduce a tensor across all processes.

    Args:
        tensor: Input tensor to reduce
        average: If True, average the result; otherwise sum

    Returns:
        Reduced tensor
    """
    if not is_dist_available_and_initialized():
        return tensor

    world_size = get_world_size()
    if world_size == 1:
        return tensor

    # Clone to avoid modifying the original
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)

    if average:
        rt = rt / world_size

    return rt


def gather_tensor(tensor):
    """
    Gather tensors from all processes to all ranks.

    Args:
        tensor: Input tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not is_dist_available_and_initialized():
        return [tensor]

    world_size = get_world_size()
    if world_size == 1:
        return [tensor]

    # Gather all tensors
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)

    return gathered


def print_rank0(*args, **kwargs):
    """Print only on rank 0 process."""
    if is_main_process():
        print(*args, **kwargs)


def barrier():
    """Synchronize all processes."""
    if is_dist_available_and_initialized():
        dist.barrier()
