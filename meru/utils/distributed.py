# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Collection of common utilities for distributed training. These are wrappers over
functions from :mod:`torch.distributed` module, but they do not raise exceptions
in absence of multi-GPU or CPU mode, and fall back to sensible default behavior.
"""
from __future__ import annotations

from typing import Callable

import torch
from loguru import logger
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.distributed.nn import all_gather as nn_all_gather


def launch(
    job_fn: Callable,
    num_machines: int = 1,
    num_gpus_per_machine: int = 1,
    machine_rank: int = 0,
    dist_url: str = "tcp://127.0.0.1:23457",
    args=(),
):
    """
    Launch a job in a distributed fashion: given `num_machines` machines, each
    with `num_gpus_per_machine` GPUs, this function will launch one process per
    GPU. This wrapper uses :func:`torch.multiprocessing.spawn`.

    The user has to launch one job on each machine, manually specifying a machine
    rank (incrementing integers from 0). This function will offset process ranks
    per machine. One process on `machine_rank = 0` will be the *main process*,
    and a free port on that machine will be used for process communication.

    Default arguments imply one machine with one GPU, and communication URL
    as `localhost`.

    .. note::

        We assume all machines have same number of GPUs per machine, with IDs as
        `(0, 1, 2 ...)`. If you do not wish to use all GPUs on a machine,
        set `CUDA_VISIBLE_DEVICES` environment variable appropriately.

    Args:
        job_fn: Function to launch -- this could be your model training function.
        num_machines: Number of machines, each with `num_gpus_per_machine` GPUs.
        num_gpus_per_machine: GPUs per machine, with IDs as `(0, 1, 2 ...)`.
        machine_rank: A manually specified rank of the machine, serves as a
            unique identifier and useful for assigning global ranks to processes.
        dist_url: Disributed process communication URL as `tcp://x.x.x.x:port`.
            Set this as the IP (and a free port) of machine with rank 0.
        args: Arguments to be passed to `job_fn`.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not found! Cannot launch distributed processes.")

    world_size = num_machines * num_gpus_per_machine

    # Spawn `num_gpus_per_machine` processes per machine, and provide
    # "local process rank" (GPU ID) as the first arg to `_dist_worker`.
    # fmt: off
    if world_size > 1:
        mp.spawn(
            _job_worker,
            nprocs=num_gpus_per_machine,
            args=(
                job_fn, world_size, num_gpus_per_machine, machine_rank, dist_url, args
            ),
            daemon=False,
        )
    else:
        # Default to single machine, single GPU, with ID 0.
        _job_worker(0, job_fn, 1, 1, 0, dist_url, args)
    # fmt: on


def _job_worker(
    local_rank: int,
    job_fn: Callable,
    world_size: int,
    num_gpus_per_machine: int,
    machine_rank: int,
    dist_url: str,
    args: tuple,
):
    """
    Single distibuted process worker. This function should never be used directly,
    only used by :func:`launch`.
    """

    # Adjust global rank of process based on its machine rank.
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend="NCCL",
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank,
        )
    except Exception as e:
        logger.error(f"Error launching processes, dist URL: {dist_url}")
        raise e

    synchronize()
    # Set GPU ID for each process according to its rank.
    torch.cuda.set_device(local_rank)
    job_fn(*args)


def synchronize() -> None:
    """Synchronize (barrier) all processes in a process group."""
    if dist.is_initialized():
        dist.barrier()


def get_world_size() -> int:
    """Return number of processes in the process group, each uses 1 GPU."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Return rank of current process in the process group."""
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    """
    Check whether current process is the main process. This check is useful
    to restrict logging and checkpointing to main process. It will always
    return `True` for single machine, single GPU execution.
    """
    return get_rank() == 0


def gather_across_processes(t: torch.Tensor) -> list[torch.Tensor]:
    """
    Gather tensors from multiple GPU processes in a list. The order of elements
    is preserved by GPU process IDs. This operation is differentiable; gradients
    will be scattered back to devices in the backward pass.

    Args:
        t: Tensor to gather across processes.
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [t]

    output = list(nn_all_gather(t))
    return output


def gpu_mem_usage() -> int:
    """
    Return gpu memory usage (in megabytes). If not using GPU, return 0 without
    raising any exceptions.
    """
    if torch.cuda.is_available():
        # This will be in bytes, so we divide by (1024 * 1024).
        return torch.cuda.max_memory_allocated() // 1048576
    else:
        return 0
