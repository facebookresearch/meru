# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
from pathlib import Path

import torch
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

import meru.utils.distributed as dist


class CheckpointManager:
    """
    Utility class to perioidically save PyTorch models and other checkpointables
    (optimizers, LR schedulers etc., which implement `state_dict` method)
    during training. For PyTorch `DistributedDataParallel` objects,
    `state_dict` of the internal model is saved.

    Examples:
        >>> model = torch.nn.Linear(10, 2)
        >>> optimizer = torch.optim.Adam(model.parameters())
        >>> ckpt_manager = CheckpointManager("/tmp", model=model, optimizer=optimizer)
        >>> for iteration in range(1000):
        ...     train(model)
        ...     if iteration % 100 == 0:
        ...         ckpt_manager.step(iteration)
    """

    def __init__(
        self, output_dir: str = "/tmp", keep_recent: int = 100, **checkpointables
    ):
        """
        Args:
            output_dir: Path to a directory to save checkpoints.
            keep_recent: Number of recent `k` checkpoints to keep, old checkpoints
                will be deleted. Set to a very large value to keep all checkpoints.
            checkpointables: Keyword arguments with any checkpointable objects, for
                example: model, optimizer, LR scheduler, AMP gradient scaler.
        """
        self.output_dir = Path(output_dir)
        self.keep_recent = keep_recent
        self._recent_iterations = []

        # Shallow copy, keeps references to tensors as original objects so they
        # can be updated externally, or loaded in here without needing explicit
        # synchronization after every operation.
        self.checkpointables = copy.copy(checkpointables)

    def step(self, iteration: int):
        """
        Save a checkpoint; keys match those in :attr:`checkpointables`.

        Args:
            iteration: Current training iteration. Will be saved with other
                checkpointables.
        """

        out_state_dict = {}
        for key in self.checkpointables:
            if isinstance(self.checkpointables[key], DistributedDataParallel):
                out_state_dict[key] = self.checkpointables[key].module.state_dict()
            else:
                out_state_dict[key] = self.checkpointables[key].state_dict()

        # We also checkpoint current iteration.
        out_state_dict["iteration"] = iteration

        # String formatting, assuming we won't train for more than 99M iterations.
        iter_str = f"{iteration:0>8d}"

        # Save checkpoint corresponding to current iteration.
        torch.save(out_state_dict, self.output_dir / f"checkpoint_{iter_str}.pth")
        with (self.output_dir / "last_checkpoint.txt").open("w") as f:
            f.write(f"checkpoint_{iter_str}.pth")

        # Remove earliest checkpoint if there are more on disk.
        self._recent_iterations.append(iter_str)
        if len(self._recent_iterations) > self.keep_recent:
            oldest_iteration = self._recent_iterations.pop(0)
            (self.output_dir / f"checkpoint_{oldest_iteration}.pth").unlink()

    def final_step(self):
        """
        Save the final checkpoint with name `checkpoint_final.pth`. This method
        does not update `last_checkpoint.txt` or delete the oldest checkpoint.
        """

        out_state_dict = {}
        for key in self.checkpointables:
            if isinstance(self.checkpointables[key], DistributedDataParallel):
                out_state_dict[key] = self.checkpointables[key].module.state_dict()
            else:
                out_state_dict[key] = self.checkpointables[key].state_dict()

        # Save checkpoint corresponding to current iteration.
        torch.save(out_state_dict, self.output_dir / f"checkpoint_final.pth")

    def resume(self) -> int:
        """
        Find the last saved checkpoint in :attr:`output_dir` (from a previous job)
        and load it to resume the job. This method will log a warning message if
        no checkpoint is found for loading.
        """

        logger.info(f"Attempting to resume job from {self.output_dir}...")

        last_ckpt_info_file = self.output_dir / "last_checkpoint.txt"
        if last_ckpt_info_file.exists():
            ckpt_path = last_ckpt_info_file.read_text().strip()
            logger.info(f"Found last checkpoint in {self.output_dir}: {ckpt_path}")
            return self.load(self.output_dir / ckpt_path)
        else:
            logger.warning(
                f"No checkpoint found in {self.output_dir} to resume job! "
                "Hopefully this is the beginning of a fresh job."
            )
            return 0

    def load(self, path: str | Path) -> int:
        """
        Load a saved checkpoint from a given file path. This method tries to find
        each of :attr:`checkpointables` in the file and load their state dict.

        Args:
            path: Path to a directory/checkpoint saved by :meth:`step`.

        Returns:
            Iteration corresponding to the loaded checkpoint (to resume training).
            If iteration is not found in file, this method will return -1.
        """

        # Each process will log a message after loading checkpoint.
        rank = dist.get_rank()

        logger.info(f"Rank {rank}: Loading checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu")
        iteration = checkpoint.pop("iteration", -1)

        # Keep flags of all checkpointables to lo which ones were not loaded.
        is_loaded = {key: False for key in self.checkpointables}

        # Load each checkpointable from checkpoint.
        for key in checkpoint:
            if key in self.checkpointables:
                logger.info(f"Rank {rank}: Loading {key} from {path}")

                if isinstance(self.checkpointables[key], DistributedDataParallel):
                    self.checkpointables[key].module.load_state_dict(checkpoint[key])
                else:
                    self.checkpointables[key].load_state_dict(checkpoint[key])

                is_loaded[key] = True
            else:
                logger.info(f"Rank {rank}: {key} not found in `checkpointables`.")

        not_loaded: list[str] = [key for key in is_loaded if not is_loaded[key]]
        if len(not_loaded) > 0:
            logger.info(f"Rank {rank}: Checkpointables not found in file: {not_loaded}")
        return iteration
