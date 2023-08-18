# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time


class Timer:
    """
    Simple timer to record time per iteration and estimate ETA of job completion.
    Use methods :meth:`tic` and :meth:`toc` in the training job.
    """

    def __init__(self, start_iteration: int = 1, total_iterations: int | None = None):
        """
        Args:
            start_iteration: Iteration from which counting should be started/resumed.
            total_iterations: Total number of iterations. ETA will not be tracked (will
                remain "N/A") if this is not provided.
        """
        # We decrement by 1 because `iteration` changes increment during
        # an iteration (for example, will change from 0 -> 1 on iteration 1).
        self.iteration = start_iteration - 1
        self.total_iters = total_iterations

        # Keep a record of time deltas between past 100 tic-toc function calls.
        self.deltas = [0.0] * 100

        self._start_time = time.perf_counter()

    def tic(self) -> None:
        """Start recording time: call at the beginning of iteration."""
        self._start_time = time.perf_counter()

    def toc(self) -> None:
        """Stop recording time: call at the end of iteration."""
        self.deltas.append(time.perf_counter() - self._start_time)
        self.deltas = self.deltas[1:]
        self.iteration += 1

    @property
    def eta_hhmm(self) -> str:
        """Return ETA in the form of `hh mm` string."""

        if self.total_iters:
            avg_time = sum(self.deltas) / len(self.deltas)
            eta_sec = int(avg_time * (self.total_iters - self.iteration))
            return f"{eta_sec // 3600}h {((eta_sec % 3600) // 60):02d}m"
        else:
            return "N/A"
