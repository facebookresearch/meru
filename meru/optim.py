# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LinearWarmupCosineDecayLR(LambdaLR):
    """
    A learning rate scheduler which linearly increases learning rate from 0
    LR, and further decreases it to zero by cosine decay.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ):
        """
        Args:
            optimizer: Wrapped optimizer.
            total_steps: Total epochs (or iterations) for training.
            warmup_steps: Number of first few steps to do linear warmup.
            last_epoch: The index of last step (epoch or iteration). We named
                it `last_epoch` instead of `last_step` to keep the naming
                consistent with other LR schedulers in PyTorch.
        """
        assert (
            warmup_steps < total_steps
        ), "Warmup steps should be less than total steps."

        self.tsteps = total_steps
        self.wsteps = warmup_steps
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step: int) -> float:
        if step < self.wsteps:
            # Linear warmup.
            multiplier = step / float(max(1, self.wsteps))
        else:
            # Cosine annealing decay.
            cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
            multiplier = math.cos(cos_factor * (math.pi / 2)) ** 2
        # Avoid negative learning rate.
        return max(0, multiplier)


def set_weight_decay_per_param(
    model: torch.nn.Module,
    weight_decay: float,
    gain_bias_decay: float | None = None,
    exclude_params: list[str] = [],
) -> list[dict]:
    """
    Set weight decay for trainable parameters of a model. This function allows
    setting different weight decay for normalization layers from rest of the
    model. The output param groups can be used to instantiate an optimizer.

    This function is adapted from the Torchvision ImageNet training script.

    Args:
        model: PyTorch module with trainable parameters.
        weight_decay: Weight decay for all params except normalization layers.
        gain_bias_decay: Weight decay for normalization layers and bias parameters
            everywhere in the model. If `None`, it defaults to `weight_decay`.
        exclude_params: List of parameter names whose weight decay should be zero.
            For example, this could be learnable softmax temperature parameter.
    """
    norm_classes = (
        torch.nn.modules.batchnorm._BatchNorm,
        torch.nn.LayerNorm,
        torch.nn.GroupNorm,
        torch.nn.modules.instancenorm._InstanceNorm,
        torch.nn.LocalResponseNorm,
    )

    gain_bias_decay = gain_bias_decay or weight_decay
    params = {"regular": [], "gain_bias": [], "excluded": []}
    params_weight_decay = {
        "regular": weight_decay,
        "gain_bias": gain_bias_decay,
        "excluded": 0.0,
    }

    # Hold references to parameters (tensors) in this set to avoid adding
    # duplicates, because some modules have shared weights (word embeddings)
    # and they may get counted twice -- PyTorch does not like it.
    already_added_parameters = set()

    def _add_params(module, prefix=""):
        for name, p in module.named_parameters(recurse=False):
            if not p.requires_grad or p in already_added_parameters:
                continue

            # Record current parameter as "visited".
            already_added_parameters.add(p)

            if any([exclude_name in name for exclude_name in exclude_params]):
                # Check the exclude substrings in parameter name.
                params["excluded"].append(p)
            elif isinstance(module, norm_classes) or "bias" in name:
                # Check the module type or `bias` in parameter name, this matching
                # is sufficient for ResNet-like and Transformer modules of PyTorch.
                params["gain_bias"].append(p)
            else:
                params["regular"].append(p)

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix != "" else child_name
            _add_params(child_module, prefix=child_prefix)

    _add_params(model)

    param_groups = []
    for key in params:
        if len(params[key]) > 0:
            param_groups.append(
                {"params": params[key], "weight_decay": params_weight_decay[key]}
            )
    return param_groups
