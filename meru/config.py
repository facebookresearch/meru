# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Core module for lazily instantiating objects from arbitrary configs. Many design
choices in this module are heavily influenced by Detectron2.
"""
from __future__ import annotations

import builtins
import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Callable

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch import nn, optim
from torch.distributed.algorithms.ddp_comm_hooks import default as ddph
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

import meru.utils.distributed as dist

__all__ = ["callable_to_str", "LazyCall", "LazyConfig", "LazyFactory"]


_CFG_PACKAGE_NAME = "meru._cfg_loader"
"""
Shared module namespace to import all config objects.
"""


@contextmanager
def _patch_import():
    """
    Enhance relative import statements in config files, so that they:
    1. locate files purely based on relative location, regardless of packages.
       e.g. you can import file without having __init__
    2. do not cache modules globally; modifications of module states has no side effect
    3. imported dict are turned into omegaconf.DictConfig automatically
    """
    old_import = builtins.__import__

    def find_relative_file(original_file, relative_import_path, level):
        cur_file = os.path.dirname(original_file)
        for _ in range(level - 1):
            cur_file = os.path.dirname(cur_file)
        cur_name = relative_import_path.lstrip(".")
        for part in cur_name.split("."):
            cur_file = os.path.join(cur_file, part)
        # NOTE: directory import is not handled. Because then it's unclear
        # if such import should produce python module or DictConfig. This can
        # be discussed further if needed.
        if not cur_file.endswith(".py"):
            cur_file += ".py"
        if not os.path.isfile(cur_file):
            raise ImportError(
                f"Cannot import name {relative_import_path} from "
                f"{original_file}: {cur_file} has to exist."
            )
        return cur_file

    def new_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            # Only deal with relative imports inside config files
            level != 0
            and globals is not None
            and (globals.get("__package__", "") or "").startswith(_CFG_PACKAGE_NAME)
        ):
            cur_file = find_relative_file(globals["__file__"], name, level)
            spec = importlib.machinery.ModuleSpec(
                _CFG_PACKAGE_NAME + "." + os.path.basename(cur_file),
                None,
                origin=cur_file,
            )
            module = importlib.util.module_from_spec(spec)
            module.__file__ = cur_file
            with open(cur_file) as f:
                content = f.read()
            exec(compile(content, cur_file, "exec"), module.__dict__)
            for name in fromlist:  # turn imported dict into DictConfig automatically
                val = DictConfig(module.__dict__[name], flags={"allow_objects": True})
                module.__dict__[name] = val
            return module
        return old_import(name, globals, locals, fromlist=fromlist, level=level)

    builtins.__import__ = new_import
    yield new_import
    builtins.__import__ = old_import


def callable_to_str(some_callable: Callable) -> str:
    # Return module and name of a callable (function or class) for OmegaConf.
    return f"{some_callable.__module__}.{some_callable.__qualname__}"


class LazyCall:
    """
    Wrap a callable so that when it's called, the call will not be executed, but
    returns a dict that describes the call. Only supports keyword arguments.
    """

    def __init__(self, target: Callable):
        if not callable(target):
            raise TypeError(f"LazyCall target must be a callable! Got {target}")
        self.T = target

    def target_str(self):
        return

    def __call__(self, **kwargs):
        # Pop `_target_` if it already exists in kwargs. This happens when the
        # callable target is changed while keeping everything else same.
        _ = kwargs.pop("_target_", None)

        # Put current target first; it reads better in printed/saved output.
        kwargs = {"_target_": callable_to_str(self.T), **kwargs}
        return DictConfig(content=kwargs, flags={"allow_objects": True})


class LazyConfig:
    """
    Provide methods to save, load, and overrides an omegaconf config object
    which may contain definition of lazily-constructed objects.
    """

    @staticmethod
    def load(filename: str | Path) -> DictConfig:
        """
        Load a config file (either Python or YAML).

        Args:
            filename: absolute path or relative path w.r.t. current directory.
        """
        filename = str(filename).replace("/./", "/")
        if os.path.splitext(filename)[1] not in [".py", ".yaml", ".yml"]:
            raise ValueError(f"Config file {filename} has to be a python or yaml file.")

        if filename.endswith(".py"):
            with _patch_import():
                # Record the filename
                module_namespace = {
                    "__file__": filename,
                    "__package__": _CFG_PACKAGE_NAME + "." + os.path.basename(filename),
                }
                with open(filename) as f:
                    content = f.read()
                # Compile first with filename to make filename appear in stacktrace
                exec(compile(content, filename, "exec"), module_namespace)

            # Collect final objects in config:
            ret = OmegaConf.create(flags={"allow_objects": True})

            for name, value in module_namespace.items():
                # Ignore "private" variables (starting with underscores).
                if name.startswith("_"):
                    continue

                if isinstance(value, (DictConfig, dict)):
                    value = DictConfig(value, flags={"allow_objects": True})
                    ret[name] = value

                if isinstance(value, (ListConfig, list)):
                    value = ListConfig(value, flags={"allow_objects": True})
                    ret[name] = value
        else:
            with open(filename) as f:
                obj = yaml.unsafe_load(f)
            ret = OmegaConf.create(obj, flags={"allow_objects": True})

        return ret

    @staticmethod
    def save(cfg: DictConfig, filename: str) -> None:
        """
        Save a config object as YAML file. (same as :meth:`OmegaConf.save`).
        """
        OmegaConf.save(cfg, filename, resolve=False)

    @staticmethod
    def apply_overrides(cfg: DictConfig, overrides: list[str]) -> DictConfig:
        """
        Return a new config by applying overrides (provided as dotlist). See
        https://hydra.cc/docs/advanced/override_grammar/basic/ for dotlist syntax.
        """
        return OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))


class LazyFactory:
    """
    Provides a clean interface to easily construct essential objects from input
    lazy configs (omegaconf): dataloader, model, optimizer, and LR scheduler.
    """

    @staticmethod
    def build_dataloader(cfg: DictConfig):
        # Instantiate dataset and wrap in dataloader.
        return DataLoader(
            instantiate(cfg.dataset),
            num_workers=cfg.train.num_workers,
            batch_size=cfg.train.total_batch_size // dist.get_world_size(),
            drop_last=True,
            pin_memory=True,
        )

    @staticmethod
    def build_model(cfg: DictConfig, device: torch.device | None = None):
        # Get the current device as set for current distributed process.
        # Check `launch` function in `meruu.utils.distributed` module.
        device = device or torch.cuda.current_device()
        model = instantiate(cfg.model).to(device)

        # Wrap model in DDP if using more than one GPUs.
        if dist.get_world_size() > 1:
            model = DistributedDataParallel(model, [device], **cfg.train.ddp)

            # Optionally add FP16 compression hook with AMP.
            if cfg.train.amp and cfg.train.ddp_fp16_compression:
                model.register_comm_hook(state=None, hook=ddph.fp16_compress_hook)

        return model

    @staticmethod
    def build_optimizer(cfg: DictConfig, model: nn.Module) -> optim.Optimizer:
        # Iterate named parameters of the model. Use internal `module` for DDP.
        if isinstance(model, DistributedDataParallel):
            model = model.module

        # Add model as an input to `set_weight_decay_per_param`.
        cfg.optim.optimizer.params.model = model
        return instantiate(cfg.optim.optimizer)

    @staticmethod
    def build_lr_scheduler(cfg: DictConfig, optimizer: optim.Optimizer):
        return instantiate(cfg.optim.lr_scheduler, optimizer=optimizer)
