# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate a trained model using implementations from `meru.evaluation` module.
"""
from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from loguru import logger

from meru.config import LazyConfig, LazyFactory
from meru.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--config", help="Path to an evaluation config file (.py)")
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")


def main(_A: argparse.Namespace):
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create evaluation and training config objects.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    _C = LazyConfig.load(_A.config)
    logger.info(OmegaConf.to_yaml(_C))

    logger.info("Command line args:")
    for arg in vars(_A):
        logger.info(f"{arg:<20}: {getattr(_A, arg)}")

    logger.info(f"Evaluating checkpoint in {_A.checkpoint_path}...")

    # Create a fresh model and evaluator for every checkpoint, so the evaluator
    # is free to modify the model weights (e.g. remove projection layers).
    evaluator = instantiate(_C.evaluator)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()
    CheckpointManager(model=model).load(_A.checkpoint_path)

    results_dict = evaluator(model)

    # Log results for copy-pasting to spreadsheet, including checkpoint path.
    header = ",".join(results_dict.keys())
    numbers = ",".join([f"{num:.1f}" for num in results_dict.values()])

    logger.info(f"copypaste: {_A.checkpoint_path}")
    logger.info(f"\ncopypaste below:\n{header}\n{numbers}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
