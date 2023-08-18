# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable

from torchvision.datasets import Country211, FGVCAircraft, RenderedSST2

import meru.data.evaluation as DE


class DatasetCatalog:
    """
    A catalog and constructor for all supported evaluation datasets in the package.
    This class holds essential information for each dataset in class attributes
    and a single method :meth:`build` that constructs datasets (iterable objects).
    """

    # Names of all datasets available for evaluation, and a partial callable that
    # shows which library implementation we are using. All datasets are either
    # sourced from `tensorflow-datasets` library or `torchvision.datasets`.
    CONSTRUCTORS: dict[str, Callable] = {
        #
        # Datasets for image classification evaluation:
        "imagenet": DE.ImageNet,
        "food101": partial(DE.TfdsWrapper, name="food101"),
        "cifar10": partial(DE.TfdsWrapper, name="cifar10"),
        "cifar100": partial(DE.TfdsWrapper, name="cifar100"),
        "cub2011": partial(DE.TfdsWrapper, name="caltech_birds2011"),
        "sun397": partial(DE.TfdsWrapper, name="sun397/standard-part1-120k"),
        "cars": partial(DE.TfdsWrapper, name="cars196"),
        "aircraft": partial(FGVCAircraft, download=True),
        "dtd": partial(DE.TfdsWrapper, name="dtd"),
        "pets": partial(DE.TfdsWrapper, name="oxford_iiit_pet"),
        "caltech101": partial(DE.TfdsWrapper, name="caltech101"),
        "flowers102": partial(DE.TfdsWrapper, name="oxford_flowers102"),
        "stl10": partial(DE.TfdsWrapper, name="stl10"),
        "eurosat": partial(DE.TfdsWrapper, name="eurosat"),
        "resisc45": partial(DE.TfdsWrapper, name="resisc45"),
        "country211": partial(Country211, download=True),
        "mnist": partial(DE.TfdsWrapper, name="mnist"),
        "clevr": DE.CLEVRCounts,
        "pcam": partial(DE.TfdsWrapper, name="patch_camelyon"),
        "sst2": partial(RenderedSST2, download=True),
        # --------------------------------------------------------------------
        #
        # Datasets for image and text retrieval evaluation:
        "coco": DE.CocoCaptions,
        "flickr30k": DE.Flickr30kCaptions,
    }

    # List of names of the official splits, in order: `[train, val, test]`
    # Datasets that have empty train/val splits do not support them.
    SPLITS: dict[str, list[str]] = {
        #
        # Datasets for image classification evaluation:
        "imagenet": ["train", "", "val"],
        "food101": ["train[:90%]", "train[90%:]", "validation"],
        "cifar10": ["train[:90%]", "train[90%:]", "test"],
        "cifar100": ["train[:90%]", "train[90%:]", "test"],
        "cub2011": ["train[:80%]", "train[80%:]", "test"],
        "sun397": ["train[:80%]", "train[80%:]", "test"],
        "cars": ["train[:80%]", "train[80%:]", "test"],
        "aircraft": ["train", "val", "test"],
        "dtd": ["train", "validation", "test"],
        "pets": ["train[:80%]", "train[80%:]", "test"],
        "caltech101": ["train[:80%]", "train[80%:]", "test"],
        "flowers102": ["train", "validation", "test"],
        "stl10": ["train[:80%]", "train[80%:]", "test"],
        "eurosat": ["train[:5000]", "train[5000:10000]", "train[10000:15000]"],
        "resisc45": ["train[:10%]", "train[10%:20%]", "train[20%:]"],
        "country211": ["train", "valid", "test"],
        "mnist": ["train[:80%]", "train[80%:]", "test"],
        "clevr": ["train[:4500]", "train[4500:5000]", "validation[:5000]"],
        "pcam": ["train", "validation", "test"],
        "sst2": ["train", "val", "test"],
        # --------------------------------------------------------------------
        #
        # Datasets for image and text retrieval evaluation:
        "coco": ["", "", "val"],
        "flickr30k": ["", "", "test"],
    }

    # fmt: off
    # Number of classes in each dataset for image classification evaluation. We
    # use mean per-class accuracy to account for any label imbalance.
    # See `meru.evaluation.classification` for more details.
    NUM_CLASSES: dict[str, int] = {
        "imagenet": 1000, "food101": 101, "cifar10": 10, "cifar100": 100,
        "cub2011": 200, "sun397": 397, "cars": 196, "aircraft": 100,
        "dtd": 47, "pets": 37, "caltech101": 102, "flowers102": 102,
        "stl10": 10, "eurosat": 10, "resisc45": 45, "country211": 211,
        "mnist": 10, "clevr": 8, "pcam": 2, "sst2": 2,
    }
    # fmt: on

    @classmethod
    def build(
        cls, name: str, root: str | Path, split: str, transform: Callable | None = None
    ):
        if name not in cls.CONSTRUCTORS:
            supported = sorted(cls.CONSTRUCTORS.keys())
            raise ValueError(f"{name} is not among supported datasets: {supported}")

        if split not in ["train", "val", "test"]:
            raise ValueError(f"split must be one of [train, val, test], not {split}")

        # Change the root directory for some Torchvision datasets because their
        # auto-download location may clutter the dataset directory.
        if name in ["aircraft", "country211", "imagenet", "sst2", "coco", "flickr30k"]:
            root = str(Path(root) / name)

        # Map split from [train, val, test] to official name.
        _idx = ["train", "val", "test"].index(split)
        split = cls.SPLITS[name][_idx]

        return cls.CONSTRUCTORS[name](root=root, split=split, transform=transform)
