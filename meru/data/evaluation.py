# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

import tensorflow_datasets as tfds
import torch
from PIL import Image
from torch.utils.data import Dataset, IterDataPipe
from torch.utils.data.datapipes.iter import ShardingFilter
from torchvision.datasets import ImageFolder


class CocoCaptions(Dataset):
    """
    COCO captions dataset. Homepage: https://cocodataset.org
    """

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None):
        """
        Args:
            root: Dataset root directory. It should contain image directories
                named `train2017` and `val2017`, and a separate directory
                containing caption annotations JSON file.
            split: Name of 2017 split to load, one of `{train, val}`.
            transform: A function/transform that takes in an PIL image and
                returns a transformed version.
        """
        super().__init__()
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Read annotations for the given split.
        json_path = self.root / "annotations" / f"captions_{split}2017.json"
        coco_json = json.load(open(json_path))

        # Build a temporary mapping between image ID and captions.
        image_id_to_anns = defaultdict(list)
        for ann in coco_json["annotations"]:
            image_id_to_anns[ann["image_id"]].append(ann)

        # Convert the above mapping to list of tuples formatted as:
        # `(image_id, image_path, list[caption_ids], list[caption])`.
        self.samples = [
            (
                image_id,
                self.root / f"{split}2017" / f"{image_id:0>12d}.jpg",
                [ann["id"] for ann in anns],
                [ann["caption"] for ann in anns],
            )
            for image_id, anns in image_id_to_anns.items()
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        image_id, image_path, caption_ids, captions = self.samples[idx]

        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image_id": image_id,
            "caption_ids": caption_ids,
            "image": image,
            "captions": captions,
        }


class Flickr30kCaptions(CocoCaptions):
    """
    Flickr30K captions dataset.

    Karpathy split JSON can be downloaded from this webpage:
    https://cs.stanford.edu/people/karpathy/deepimagesent/
    """

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None):
        """
        Args:
            root: Dataset root directory. It should contain a JSON file named
                `dataset_flickr30k.json` containing Karpathy splits, and a
                directory named `flickr30k_images` with all images (~31K).
            split: Name of split to load, one of `{train, val, test}`.
            transform: A function/transform that takes in an PIL image and
                returns a transformed version.
        """
        self.root = Path(root)
        self.split = split
        self.transform = transform

        # Read annotations and keep only those belonging to specified split.
        flickr_json = json.load(open(self.root / "dataset_flickr30k.json"))

        # Convert the filtered list of tuples formatted as:
        # `(image_id, image_path, list[caption_ids], list[caption])`.
        # Only keep images that belong to required split.
        self.samples = [
            (
                int(ann["filename"][:-4]),
                self.root / "flickr30k_images" / ann["filename"],
                ann["sentids"],
                [entry["raw"] for entry in ann["sentences"]],
            )
            for ann in flickr_json["images"]
            if ann["split"] == split
        ]


class ImageNet(ImageFolder):
    """
    Lightweight wrapper over Torchvision `ImageFolder` to load ImageNet dataset.
    """

    def __init__(self, root: str, split: str = "train", **kwargs):
        super().__init__(str(Path(root) / split), **kwargs)


class TfdsWrapper(IterDataPipe):
    """
    Minimal wrapper on `tensorflow-datasets` to serve `(image, label)`
    tuples for image classification datasets. This wrapper enables a consistent
    output format with dataset implementations from the Torchvision library.
    """

    def __init__(
        self,
        name: str,
        root: str | Path,
        split: str,
        transform: Callable | None = None,
    ):
        """
        Args:
            name: Name of a dataset supported by Tensorflow datasets. See
                https://www.tensorflow.org/datasets/catalog/overview for details.
            root: Dataset root directory. This is passed to the `data_dir`
                argument of `tfds.load`. All datasets are auto-downloaded and
                cached in this directory.
            split: Which dataset split to load. This should be one of the official
                splits for the given dataset.
            transform: A function/transform that takes in an PIL image and
                returns a transformed version.
        """

        super().__init__()
        self.name = name
        self.split = split
        self.transform = transform

        dset = tfds.load(name, split=split, data_dir=root)
        dset = tfds.as_numpy(dset)

        # Record length of the dataset before further wrapping.
        self._length = len(dset)

        # Wrap the tensorflow dataset with `IterDataPipe` and apply sharding filter
        # to avoid duplicates when multiple CPU workers are used in DataLoader.
        self.dset = ShardingFilter(dset)

    def __repr__(self):
        return f"TfDatasetWrapper(name={self.name}, split={self.split})"

    def __len__(self):
        return self._length

    def __iter__(self) -> Iterator[tuple[Image.Image, torch.Tensor]]:
        for instance in self.dset:
            # Convert numpy arrays: image (PIL.Image) and label (tensor).
            # Handle special case with MNIST images.
            if self.name == "mnist":
                image = Image.fromarray(instance["image"][..., 0], mode="L")
            else:
                image = Image.fromarray(instance["image"])

            image = image.convert("RGB")
            label = torch.tensor(instance["label"])

            if self.transform is not None:
                image = self.transform(image)

            yield image, label


class CLEVRCounts(TfdsWrapper):
    """
    CLEVR-Counts image classification dataset. Counting the number of objects in
    a scene is framed as a classification task. This task was included in the
    Visual Task Adaptation Benchmark (VTAB), and used in CLIP evaluation suite.
    """

    def __init__(self, root: str | Path, split: str, transform: Callable | None = None):
        super().__init__("clevr", root, split, transform)

        # Convert counts to contiguous labels.
        self._labels = [10, 3, 4, 5, 6, 7, 8, 9]

    def __iter__(self) -> Iterator[tuple[Image.Image, torch.Tensor]]:
        for instance in self.dset:
            image = Image.fromarray(instance["image"]).convert("RGB")
            num_objects = len(instance["objects"]["color"])
            label = torch.tensor(self._labels.index(num_objects))

            if self.transform is not None:
                image = self.transform(image)

            yield image, label
