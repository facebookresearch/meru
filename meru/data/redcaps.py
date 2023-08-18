# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import copy
import glob
import random
from typing import Callable

import webdataset as wds
import wordsegment as ws
from loguru import logger
from torch.utils.data import IterableDataset
from torchvision import transforms as T

import meru.utils.distributed as dist

ws.load()


class ImageTextWebDataset(IterableDataset):
    """
    Iterable dataset that serves instances from a lot of TAR file shards.
    This class uses `WebDataset <https://github.com/webdataset/webdataset>`_
    internally, and expects TAR files to be arranged in a compatible format.
    """

    def __init__(
        self,
        tarfiles: str | list[str],
        mapper: Callable,
        buffer_size: int = 5000,
        infinite_stream: bool = True,
        seed: int = 0,
    ):
        """
        Args:
            tarfiles: Path(s) or glob-patterns for TAR files in WebDataset format.
            mapper: A callable to transform a single dataset dict (image and
                annotations). May implement data augmentation and tokenization.
            buffer_size: Size of the internal buffer of instances. Data is read
                sequentially from TAR files into this buffer and served randomly.
                Shuffling will be disabled if this is set to zero.
            infinite_stream: Yield an infinite stream of instances if this is
                True. In such cases, the user must terminate this iterator manually
                (e.g. run a fixed sized for-loop in training code).
            seed: Random seed for buffer shuffling. If provided, this dataloader
                will load batches deterministically across different runs (only if
                batch size and number of GPUs/CPUs are same). This seed can either
                be same or different per GPU process for multi-GPU training.
        """
        super().__init__()
        self.mapper = mapper
        self.buffer_size = buffer_size
        self.infinite_stream = infinite_stream
        self.seed = seed

        # Convert a single path (glob) to a list.
        if isinstance(tarfiles, str):
            tarfiles = [tarfiles]

        # Expand all glob patterns to list a full list of individual TAR files.
        self.tarfiles = []
        for _path in tarfiles:
            for _single_glob in _path.split():
                self.tarfiles.extend(glob.glob(_single_glob))

        # Sort paths; webdataset performs a deterministic shuffle (internally).
        self.tarfiles = sorted(self.tarfiles)
        logger.info(f"{self.__class__.__name__} found {len(self.tarfiles)} TARs.")

        # Shard the TAR file paths as per number of GPU processes to avoid loading
        # duplicates.
        _rank, _world_size = dist.get_rank(), dist.get_world_size()
        self.tarfiles = self.tarfiles[_rank::_world_size]
        logger.info(f"RANK {_rank} will load {len(self.tarfiles)} TARs.")

    def __iter__(self):
        rng = random.Random(self.seed)
        pipeline = wds.DataPipeline(
            wds.SimpleShardList(self.tarfiles, seed=self.seed),
            wds.split_by_worker,
            wds.tarfile_to_samples(),
        )

        if self.buffer_size > 1:
            pipeline.append(
                wds.shuffle(self.buffer_size, initial=self.buffer_size, rng=rng),
            )

        # Decode images using PIL and apply custom mapper.
        pipeline.append(wds.decode("pil", handler=wds.warn_and_continue))
        pipeline.append(wds.map(self.mapper))

        if self.infinite_stream:
            # Sample an infinite stream of dataset dicts.
            while True:
                pipeline_copy = copy.deepcopy(pipeline)
                yield from pipeline_copy
        else:
            # Run for one epoch and stop:
            yield from pipeline


class RedCapsTarMapper:
    """
    Mapper to pre-process image-text instances from RedCaps TAR files.
    """

    def __init__(
        self,
        image_transform: list[Callable] = [
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
        ],
        redcaps_prefix_prob: float = 0.5,
    ):
        """
        Args:
            image_transform: List of image transformations from torchvision.
            redcaps_prefix_prob: Probability of randomly prefixing subreddit
                name (word-segmented) to the caption for RedCaps instances.
        """
        self.image_transform = T.Compose(image_transform)
        self.redcaps_prefix_prob = redcaps_prefix_prob

        # Make a dictionary of word-segmented subreddit names.
        # (e.g. itookapicture -> i took a picture).
        self._subreddit_segs = {}

    def __call__(self, dataset_dict: dict):
        json_field = dataset_dict["json"]
        caption = json_field["caption"]

        # Randomly prefix the caption with (segmented) subreddit name.
        if random.random() <= self.redcaps_prefix_prob:
            subreddit = json_field["subreddit"]

            # Segment the subreddit name once and cache it.
            if subreddit not in self._subreddit_segs:
                segmented = ws.segment(ws.clean(subreddit))
                self._subreddit_segs[subreddit] = " ".join(segmented)

            # Prefix the segmented subreddit name.
            caption = self._subreddit_segs[subreddit] + " : " + caption

        return {
            "__key__": dataset_dict["__key__"],
            "image": self.image_transform(dataset_dict["jpg"]),
            "text": caption,
        }
