# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from meru.config import LazyCall as L
from meru.evaluation.classification import LinearProbeClassificationEvaluator


evaluator = L(LinearProbeClassificationEvaluator)(
    datasets=[
        "food101",
        "cifar10",
        "cifar100",
        "cub2011",
        "sun397",
        "cars",
        "aircraft",
        "dtd",
        "pets",
        "caltech101",
        "flowers102",
        "stl10",
        "eurosat",
        "resisc45",
        "country211",
        "mnist",
        "pcam",
        "clevr",
        "sst2",
    ],
    data_dir="datasets/eval",
    image_size=224,
    num_workers=4,
)
