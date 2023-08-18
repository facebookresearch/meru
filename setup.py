# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import find_packages, setup

setup(
    name="meru",
    version="1.0",
    python_requires=">=3.9",
    zip_safe=True,
    packages=find_packages(include=["meru"]),
)
