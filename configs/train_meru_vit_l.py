# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Each config file should have four dicts or OmegaConf objects:
`dataset`, `model`, `optim`, and `train`.

User can compose config files by importing these objects and overriding specific
parameters. See examples in other training configs.

Reference: https://detectron2.readthedocs.io/en/latest/tutorials/lazyconfigs.html
"""

from torch.optim import AdamW
from torchvision import transforms as T

from meru.config import LazyCall as L
from meru.data.redcaps import RedCapsTarMapper, ImageTextWebDataset
from meru.encoders.image_encoders import build_timm_vit
from meru.encoders.text_encoders import TransformerTextEncoder
from meru.models import MERU
from meru.optim import LinearWarmupCosineDecayLR, set_weight_decay_per_param


dataset = L(ImageTextWebDataset)(
    tarfiles=["datasets/redcaps/*.tar"],
    mapper=L(RedCapsTarMapper)(
        image_transform=[
            L(T.RandomResizedCrop)(
                size=224, scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC
            ),
            L(T.ToTensor)(),
        ],
    ),
    buffer_size=4000,
    seed="${..train.seed}",
)

model = L(MERU)(
    visual=L(build_timm_vit)(
        arch="vit_large_patch16_224",
        global_pool="token",
        use_sincos2d_pos=True,
    ),
    textual=L(TransformerTextEncoder)(
        arch="L12_W512", vocab_size=49408, context_length=77
    ),
    embed_dim=512,
    curv_init=1.0,
    learn_curv=True,
    entail_weight=0.2,
)

# AdamW with no weight decay for norm, bias, and other learnable scalars.
optim = dict(
    optimizer=L(AdamW)(
        params=L(set_weight_decay_per_param)(
            weight_decay="${..weight_decay}",
            gain_bias_decay=0.0,
            exclude_params=[
                "logit_scale", "visual_alpha", "textual_alpha", "curv"
            ],
        ),
        lr=5e-4,
        betas=(0.9, 0.98),
        weight_decay=0.2,
    ),
    lr_scheduler=L(LinearWarmupCosineDecayLR)(
        total_steps="${...train.num_iterations}", warmup_steps=4000
    ),
)


# Other parameters useful for training script.
train = dict(
    seed=0,
    amp=True,
    total_batch_size=2048,
    num_iterations=120000,
    cudnn_benchmark=True,
    cudnn_deterministic=False,
    num_workers=4,
    ddp=dict(  # options for DistributedDataParallel
        broadcast_buffers=False, static_graph=True
    ),
    ddp_fp16_compression=True,
)
