# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import timm
import torch
from torch import nn


@timm.models.register_model
def vit_small_mocov3_patch16_224(**kwargs):
    """
    Small Vision Transformer used by MoCo-v3 (https://arxiv.org/abs/2104.02057)
    This model has 12 heads instead of 6, gives better empirical performance.
    """

    return timm.models.vision_transformer._create_vision_transformer(
        "vit_small_patch16_224",
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=12,
        **kwargs,
    )


def build_timm_vit(
    arch: str,
    global_pool: str = "token",
    use_sincos2d_pos: bool = True,
    grad_checkpointing: bool = False,
):
    """
    Build a Vision Transformer (ViT) image encoder from timm.

    Args:
        global_pool: How to perform global pooling after final layer? If this is
            `token` (default), we use the [cls] token. For `avg` we perform
            global average pooling like ConvNets.
        use_sincos2d_pos: Use a fixed 2D sine-cosine position embedding. If this
            is False; position embeddings are randomly initialized (and learned).
    """

    _supported = timm.list_models("vit_*")
    if arch not in _supported:
        raise ValueError(f"{arch} is not a supported ViT, choose: {_supported}")

    model = timm.create_model(
        arch,
        # `num_classes = 0` does not create the final classification head.
        num_classes=0,
        global_pool=global_pool,
        #
        # Do not use [CLS] token for models that use global average pooling.
        class_token=global_pool == "token",
        #
        # Use LayerNorm with default `eps = 1e-5` (timm uses 1e-6, not sure why)
        norm_layer=nn.LayerNorm,
    )
    model.set_grad_checkpointing(grad_checkpointing)

    # Set `width` attribute to access in model.
    model.width = model.embed_dim

    # ------------------------------------------------------------------------
    # 2D sine-cosine embedding for Vision Transformers. This implementation
    # is adapted from MoCo-v3 (https://github.com/facebookresearch/moco-v3) and
    # it produces exactly same embeddings.
    # ------------------------------------------------------------------------
    if use_sincos2d_pos:
        h, w = model.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)

        assert (
            model.embed_dim % 4 == 0
        ), "ViT embed_dim must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = model.embed_dim // 4

        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (10000.0**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        if global_pool == "token":
            pe_token = torch.zeros([1, 1, model.embed_dim], dtype=torch.float32)
            pos_emb = torch.cat([pe_token, pos_emb], dim=1)

        # Override position embedding.
        model.pos_embed.data.copy_(pos_emb)
        model.pos_embed.requires_grad = False

    return model
