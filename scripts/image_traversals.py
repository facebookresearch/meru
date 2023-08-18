# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Perform image traversals using a trained MERU or CLIP model, and a pool of
text (and their encoded text representations).
"""
from __future__ import annotations

import argparse
import json

import torch
from PIL import Image
from torchvision import transforms as T

from meru import lorentz as L
from meru.config import LazyConfig, LazyFactory
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer
from meru.utils.checkpointing import CheckpointManager


parser = argparse.ArgumentParser(description=__doc__)
_AA = parser.add_argument
_AA("--checkpoint-path", help="Path to checkpoint of a trained MERU/CLIP model.")
_AA("--train-config", help="Path to train config (.yaml/py) for given checkpoint.")
_AA("--image-path", help="Path to an image (.jpg) for perfoming traversal.")
_AA("--steps", type=int, default=50, help="Number of traversal steps.")


def interpolate(model, feats: torch.Tensor, root_feat: torch.Tensor, steps: int):
    """
    Interpolate between given feature vector and `[ROOT]` depending on model type.
    """

    # Linear interpolation between root and image features. For MERU, this happens
    # in the tangent space of the origin.
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())

    interp_feats = [
        torch.lerp(root_feat, feats, weight.item())
        for weight in torch.linspace(0.0, 1.0, steps=steps)
    ]
    interp_feats = torch.stack(interp_feats)

    # Lift on the Hyperboloid (for MERU), or L2 normalize (for CLIP).
    if isinstance(model, MERU):
        feats = L.log_map0(feats, model.curv.exp())
        interp_feats = L.exp_map0(interp_feats, model.curv.exp())
    else:
        interp_feats = torch.nn.functional.normalize(interp_feats, dim=-1)

    # Reverse the traversal order: (image first, root last)
    return interp_feats.flip(0)


def calc_scores(
    model, image_feats: torch.Tensor, text_feats: torch.Tensor, has_root: bool
):
    """
    Calculate similarity scores between the given image and text features depending
    on model type.

    Args:
        has_root: Flag to indicate whether the last text embedding (at dim=0)
            is the `[ROOT]` embedding.
    """

    if isinstance(model, MERU):
        scores = L.pairwise_inner(image_feats, text_feats, model.curv.exp())

        # For MERU, exclude text embeddings that do not entail the given image.
        _aper = L.half_aperture(text_feats, model.curv.exp())
        _oxy_angle = L.oxy_angle(
            text_feats[:, None, :], image_feats[None, :, :], model.curv.exp()
        )
        entailment_energy = _oxy_angle - _aper[..., None]

        # Root entails everything.
        if has_root:
            entailment_energy[-1, ...] = 0

        # Set a large negative score if text does not entail image.
        scores[entailment_energy.T > 0] = -1e12
        return scores
    else:
        # model is not needed here.
        return image_feats @ text_feats.T


@torch.inference_mode()
def get_text_feats(model: MERU | CLIPBaseline) -> tuple[list[str], torch.Tensor]:
    # Get all captions, nouns, and ajectives collected from pexels.com website
    pexels_text = json.load(open("assets/pexels_text.json"))

    # Use very simple prompts for noun and adjective tags.
    tokenizer = Tokenizer()
    NOUN_PROMPT = "a photo of a {}."
    ADJ_PROMPT = "this photo is {}."

    all_text_feats = []

    # Tokenize and encode captions.
    caption_tokens = tokenizer(pexels_text["captions"])
    all_text_feats.append(model.encode_text(caption_tokens, project=True))

    # Tokenize and encode prompts filled with tags.
    # Extract features of all captions and tags.
    noun_prompt_tokens = tokenizer(
        [NOUN_PROMPT.format(tag) for tag in pexels_text["nouns"]]
    )
    all_text_feats.append(model.encode_text(noun_prompt_tokens, project=True))

    adj_prompt_tokens = tokenizer(
        [ADJ_PROMPT.format(tag) for tag in pexels_text["adjectives"]]
    )
    all_text_feats.append(model.encode_text(adj_prompt_tokens, project=True))

    all_text_feats = torch.cat(all_text_feats, dim=0)
    all_pexels_text = [
        *pexels_text["captions"],
        *pexels_text["nouns"],
        *pexels_text["adjectives"],
    ]
    return all_pexels_text, all_text_feats


@torch.inference_mode()
def main(_A: argparse.Namespace):
    # Get the current device (this will be `cuda:0` here by default) or use CPU.
    device = (
        torch.cuda.current_device()
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # Create the model using training config and load pre-trained weights.
    _C_TRAIN = LazyConfig.load(_A.train_config)
    model = LazyFactory.build_model(_C_TRAIN, device).eval()

    CheckpointManager(model=model).load(_A.checkpoint_path)

    if isinstance(model, MERU):
        root_feat = torch.zeros(_C_TRAIN.model.embed_dim, device=device)
    else:
        # CLIP model checkpoint should have the 'root' embedding.
        root_feat = torch.load(_A.checkpoint_path)["root"].to(device)

    # If no external text features are provided, use captions/tags from pexels.
    text_pool, text_feats_pool = get_text_feats(model)

    # Add [ROOT] to the pool of text feats.
    text_pool.append("[ROOT]")
    text_feats_pool = torch.cat([text_feats_pool, root_feat[None, ...]])

    # ------------------------------------------------------------------------
    print(f"\nPerforming image traversals with source: {_A.image_path}...")
    # ------------------------------------------------------------------------
    image = Image.open(_A.image_path).convert("RGB")

    image_transform = T.Compose(
        [T.Resize(224, T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor()]
    )
    image = image_transform(image).to(device)
    image_feats = model.encode_image(image[None, ...], project=True)[0]

    interp_feats = interpolate(model, image_feats, root_feat, _A.steps)
    nn1_scores = calc_scores(model, interp_feats, text_feats_pool, has_root=True)

    nn1_scores, _nn1_idxs = nn1_scores.max(dim=-1)
    nn1_texts = [text_pool[_idx.item()] for _idx in _nn1_idxs]

    # De-duplicate retrieved texts (multiple points may have same NN) and print.
    print(f"Texts retrieved from [IMAGE] -> [ROOT] traversal:")
    unique_nn1_texts = []
    for _text in nn1_texts:
        if _text not in unique_nn1_texts:
            unique_nn1_texts.append(_text)
            print(f"  - {_text}")


if __name__ == "__main__":
    _A = parser.parse_args()
    main(_A)
