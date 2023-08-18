# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch
import torchvision.transforms as T
from tqdm import tqdm

from meru import lorentz as L
from meru.data.evaluation import CocoCaptions, Flickr30kCaptions
from meru.evaluation.catalog import DatasetCatalog
from meru.models import MERU, CLIPBaseline
from meru.tokenizer import Tokenizer


class ZeroShotRetrievalEvaluator:
    """
    Evaluate trained models for zero-shot image and text retrieval. Image-text
    contrastive models like MERU and CLIP are optimized to perform retrieval
    within a batch during training. Hence, they can be seamlessly tranferred
    zero-shot for image-text retrieval without any task-specific adaptation.
    """

    def __init__(
        self,
        datasets: list[str],
        data_dir: str | Path,
        ks: list[int] = [5, 10],
        image_size: int = 224,
    ):
        """
        Args:
            datasets: List of dataset names to evaluate on, these names should be
                among supported datasets in `DatasetCatalog`.
            data_dir: Path to directory containing sub-directories of all datasets
                that are supported by the dataset catalog.
            ks: Top-k image/text to retrieve for calculating metrics.
            image_size: Resize images to this size for evaluation. All images
                are _squeezed_ in squares using bicubic interpolation.
        """
        self._datasets = datasets
        self._data_dir = Path(data_dir).resolve()
        self._ks = ks
        self._image_size = image_size
        super().__init__()

    @torch.inference_mode()
    def __call__(self, model: MERU | CLIPBaseline) -> dict[str, float]:
        model = model.eval()

        _resize = (self._image_size, self._image_size)
        image_transform = T.Compose(
            [T.Resize(_resize, T.InterpolationMode.BICUBIC), T.ToTensor()]
        )

        # Collect results per task in this dict:
        results_dict = {}

        for dname in self._datasets:
            data_loader = DatasetCatalog.build(
                dname, self._data_dir, "test", image_transform
            )

            # Encode all images and captions.
            encoded_data = _encode_dataset(data_loader, model)
            image_feats = encoded_data["image_feats"].to(model.device)
            text_feats = encoded_data["text_feats"].to(model.device)

            image_ids = torch.tensor(encoded_data["image_ids"])
            text_ids = torch.tensor(encoded_data["text_ids"])

            # Text-to-image retrieval: make mapping as {text_id: [sorted image_ids]}
            text_to_image_retr = {}

            for _ids, _queries in zip(text_ids.split(256), text_feats.split(256)):
                # Compute pairwise similarity depending on model type:
                if isinstance(model, MERU):
                    scores = L.pairwise_inner(_queries, image_feats, model.curv.exp())
                else:
                    scores = _queries @ image_feats.T

                # Scores are "higher is better" so sort their negative, and use
                # that order to obtain image IDs per caption.
                _retrieval_order = scores.argsort(dim=1, descending=True).cpu()
                retrieved_image_ids = image_ids[_retrieval_order]

                for _id, _image_ids in zip(_ids, retrieved_image_ids):
                    text_to_image_retr[_id.item()] = _image_ids.tolist()

            # Text-to-image retrieval: make mapping as {text_id: [sorted image_ids]}
            image_to_text_retr = {}

            for _ids, _queries in zip(image_ids.split(256), image_feats.split(256)):
                if isinstance(model, MERU):
                    scores = L.pairwise_inner(_queries, text_feats, model.curv.exp())
                else:
                    scores = _queries @ text_feats.T

                _retrieval_order = scores.argsort(dim=1, descending=True).cpu()
                retrieved_text_ids = text_ids[_retrieval_order]

                for _id, _text_ids in zip(_ids, retrieved_text_ids):
                    image_to_text_retr[_id.item()] = _text_ids.tolist()

            # Compute text-to-image and image-to-text recall@K for both datasets.
            for _k in self._ks:
                results_dict[f"{dname}_t2i_r{_k}"] = _compute_recall(
                    text_to_image_retr, encoded_data["text_to_image_gt"], _k
                )

            for _k in self._ks:
                results_dict[f"{dname}_i2t_r{_k}"] = _compute_recall(
                    image_to_text_retr, encoded_data["image_to_text_gt"], _k
                )

        return results_dict


def _compute_recall(
    predictions: dict[int, list[int]],
    ground_truth: dict[int, set[int]],
    K: int,
):
    """
    Compute recall @ K for COCO and Flickr30K image/text retrieval.

    Args:
        predictions: Dict with integer keys representing image (or text) IDs, and
            values being a ranked list of retrieved text (or image) IDs.
        ground_truth: Dict with integer keys representing image (or text) IDs
            (same as `predictions`) and values being a list of integer IDs
            of the paired text/images.
        K: Measure recall among Top-K retrievals.

    Returns:
        Single float value giving the average recall@k across all ground-truth.
    """

    num_correct_retrievals = 0.0
    for query_id, paired_ids in ground_truth.items():
        predictions_id = predictions.get(query_id, [])

        if set(predictions_id[:K]) & paired_ids:
            num_correct_retrievals += 1.0

    return 100.0 * num_correct_retrievals / len(ground_truth)


@torch.inference_mode()
def _encode_dataset(
    data_loader: CocoCaptions | Flickr30kCaptions,
    model: MERU | CLIPBaseline,
):
    """
    Extract image-text features and their instance IDs using a given dataset
    (COCO or Flickr30k) and a given model (MERU or CLIP).
    """

    encoded_data = {
        "image_ids": [],
        "text_ids": [],
        "image_feats": [],
        "text_feats": [],
        # Dict mapping as {image_id: {matching_text_ids} } and vice-versa.
        "image_to_text_gt": defaultdict(set),
        "text_to_image_gt": defaultdict(set),
    }

    tokenizer = Tokenizer()

    for inst in tqdm(data_loader, desc="Extracting image-text features"):
        # Add entries to ground-truth dict.
        image_id = inst["image_id"]
        encoded_data["image_to_text_gt"][image_id].update(inst["caption_ids"])

        for _id in inst["caption_ids"]:
            encoded_data["text_to_image_gt"][_id].add(image_id)

        image_feats = model.encode_image(
            inst["image"][None, ...].to(model.device), project=True
        )

        caption_tokens = tokenizer(inst["captions"])
        caption_feats = model.encode_text(caption_tokens, project=True)

        # Add current entries to extracted features and IDs.
        encoded_data["image_ids"].append(inst["image_id"])
        encoded_data["image_feats"].append(image_feats.cpu())
        encoded_data["text_ids"].extend(inst["caption_ids"])
        encoded_data["text_feats"].append(caption_feats.cpu())

    # shape: (dataset_size, model.embed_dim), (dataset_size, model.embed_dim)
    encoded_data["image_feats"] = torch.cat(encoded_data["image_feats"], dim=0)
    encoded_data["text_feats"] = torch.cat(encoded_data["text_feats"], dim=0)

    return encoded_data
