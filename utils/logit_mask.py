"""Logit masking for ImageNet-1k label-subset datasets (ImageNet-A/R/ES).

Manifests for these datasets carry canonical ImageNet-1k class indices
(0..999, resolved via scripts/data_prep/build_class_index.py — never a local
0..199 re-sort), so the frozen 1000-way pretrained head can be reused
unchanged. Restriction to the dataset's class subset happens here, at
evaluation time, by masking out logits for classes the dataset can't contain
— applied identically for every comparison method (finetune/randopt/probe).
"""
import json
from typing import List

import torch


def load_subset_indices(dataset_meta_path: str) -> List[int]:
    """Read `subset_indices` from a scripts/data_prep/make_shift_manifests.py
    sidecar (_meta/<dataset>_meta.json). None/absent means unmasked (full
    1000-way), e.g. ImageNet-Sketch."""
    with open(dataset_meta_path) as f:
        meta = json.load(f)
    return meta.get("subset_indices")


def build_mask(subset_indices: List[int], num_classes: int = 1000) -> torch.Tensor:
    """Additive mask: 0.0 for allowed classes, -inf elsewhere.
    `logits_masked = logits + build_mask(subset_indices)` then argmax as usual.
    `subset_indices=None` returns an all-zero (no-op) mask.
    """
    mask = torch.full((num_classes,), float("-inf"))
    if subset_indices is None:
        return torch.zeros(num_classes)
    mask[torch.as_tensor(subset_indices, dtype=torch.long)] = 0.0
    return mask
