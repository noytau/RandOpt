"""Clean ImageNet-1k dataset handler (evaluation side).

Same manifest schema and loader as ImageNet-C (data_handlers/imagenet_c.py).
Points at the official ImageNet val10 manifest — physically disjoint from
the train10 manifest used for perturbation scoring (scripts/randopt_shift.py
--train_manifest) — so scoring and this eval target never share images.
Full 1000-class label space, no logit mask.

Note: this manifest labels every entry "train" regardless of role (a
leftover of how it was generated for the earlier SSL series); load_data
must be called with split="train" here, not "test".
"""
import os

from .imagenet_c import ImageNetCHandler

_MANIFEST = os.path.join("data", "imagenet_val10", "data.json")


class ImageNetHandler(ImageNetCHandler):
    name = "imagenet"
    default_train_path = _MANIFEST
    default_test_path = _MANIFEST
