"""ImageNet-A dataset handler.

Same manifest schema and loader as ImageNet-C (data_handlers/imagenet_c.py) —
{"image", "label", "wnid", "split"} — only the default manifest location
differs. `label` is the canonical ImageNet-1k index (0..999), resolved via
scripts/data_prep/build_class_index.py, NOT a local 0..199 re-sort of this
dataset's 200 classes; masking to the 200-class subset happens at evaluation
time (utils/logit_mask.py), never by remapping labels here.
"""
import os

from .imagenet_c import ImageNetCHandler

_DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")
_MANIFEST = os.path.join(_DATASETS_ROOT, "manifests", "imagenet_a.json")


class ImageNetAHandler(ImageNetCHandler):
    name = "imagenet_a"
    default_train_path = _MANIFEST
    default_test_path = _MANIFEST
