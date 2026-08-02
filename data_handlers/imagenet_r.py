"""ImageNet-R dataset handler.

Same manifest schema and loader as ImageNet-C (data_handlers/imagenet_c.py) —
{"image", "label", "wnid", "split"} — only the default manifest location
differs. `label` is the canonical ImageNet-1k index (0..999). The published
L2P/DualPrompt train/test split (Wang et al., 2022) is used for `split`, but
only for its file-path list — its own `targets` field encodes local 0..199
continual-learning indices, not canonical ImageNet-1k indices, and is
discarded (see scripts/data_prep/make_shift_manifests.py).
"""
import os

from .imagenet_c import ImageNetCHandler

_DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")
_MANIFEST = os.path.join(_DATASETS_ROOT, "manifests", "imagenet_r.json")


class ImageNetRHandler(ImageNetCHandler):
    name = "imagenet_r"
    default_train_path = _MANIFEST
    default_test_path = _MANIFEST
