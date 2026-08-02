"""ImageNet-Sketch dataset handler.

Same manifest schema and loader as ImageNet-C (data_handlers/imagenet_c.py) —
{"image", "label", "wnid", "split"} — only the default manifest location
differs. Full 1000-class label space (no logit mask needed at eval time,
unlike ImageNet-A/R/ES).
"""
import os

from .imagenet_c import ImageNetCHandler

_DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")
_MANIFEST = os.path.join(_DATASETS_ROOT, "manifests", "imagenet_sketch.json")


class ImageNetSketchHandler(ImageNetCHandler):
    name = "imagenet_sketch"
    default_train_path = _MANIFEST
    default_test_path = _MANIFEST
