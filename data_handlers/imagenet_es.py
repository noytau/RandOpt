"""ImageNet-ES dataset handler.

Same manifest schema and loader as ImageNet-C (data_handlers/imagenet_c.py) —
{"image", "label", "wnid", "split"} — only the default manifest location
differs. `label` is the canonical ImageNet-1k index for the Tiny-ImageNet-200
wnids that also appear in ILSVRC-2012 (see scripts/data_prep/
make_shift_manifests.py for the intersection count/dropped-wnid log); manifest
entries additionally carry a non-null `condition` field (light/sensor
setting) and never include the `sampled_tin_no_resize*` reference-sample
directories.
"""
import os

from .imagenet_c import ImageNetCHandler

_DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")
_MANIFEST = os.path.join(_DATASETS_ROOT, "manifests", "imagenet_es.json")


class ImageNetESHandler(ImageNetCHandler):
    name = "imagenet_es"
    default_train_path = _MANIFEST
    default_test_path = _MANIFEST
