"""ExDark dataset handler (dark/low-light images, own 12-class label space).

Same manifest schema and loader as ImageNet-C (data_handlers/imagenet_c.py) —
{"image", "label", "split"} — only the manifest location differs. Unlike
imagenet_a/r/sketch/es, `label` is NOT an ImageNet-1k index: it's this
dataset's own LOCAL 0..11 class index (see
scripts/data_prep/make_new_dataset_manifests.py / manifests/_meta/
exdark_class_index.json). There is no shared frozen head to mask against —
scripts/randopt_selfcontained.py fits + perturbs a dedicated 12-way head
instead (see vision/head_probe.py).
"""
import os

from .imagenet_c import ImageNetCHandler

_DATASETS_ROOT = os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")
_MANIFEST = os.path.join(_DATASETS_ROOT, "manifests", "exdark.json")


class ExDarkHandler(ImageNetCHandler):
    name = "exdark"
    default_train_path = _MANIFEST
    default_test_path = _MANIFEST
