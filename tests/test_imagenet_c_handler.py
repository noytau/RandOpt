"""Tests for the ImageNet-C data loader: manifest generator
(scripts/make_imagenet_c_manifest.py) -> handler (data_handlers/imagenet_c.py)
-> decode helpers, as one pipeline — the way they're used in runs.

Builds a self-contained fake dataset (3 classes x 6 images, 224x224) in a
tempdir, generates a real manifest over it (splits 3/1/2 per class), then
checks split filtering, item schema, slicing, decode, determinism, and the
registry.

Run:  python -m pytest tests/test_imagenet_c_handler.py -v
  or: python tests/test_imagenet_c_handler.py
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WNIDS = ["n01440764", "n01443537", "n01484850"]          # sorted -> labels 0,1,2
PER_CLASS = {"train": 3, "val": 1, "test": 2}


def _build_fixture():
    """Fake wnid tree + real generated manifest; returns (root, manifest_path)."""
    from PIL import Image
    root = tempfile.mkdtemp(prefix="inc_loader_test_")
    data_dir = os.path.join(root, "gaussian_noise", "3")
    for w in WNIDS:
        os.makedirs(os.path.join(data_dir, w))
        for i in range(6):                               # 6 imgs = 3+1+2
            Image.new("RGB", (224, 224), color=(i * 40, 0, 0)).save(
                os.path.join(data_dir, w, f"ILSVRC2012_val_{i:08d}.JPEG"))
    manifest = os.path.join(root, "data.json")
    subprocess.run(
        [sys.executable, os.path.join(REPO, "scripts/make_imagenet_c_manifest.py"),
         "--data_dir", data_dir, "--out", manifest,
         "--train_per_class", "3", "--val_per_class", "1",
         "--test_per_class", "2"],
        check=True, capture_output=True)
    return root, manifest


ROOT, MANIFEST = _build_fixture()


def _handler():
    from data_handlers import get_dataset_handler
    return get_dataset_handler("imagenet_c")


# -- split retrieval -------------------------------------------------------

def test_split_sizes():
    h = _handler()
    for split, n in PER_CLASS.items():
        assert len(h.load_data(MANIFEST, split=split)) == n * len(WNIDS)


def test_splits_are_disjoint():
    h = _handler()
    seen = [d["image_path"] for s in PER_CLASS for d in h.load_data(MANIFEST, split=s)]
    assert len(seen) == len(set(seen)) == 6 * len(WNIDS)


def test_unknown_split_returns_empty():
    assert _handler().load_data(MANIFEST, split="nope") == []


# -- item schema -----------------------------------------------------------

def test_item_schema_and_label_consistency():
    for d in _handler().load_data(MANIFEST, split="train"):
        assert sorted(d) == ["class_id", "ground_truth", "image_path", "messages"]
        assert d["ground_truth"] == str(d["class_id"])   # reward compares strings
        assert os.path.isfile(d["image_path"])
        assert d["messages"] == []


def test_labels_follow_sorted_wnid_order():
    items = _handler().load_data(MANIFEST, split="train")
    for d in items:
        wnid = os.path.basename(os.path.dirname(d["image_path"]))
        assert d["class_id"] == WNIDS.index(wnid)


# -- slicing ---------------------------------------------------------------

def test_max_samples_and_start_index():
    h = _handler()
    full = h.load_data(MANIFEST, split="train")
    assert len(h.load_data(MANIFEST, split="train", max_samples=4)) == 4
    tail = h.load_data(MANIFEST, split="train", start_index=2)
    assert tail == full[2:]


# -- decode helpers --------------------------------------------------------

def test_load_image_batch_shape_and_range():
    from data_handlers.imagenet_c import load_image_batch
    b = load_image_batch(_handler().load_data(MANIFEST, split="test")[:4])
    assert tuple(b.shape) == (4, 3, 224, 224)
    assert 0.0 <= float(b.min()) and float(b.max()) <= 1.0


def test_decode_applies_transform():
    from torchvision import transforms
    from data_handlers.imagenet_c import load_image
    t = transforms.Compose([transforms.Resize((112, 112)), transforms.ToTensor()])
    x = load_image(_handler().load_data(MANIFEST, split="test")[0], t)
    assert tuple(x.shape) == (3, 112, 112)


# -- determinism -----------------------------------------------------------

def test_reload_is_deterministic():
    h = _handler()
    assert h.load_data(MANIFEST, split="test") == h.load_data(MANIFEST, split="test")


def test_missing_manifest_raises():
    try:
        _handler().load_data(os.path.join(ROOT, "absent.json"))
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
