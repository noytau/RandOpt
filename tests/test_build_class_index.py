"""Tests for scripts/data_prep/build_class_index.py — the canonical
wnid -> 0..999 index every other shift-dataset script resolves labels
through. Uses a synthetic 1000-dir tree (empty dirs; build_class_index only
lists directory names, never opens images) instead of a real ImageNet copy.

Run:  python -m pytest tests/test_build_class_index.py -v
  or: python tests/test_build_class_index.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "scripts", "data_prep"))

from build_class_index import build_class_index  # noqa: E402

_ANCHOR_A, _ANCHOR_B = "n01440764", "n01443537"


def _make_val_dir(n=1000, drop_anchor=False):
    root = tempfile.mkdtemp(prefix="class_index_test_")
    names = [] if drop_anchor else [_ANCHOR_A, _ANCHOR_B]
    names += [f"n02{i:06d}" for i in range(n - len(names))]
    for w in names:
        os.makedirs(os.path.join(root, w))
    return root


def test_anchors_map_to_0_and_1():
    root = _make_val_dir()
    idx = build_class_index(root)
    assert idx[_ANCHOR_A] == 0
    assert idx[_ANCHOR_B] == 1
    assert len(idx) == 1000


def test_sorted_order_is_contiguous_0_999():
    root = _make_val_dir()
    idx = build_class_index(root)
    assert sorted(idx.values()) == list(range(1000))


def test_wrong_class_count_raises():
    root = _make_val_dir(n=999)
    try:
        build_class_index(root)
        assert False, "expected ValueError for incomplete class dirs"
    except ValueError:
        pass


def test_missing_anchor_raises():
    root = _make_val_dir(drop_anchor=True)
    try:
        build_class_index(root)
        assert False, "expected AssertionError when anchors are missing"
    except AssertionError:
        pass


def test_ignores_non_directory_entries():
    root = _make_val_dir()
    with open(os.path.join(root, "not_a_class.txt"), "w") as f:
        f.write("stray file")
    idx = build_class_index(root)
    assert len(idx) == 1000


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
