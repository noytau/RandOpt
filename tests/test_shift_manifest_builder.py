"""Tests for scripts/data_prep/make_shift_manifests.py's per-dataset builders.

Focus: the regression this whole pipeline exists to prevent — labels must
come from the canonical class index, never from a local sort of the wnids
present in a dataset's own directory. Every synthetic fixture below uses a
"scrambled" wnid_to_idx (canonical indices that do NOT match alphabetical
position) specifically so a bug that falls back to local sorting would be
caught by an assertion, not hidden by coincidence.

Network calls are avoided by pre-seeding the on-disk caches the builders
already check for (imagenet_r_wnids.txt, the split-yaml caches,
imagenet_a_indices_in_1k.json) — the same files a real run would leave behind
after its first network fetch.

Run:  python -m pytest tests/test_shift_manifest_builder.py -v
  or: python tests/test_shift_manifest_builder.py
"""
import json
import os
import sys
import tempfile
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  "scripts", "data_prep"))

import make_shift_manifests as msm  # noqa: E402

# canonical indices deliberately NOT equal to alphabetical position (0,1,2)
_SCRAMBLED = {"n00000001": 500, "n00000002": 5, "n00000003": 900}


def _tmp_paths():
    root = tempfile.mkdtemp(prefix="shift_manifest_test_")
    paths = {
        "raw": os.path.join(root, "raw"),
        "manifests": os.path.join(root, "manifests"),
        "meta": os.path.join(root, "manifests", "_meta"),
        "logs": os.path.join(root, "logs"),
    }
    for p in paths.values():
        os.makedirs(p, exist_ok=True)
    return paths


def _touch(*parts):
    """Write a file with content unique to its own path — NOT empty. The
    manifest builders now group files by content hash before splitting (to
    keep exact-duplicate images out of different splits), so same-content
    fixtures would all collapse into one hash group and break split-count
    assertions below; each fixture file must hash differently, like real
    (non-duplicate) photos do."""
    path = os.path.join(*parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(path)
    return path


# -- imagenet_sketch: fully local, no network dependency --------------------

def test_sketch_labels_use_canonical_index_not_local_sort():
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_sketch")
    for w in _SCRAMBLED:
        for i in range(10):
            _touch(raw, w, f"img_{i}.jpg")
    # pad to 1000 classes (build_imagenet_sketch asserts exactly 1000)
    wnid_to_idx = dict(_SCRAMBLED)
    for j in range(997):
        w = f"n9{j:07d}"
        wnid_to_idx[w] = 1000 + j  # out of [0,999] on purpose: never checked
        # for these padding classes since only the 3 real ones are asserted
        for i in range(10):
            _touch(raw, w, f"img_{i}.jpg")

    entries, subset_indices, wnids = msm.build_imagenet_sketch(paths, wnid_to_idx, seed=0)
    assert subset_indices is None  # full 1000-way: no logit mask
    assert len(wnids) == 1000
    by_wnid = {}
    for e in entries:
        by_wnid.setdefault(e["wnid"], []).append(e)
    for w, expected_label in _SCRAMBLED.items():
        labels = {e["label"] for e in by_wnid[w]}
        assert labels == {expected_label}, (
            f"{w}: got labels {labels}, expected only the canonical "
            f"{expected_label} — local-sort regression")


def test_sketch_splits_partition_each_class_without_overlap():
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_sketch")
    wnid_to_idx = {}
    for j in range(1000):
        w = f"n{j:08d}"
        wnid_to_idx[w] = j
        n = 2 if j == 0 else 10  # one deliberately-small class (<3 images)
        for i in range(n):
            _touch(raw, w, f"img_{i}.jpg")

    entries, _, _ = msm.build_imagenet_sketch(paths, wnid_to_idx, seed=0)
    small_class = "n00000000"
    small_entries = [e for e in entries if e["wnid"] == small_class]
    assert len(small_entries) == 2
    assert all(e["split"] == "test" for e in small_entries)  # spec S3.3/S6

    normal_class = "n00000001"
    normal = [e for e in entries if e["wnid"] == normal_class]
    paths_by_split = {s: {e["image"] for e in normal if e["split"] == s}
                       for s in ("train", "val", "test")}
    assert paths_by_split["train"] & paths_by_split["val"] == set()
    assert paths_by_split["train"] & paths_by_split["test"] == set()
    assert paths_by_split["val"] & paths_by_split["test"] == set()
    assert sum(len(v) for v in paths_by_split.values()) == 10


def test_duplicate_content_files_never_split_across_boundary():
    """Regression for the real bug this pipeline caught in ImageNet-A: two
    files with identical bytes (same photo, two filenames) must land in the
    SAME split, never leak across train/val/test."""
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_a")
    w = "n00000001"
    class_dir = os.path.join(raw, w)
    for i in range(8):
        _touch(class_dir, f"unique_{i}.jpg")
    # two duplicate pairs: identical content, different filenames
    dup_path_1 = os.path.join(class_dir, "dup_a.jpg")
    dup_path_2 = os.path.join(class_dir, "dup_b.jpg")
    os.makedirs(class_dir, exist_ok=True)
    with open(dup_path_1, "w") as f:
        f.write("identical-content")
    with open(dup_path_2, "w") as f:
        f.write("identical-content")

    with open(os.path.join(paths["meta"], "imagenet_a_indices_in_1k.json"), "w") as f:
        json.dump([7], f)
    wnid_to_idx = {w: 7}

    entries, _, _ = msm.build_imagenet_a(paths, wnid_to_idx, seed=1)
    split_by_fname = {os.path.basename(e["image"]): e["split"] for e in entries}
    assert split_by_fname["dup_a.jpg"] == split_by_fname["dup_b.jpg"], (
        "duplicate-content files leaked across splits")


# -- imagenet_a: network calls short-circuited via a pre-seeded cache file --

def test_imagenet_a_matching_indices_stratifies_and_uses_canonical_labels():
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_a")
    for w in _SCRAMBLED:
        for i in range(10):
            _touch(raw, w, f"img_{i}.jpg")
    with open(os.path.join(paths["meta"], "imagenet_a_indices_in_1k.json"), "w") as f:
        json.dump(sorted(_SCRAMBLED.values()), f)

    entries, subset_indices, wnids = msm.build_imagenet_a(paths, _SCRAMBLED, seed=0)
    assert sorted(subset_indices) == sorted(_SCRAMBLED.values())
    by_wnid = {}
    for e in entries:
        by_wnid.setdefault(e["wnid"], []).append(e)
    for w, expected_label in _SCRAMBLED.items():
        assert {e["label"] for e in by_wnid[w]} == {expected_label}
        counts = {}
        for e in by_wnid[w]:
            counts[e["split"]] = counts.get(e["split"], 0) + 1
        assert counts == {"train": 6, "val": 2, "test": 2}  # 60/20/20 of 10


def test_imagenet_a_tiny_class_still_gets_a_test_image():
    """A class with <5 images can't really be 60/20/20, but every class must
    still appear in test (spec S7#6, a hard requirement) — caught for real
    when round(n*0.2) rounded to 0 for a couple of n=3 ImageNet-A classes."""
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_a")
    for i in range(3):
        _touch(raw, "n00000001", f"img_{i}.jpg")
    with open(os.path.join(paths["meta"], "imagenet_a_indices_in_1k.json"), "w") as f:
        json.dump([7], f)
    entries, _, _ = msm.build_imagenet_a(paths, {"n00000001": 7}, seed=0)
    assert sum(1 for e in entries if e["split"] == "test") >= 1


def test_class_coverage_guaranteed_even_with_few_giant_duplicate_groups():
    """Regression for a real bug found on the actual ImageNet-Sketch data: a
    class with 50 files but only 4 distinct images (each duplicated ~12x)
    left `test` with ZERO images — plain ratio-greedy gave train 3 of the 4
    oversized groups before val/test ever got a look, since every group
    (~12-13) dwarfed val/test's target of 8. Every split with target > 0
    must get >= 1 group whenever there are enough distinct groups to cover
    every split at all."""
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_a")
    class_dir = os.path.join(raw, "n00000001")
    # 4 distinct contents, ~12-13 duplicate filenames each = 50 files, same
    # shape as the real failing class
    sizes = [13, 13, 12, 12]
    idx = 0
    for content_id, n in enumerate(sizes):
        content = f"content-{content_id}"
        for _ in range(n):
            path = os.path.join(class_dir, f"img_{idx}.jpg")
            os.makedirs(class_dir, exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            idx += 1

    with open(os.path.join(paths["meta"], "imagenet_a_indices_in_1k.json"), "w") as f:
        json.dump([7], f)
    entries, _, _ = msm.build_imagenet_a(paths, {"n00000001": 7}, seed=0)
    counts = Counter(e["split"] for e in entries)
    assert counts["test"] >= 1, f"test got zero images: {counts}"
    assert counts["val"] >= 1, f"val got zero images: {counts}"
    assert sum(counts.values()) == 50


def test_imagenet_a_index_mismatch_raises():
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_a")
    for w in _SCRAMBLED:
        _touch(raw, w, "img_0.jpg")
    # deliberately wrong reference set (missing one real index, has a fake one)
    wrong = sorted(_SCRAMBLED.values())[:-1] + [123456]
    with open(os.path.join(paths["meta"], "imagenet_a_indices_in_1k.json"), "w") as f:
        json.dump(wrong, f)
    try:
        msm.build_imagenet_a(paths, _SCRAMBLED, seed=0)
        assert False, "expected RuntimeError on index mismatch"
    except RuntimeError:
        pass


# -- imagenet_r: split membership from the yaml's file-path list only,      --
# -- its `targets` field (local 0..N-1 indices) must be discarded            --

def test_imagenet_r_uses_split_paths_not_local_targets():
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_r")
    wnids = sorted(_SCRAMBLED)
    for w in wnids:
        for i in range(4):
            _touch(raw, w, f"img_{i}.jpg")

    with open(os.path.join(paths["meta"], "imagenet_r_wnids.txt"), "w") as f:
        f.write("\n".join(wnids) + "\n")

    train_lines = ["data:"]
    target_lines = ["targets:"]
    for local_idx, w in enumerate(wnids):
        for i in range(3):  # 3/4 images per class -> train_full
            train_lines.append(f"- data/imagenet-r/{w}/img_{i}.jpg")
            target_lines.append(f"- {local_idx}")  # WRONG on purpose: 0,1,2
    with open(os.path.join(paths["meta"], "_imagenet_r_train_split.yaml.cache"), "w") as f:
        f.write("\n".join(train_lines + target_lines) + "\n")

    test_lines = ["data:"]
    test_targets = ["targets:"]
    for local_idx, w in enumerate(wnids):
        test_lines.append(f"- data/imagenet-r/{w}/img_3.jpg")  # 1/4 images -> test
        test_targets.append(f"- {local_idx}")
    with open(os.path.join(paths["meta"], "_imagenet_r_test_split.yaml.cache"), "w") as f:
        f.write("\n".join(test_lines + test_targets) + "\n")

    entries, subset_indices, ref_wnids = msm.build_imagenet_r(paths, _SCRAMBLED, seed=0)

    assert sorted(subset_indices) == sorted(_SCRAMBLED.values())
    by_wnid = {}
    for e in entries:
        by_wnid.setdefault(e["wnid"], []).append(e)
    for w, expected_label in _SCRAMBLED.items():
        labels = {e["label"] for e in by_wnid[w]}
        # canonical label only — never the 0/1/2 planted in the yaml's targets
        assert labels == {expected_label}, (
            f"{w}: got {labels}, expected only canonical {expected_label} "
            f"— the yaml's local `targets` field leaked into the manifest")
        counts = {}
        for e in by_wnid[w]:
            counts[e["split"]] = counts.get(e["split"], 0) + 1
        assert counts.get("test", 0) == 1  # 1 image per class went to the
        # published test split; the other 3 (train_full) are split train/val
        assert counts.get("train", 0) + counts.get("val", 0) == 3


def test_reconcile_duplicate_content_across_splits():
    """Regression for a second real bug: 28 duplicate-content images leaked
    across imagenet_r's PUBLISHED split (which isn't hash-aware at all), and
    319 CROSS-CLASS duplicates leaked in imagenet_sketch (per-class hash
    grouping can't see across classes). This final reconciliation pass must
    catch both regardless of which builder or class produced them."""
    entries = [
        {"image": "/a", "sha256": "H1", "split": "train"},
        {"image": "/b", "sha256": "H1", "split": "train"},
        {"image": "/c", "sha256": "H1", "split": "val"},  # minority -> reassigned
        {"image": "/d", "sha256": "H2", "split": "test"},  # untouched: no dup
        {"image": "/e", "sha256": "H3", "split": "train"},  # cross-class dup pair
        {"image": "/f", "sha256": "H3", "split": "test"},
    ]
    fixed = msm._reconcile_duplicate_content_across_splits(entries, "unit_test")
    by_hash = {}
    for e in fixed:
        by_hash.setdefault(e["sha256"], set()).add(e["split"])
    assert by_hash["H1"] == {"train"}  # majority split won
    assert by_hash["H2"] == {"test"}  # unaffected
    assert len(by_hash["H3"]) == 1  # tie resolved to a single split


def test_imagenet_r_wnid_mismatch_raises():
    paths = _tmp_paths()
    raw = os.path.join(paths["raw"], "imagenet_r")
    for w in _SCRAMBLED:
        _touch(raw, w, "img_0.jpg")
    # reference list omits one real wnid and adds a fake one
    wrong = sorted(_SCRAMBLED)[:-1] + ["n99999999"]
    with open(os.path.join(paths["meta"], "imagenet_r_wnids.txt"), "w") as f:
        f.write("\n".join(wrong) + "\n")
    try:
        msm.build_imagenet_r(paths, _SCRAMBLED, seed=0)
        assert False, "expected RuntimeError on wnid mismatch"
    except RuntimeError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed")
