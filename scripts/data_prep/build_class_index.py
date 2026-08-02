"""Build the canonical ImageNet-1k class index: wnid -> 0..999.

Every other script in this pipeline (make_shift_manifests.py,
verify_shift_manifests.py, zero_shot_sanity.py) resolves labels through this
one file instead of re-deriving them from a per-dataset directory listing.
That per-dataset shortcut is what scripts/make_imagenet_c_manifest.py does
today (`enumerate(sorted(os.listdir(data_dir)))`) — safe only when the
directory holds all 1000 classes, and silently wrong for a 200-class subset
(ImageNet-A/R/ES), which would get relabeled 0..199 instead of keeping its
true ImageNet-1k indices.

Source of truth: the wnid subdirectories already present under this
pipeline's `imagenet/val` (1000 classes, one dir each) — the same directory
`vision/ssl_engine.py`'s released head and every existing manifest already
implicitly assume matches the standard torchvision/ILSVRC-2012 ordering
(sorted wnid order). Two known anchors are asserted as a regression guard:
n01440764 (tench) -> 0, n01443537 (goldfish) -> 1.

Usage:
    python scripts/data_prep/build_class_index.py \
        --imagenet_val_dir /mnt5/noy/datasets/imagenet/val \
        --out /mnt5/noy/datasets/manifests/_meta/imagenet_class_index.json
"""
import argparse
import json
import os

_ANCHORS = {"n01440764": 0, "n01443537": 1}


def parse_args():
    p = argparse.ArgumentParser()
    root = os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")
    p.add_argument("--imagenet_val_dir",
                    default=os.path.join(root, "imagenet", "val"))
    p.add_argument("--out",
                    default=os.path.join(root, "manifests", "_meta",
                                          "imagenet_class_index.json"))
    return p.parse_args()


def build_class_index(imagenet_val_dir: str) -> dict:
    wnids = sorted(d for d in os.listdir(imagenet_val_dir)
                    if os.path.isdir(os.path.join(imagenet_val_dir, d)))
    if len(wnids) != 1000:
        raise ValueError(
            f"expected 1000 class dirs under {imagenet_val_dir}, found "
            f"{len(wnids)} — refusing to build a canonical index off an "
            f"incomplete ImageNet-1k val copy")
    wnid_to_idx = {w: i for i, w in enumerate(wnids)}
    for wnid, expected in _ANCHORS.items():
        if wnid_to_idx.get(wnid) != expected:
            raise AssertionError(
                f"anchor check failed: {wnid} -> {wnid_to_idx.get(wnid)}, "
                f"expected {expected} — sorted-wnid order no longer matches "
                f"the standard ILSVRC-2012 class index; do not proceed")
    return wnid_to_idx


def main(args):
    wnid_to_idx = build_class_index(args.imagenet_val_dir)
    idx_to_wnid = [None] * 1000
    for w, i in wnid_to_idx.items():
        idx_to_wnid[i] = w
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"wnid_to_idx": wnid_to_idx, "idx_to_wnid": idx_to_wnid},
                   f, indent=1)
    print(f"wrote canonical class index (1000 wnids) -> {args.out}")


if __name__ == "__main__":
    main(parse_args())
