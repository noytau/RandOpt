"""Clean-VAL scoring manifest: the out-of-sample, leakage-free scoring set.

Scoring perturbations on ImageNet TRAIN images measures preservation of the
head's memorization (it was trained on them; in-sample base ~92 vs ~87 val).
Scoring on arbitrary VAL images instead leaks: ImageNet-C is generated FROM
val, so clean val images may be the uncorrupted twins of IC test images.

This script threads the needle using the IC manifest's own split: it takes
the IC entries with split=train (never used for RandOpt testing), maps each
corrupted image back to its clean val original (ImageNet-C preserves the
ILSVRC2012_val_* filenames), subsamples per class, and writes a manifest of
CLEAN VAL images that are (a) out-of-sample for the head and (b) image-level
disjoint from the IC test split. Asserts both properties.

Usage:
    python scripts/make_val_scoring_manifest.py \
        --ic_manifest data/imagenet_c/data.json \
        --val_root /mnt5/noy/datasets/imagenet/val \
        --per_class 10 --out data/imagenet_val10/data.json
"""
import argparse
import json
import os
import random
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ic_manifest", default="data/imagenet_c/data.json")
    p.add_argument("--val_root", required=True,
                   help="clean val export: <val_root>/<wnid>/ILSVRC2012_val_*.JPEG")
    p.add_argument("--per_class", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="data/imagenet_val10/data.json")
    return p.parse_args()


def main(args):
    ic = json.load(open(args.ic_manifest))
    rng = random.Random(args.seed)

    ic_train = defaultdict(list)   # wnid -> [(fname, label)]
    test_fnames = set()            # image-level disjointness guard
    for e in ic:
        fname = os.path.basename(e["image"])
        if e["split"] == "train":
            ic_train[e["wnid"]].append((fname, e["label"]))
        elif e["split"] == "test":
            test_fnames.add(fname)

    entries, missing = [], []
    for wnid in sorted(ic_train):
        pool = sorted(ic_train[wnid])
        rng.shuffle(pool)
        picked = 0
        for fname, label in pool:
            if picked >= args.per_class:
                break
            path = os.path.join(args.val_root, wnid, fname)
            if not os.path.exists(path):
                missing.append(path)
                continue
            assert fname not in test_fnames, f"leak: {fname} is in IC test"
            entries.append({"image": path, "label": label, "wnid": wnid,
                            "split": "train"})
            picked += 1
        assert picked == args.per_class, \
            f"{wnid}: only {picked}/{args.per_class} val images found"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(entries, f, indent=1)
    print(f"wrote {len(entries)} clean-val entries "
          f"({len(ic_train)} classes, {args.per_class}/class) -> {args.out}")
    if missing:
        print(f"note: {len(missing)} IC-train images had no clean val twin "
              f"(skipped), e.g. {missing[:2]}")


if __name__ == "__main__":
    main(parse_args())
