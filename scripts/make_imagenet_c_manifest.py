"""Write the ImageNet-C manifest JSON: links to image paths + labels.

Walks <data_dir>/<wnid>/*.JPEG (one corruption/severity, e.g.
/mnt5/noy/datasets/imagenet_c/gaussian_noise/3), labels classes 0..999 by
sorted-wnid order (the standard ImageNet class index), and carves each class's
images into disjoint train/val/test splits with a seeded shuffle. The split
lives IN the manifest, so it is frozen on disk and every consumer sees the
same one.

Output entries:
    {"image": "/abs/path/....JPEG", "label": 17, "wnid": "n01440764",
     "split": "train"}

Usage:
    python scripts/make_imagenet_c_manifest.py \
        --data_dir /mnt5/noy/datasets/imagenet_c/gaussian_noise/3 \
        --out data/imagenet_c/data.json
"""
import argparse
import json
import os
import random


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True,
                   help="one corruption/severity dir holding <wnid>/*.JPEG")
    p.add_argument("--out", default="data/imagenet_c/data.json")
    p.add_argument("--train_per_class", type=int, default=25)
    p.add_argument("--val_per_class", type=int, default=10)
    p.add_argument("--test_per_class", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main(args):
    rng = random.Random(args.seed)
    wnids = sorted(d for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d)))
    entries = []
    for label, wnid in enumerate(wnids):
        files = sorted(f for f in os.listdir(os.path.join(args.data_dir, wnid))
                       if f.lower().endswith((".jpeg", ".jpg", ".png")))
        rng.shuffle(files)
        n_tr, n_va = args.train_per_class, args.val_per_class
        n_te = args.test_per_class
        for i, fname in enumerate(files[:n_tr + n_va + n_te]):
            split = ("train" if i < n_tr
                     else "val" if i < n_tr + n_va else "test")
            entries.append({
                "image": os.path.join(args.data_dir, wnid, fname),
                "label": label,
                "wnid": wnid,
                "split": split,
            })
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(entries, f, indent=1)
    counts = {s: sum(1 for e in entries if e["split"] == s)
              for s in ("train", "val", "test")}
    print(f"wrote {len(entries)} entries ({len(wnids)} classes) -> {args.out}")
    print(f"splits: {counts}")


if __name__ == "__main__":
    main(parse_args())
