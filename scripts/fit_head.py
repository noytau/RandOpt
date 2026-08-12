"""Fit + save a fresh N-way linear head for a self-contained dataset (own
train/val/test split, e.g. exdark) — a one-time prerequisite before
scripts/randopt_selfcontained.py can run (RandOpt needs a trained center to
perturb around; these datasets have no off-the-shelf pretrained head the way
ImageNet-1k does). Runs as a plain single-GPU script (no Ray), same style as
scripts/ft_matched_baseline.py.

Usage:
    python scripts/fit_head.py --dataset exdark
    python scripts/fit_head.py --dataset exdark --epochs 3 --train_samples 300
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class  # noqa: E402
from scripts.randopt_selfcontained import _SELFCONTAINED_REGISTRY  # noqa: E402
from vision.head_probe import fit_linear_head  # noqa: E402
from vision.ssl_engine import SSLEngineImpl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(_SELFCONTAINED_REGISTRY))
    p.add_argument("--backbone_family", default="dinov2", choices=["dinov2", "dinov3"])
    p.add_argument("--backbone_name", default=None)
    p.add_argument("--weights_path", default=None)
    p.add_argument("--head_dir", default="checkpoints")
    p.add_argument("--scope", default="head", choices=["head", "last_n_blocks", "all"],
                    help="'head' (default) is a true linear probe (backbone "
                         "frozen) — the cheapest, most standard center to "
                         "perturb around first")
    p.add_argument("--last_n_blocks", type=int, default=0)
    p.add_argument("--train_samples", type=int, default=0,
                    help="class-balanced sample from the dataset's own train "
                         "split; 0 = full split")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    return p.parse_args()


def main(args):
    from data_handlers import get_dataset_handler
    reg = _SELFCONTAINED_REGISTRY[args.dataset]
    handler = get_dataset_handler(args.dataset)
    manifest = reg["manifest"] or handler.default_train_path

    rng = np.random.default_rng(args.global_seed)
    train_items = sample_per_class(handler.load_data(manifest, split="train"),
                                   args.train_samples, rng)
    print(f"[{args.dataset}] probe train set: {len(train_items)} "
          f"({reg['num_classes']} classes)")

    engine = SSLEngineImpl(backbone_family=args.backbone_family,
                           backbone_name=args.backbone_name,
                           weights_path=args.weights_path,
                           num_classes=reg["num_classes"],
                           input_mode=reg["input_mode"])
    acc = fit_linear_head(engine, train_items, handler, reg["input_mode"],
                          lr=args.lr, epochs=args.epochs,
                          batch_size=args.batch_size, scope=args.scope,
                          last_n_blocks=args.last_n_blocks,
                          seed=args.global_seed)
    print(f"[{args.dataset}] final probe train accuracy: {acc:.4f}")

    os.makedirs(args.head_dir, exist_ok=True)
    head_path = os.path.join(args.head_dir, f"{args.dataset}_head.pth")
    torch.save(engine.head.state_dict(), head_path)
    print(f"saved: {head_path}")


if __name__ == "__main__":
    main(parse_args())
