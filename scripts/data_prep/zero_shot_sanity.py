"""Zero-shot sanity baseline (spec S7 check 10 — "the single most valuable
diagnostic in this document"). Loads the same frozen DINOv2 backbone +
released ImageNet-1k head already used by vision/ssl_engine.py, applies each
dataset's logit mask, evaluates the `test` split, and checks the expected
difficulty ordering: ImageNet-ES > ImageNet-A > ImageNet-R > ImageNet-Sketch.

A badly-violated ordering (especially ImageNet-Sketch NOT being hardest)
means "suspect a label-mapping error in S4" per the spec — logged as a loud
warning here rather than a hard failure, since S1's architecture-bias caveat
(ImageNet-A's ResNet-50 filter) means the ordering isn't a strict invariant.

Usage:
    python scripts/data_prep/zero_shot_sanity.py --dataset all
    python scripts/data_prep/zero_shot_sanity.py --dataset imagenet_a --max_samples 500 \
        --wandb_project randopt
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch

from data_handlers import get_dataset_handler
from utils.logit_mask import build_mask
from vision.ssl_engine import SSLEngineImpl

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

_EXPECTED_ORDER = ["imagenet_es", "imagenet_a", "imagenet_r", "imagenet_sketch"]


def _root():
    return os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")


def evaluate_one(engine, name, max_samples=None, batch_size=32):
    handler = get_dataset_handler(name)
    manifest_path = os.path.join(_root(), "manifests", f"{name}.json")
    meta_path = os.path.join(_root(), "manifests", "_meta", f"{name}_meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    subset_indices = meta.get("subset_indices")
    mask = build_mask(subset_indices) if subset_indices is not None else None

    items = handler.load_data(manifest_path, split="test", max_samples=max_samples)
    if not items:
        raise RuntimeError(f"{name}: empty test split at {manifest_path}")

    correct = 0
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        preds = engine.predict(batch, input_mode="official_resize", logit_mask=mask)
        for pred, item in zip(preds, batch):
            correct += int(pred == item["ground_truth"])
    top1 = correct / len(items)
    print(f"{name}: top-1 = {top1:.4%} ({correct}/{len(items)})")
    return top1, len(items)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                    choices=_EXPECTED_ORDER + ["all"])
    p.add_argument("--max_samples", type=int, default=None,
                    help="cap test-split size per dataset for speed; default "
                         "uses the full test split")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--wandb_project", type=str, default=None,
                    help="W&B project name. If not set, wandb logging is disabled.")
    p.add_argument("--wandb_run_name", type=str, default=None)
    args = p.parse_args()

    if WANDB_AVAILABLE and args.wandb_project:
        run_name = (args.wandb_run_name
                    or f"shift-sanity-{args.dataset}-n{args.max_samples or 'full'}")
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    targets = _EXPECTED_ORDER if args.dataset == "all" else [args.dataset]
    engine = SSLEngineImpl(input_mode="official_resize")

    results = {}
    for name in targets:
        top1, n = evaluate_one(engine, name, args.max_samples, args.batch_size)
        results[name] = top1
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({f"sanity/{name}/top1": top1, f"sanity/{name}/n_samples": n})

    print("\n=== zero-shot sanity summary ===")
    for name in _EXPECTED_ORDER:
        if name in results:
            print(f"  {name:16s} {results[name]:.4%}")

    if args.dataset == "all":
        ordered = [n for n in _EXPECTED_ORDER if n in results]
        accs = [results[n] for n in ordered]
        ordering_holds = accs == sorted(accs, reverse=True)
        if not ordering_holds:
            print("\nWARNING: expected difficulty ordering "
                  "(ES > A > R > Sketch) is violated — per spec S7#10, "
                  "suspect a label-mapping error in the manifests before "
                  "treating this as a genuine finding.")
        else:
            print("\nExpected difficulty ordering holds (ES > A > R > Sketch).")
        if WANDB_AVAILABLE and wandb.run:
            wandb.log({"sanity/ordering_holds": ordering_holds})

    if WANDB_AVAILABLE and wandb.run:
        wandb.finish()


if __name__ == "__main__":
    main()
