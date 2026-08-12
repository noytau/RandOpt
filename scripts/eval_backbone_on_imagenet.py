"""Catastrophic-forgetting check: does adapting the DINOv2 backbone to
exdark — via RandOpt (perturb-and-select) or via full gradient FT — damage
its ORIGINAL ImageNet-1k classification ability?

Protocol: take the exact backbone transformation from an exdark experiment
and apply it to a FRESH engine carrying Meta's original, UNTOUCHED pretrained
ImageNet-1k head (not the exdark head), then evaluate on every named
--dataset target, resolved through randopt_shift.py's own _DATASET_REGISTRY
(manifest, split, input_mode, logit mask) -- the SAME registry every other
ImageNet-family comparison in this repo uses, so numbers here are directly
comparable to base/FT/RandOpt runs elsewhere. --dataset defaults to "all6" =
every registry entry (imagenet, imagenet_c, imagenet_a, imagenet_r,
imagenet_sketch, imagenet_es) -- NOTE this differs from randopt_shift.py's
own "all" shorthand, which means only the 4 shift datasets (excludes plain
imagenet/imagenet_c); this script's default is deliberately broader.

  --mode base    : sanity-check reference point (unmodified backbone +
                   original head) -- should match the project's known DINOv2
                   clean-ImageNet baseline.
  --mode randopt : replay each top-K (seed, sigma) pair from an exdark
                   results.json against the original head, majority-vote
                   ensemble exactly as the original run did -- evaluated on
                   every named target in the SAME perturb/restore pass (only
                   an extra forward pass per target, not extra perturb cost).
  --mode ft      : swap in the fine-tuned backbone saved by
                   scripts/ft_selfcontained.py (--backbone_out).

perturb_target="backbone" (vision/ssl_engine.py) perturbs ONLY the backbone:
noise is generated per-parameter from (seed, that parameter's own shape),
identical regardless of what head is attached, so replaying a (seed, sigma)
pair here reproduces the EXACT SAME backbone each RandOpt ensemble member
had -- only the head reading its features differs.

Usage:
    python scripts/eval_backbone_on_imagenet.py --mode base
    python scripts/eval_backbone_on_imagenet.py --mode randopt \\
        --results_json results/randopt-ssl-self-exdark-N3000-holdout/results.json
    python scripts/eval_backbone_on_imagenet.py --mode ft \\
        --ft_backbone_path checkpoints/exdark_ft_all_backbone.pth
"""
import argparse
import json
import os
import sys

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class, score  # noqa: E402
from scripts.randopt_shift import (  # noqa: E402
    _DATASET_REGISTRY, load_logit_mask, run_ensemble_multi)

_ALL6 = list(_DATASET_REGISTRY)  # imagenet, imagenet_c, imagenet_a, imagenet_r, imagenet_sketch, imagenet_es


def _parse_dataset_list(raw: str):
    names = []
    for token in (t.strip() for t in raw.split(",")):
        if token == "all6":
            names.extend(_ALL6)
        elif token in _DATASET_REGISTRY:
            names.append(token)
        else:
            raise ValueError(f"unknown --dataset entry '{token}' -- valid: "
                              f"{_ALL6}, or 'all6'")
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["base", "randopt", "ft"])
    p.add_argument("--dataset", default="all6",
                    help="comma-separated eval targets, any of: " +
                         ", ".join(_ALL6) + ", or 'all6' (default, every "
                         "registry entry -- broader than randopt_shift.py's "
                         "own 'all' shorthand)")
    p.add_argument("--results_json", default=None,
                    help="[randopt mode] results.json with top_k_perturbs")
    p.add_argument("--ft_backbone_path", default=None,
                    help="[ft mode] saved fine-tuned backbone state_dict, "
                         "from scripts/ft_selfcontained.py's --backbone_out")
    p.add_argument("--backbone_family", default="dinov2", choices=["dinov2", "dinov3"])
    p.add_argument("--test_samples", type=int, default=1000,
                    help="class-balanced sample per target; 0 = full split")
    p.add_argument("--num_engines", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    args = p.parse_args()
    args.dataset_targets = _parse_dataset_list(args.dataset)
    if args.mode == "randopt" and not args.results_json:
        p.error("--mode randopt requires --results_json")
    if args.mode == "ft" and not args.ft_backbone_path:
        p.error("--mode ft requires --ft_backbone_path")
    return args


def _datasets_root():
    return os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")


def _load_targets(dataset_targets, test_samples, rng):
    """One entry per target, resolved through the SAME registry every other
    ImageNet-family comparison in this repo uses -- manifest, split,
    input_mode, logit mask all identical to what a randopt_shift.py run on
    these targets would see."""
    from data_handlers import get_dataset_handler
    targets = {}
    for name in dataset_targets:
        reg = _DATASET_REGISTRY[name]
        h = get_dataset_handler(name)
        manifest = reg["manifest"] or h.default_train_path
        mask = load_logit_mask(name)
        items = sample_per_class(h.load_data(manifest, split=reg["split"]),
                                 test_samples, rng)
        targets[name] = {"handler": h, "mask": mask, "items": items,
                         "input_mode": reg["input_mode"]}
        print(f"eval target ({name}, split={reg['split']}): {manifest} | "
              f"logit_mask: {'None (full 1000-way)' if mask is None else f'{int((mask == 0).sum())}-class subset'} "
              f"| n={len(items)}")
    return targets


def main(args):
    rng = np.random.default_rng(args.global_seed)
    targets = _load_targets(args.dataset_targets, args.test_samples, rng)

    tag = f"imagenet-forget-check-{args.mode}"
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project,
                               name=args.wandb_name or tag, config=vars(args))

    ray_tmp = os.path.join(_datasets_root(), "ray_tmp")
    os.makedirs(ray_tmp, exist_ok=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=ray_tmp)
    from vision import launch_ssl_engines

    if args.mode == "base":
        engines = launch_ssl_engines(1, backbone_family=args.backbone_family,
                                     input_mode="official_resize")
        for name, info in targets.items():
            acc = score(info["handler"], ray.get(engines[0].predict.remote(
                info["items"], info["input_mode"], info["mask"])), info["items"])
            print(f"[{name}] BASE (unmodified backbone + original ImageNet head): {acc:.4f}")
            if wandb_run:
                wandb_run.log({f"imagenet_forget/{name}/base_accuracy": acc})

    elif args.mode == "randopt":
        with open(args.results_json) as f:
            results = json.load(f)
        top_k_perturbs = [tuple(pr) for pr in results["top_k_perturbs"]]
        k = len(top_k_perturbs)
        print(f"replaying {k} RandOpt (seed, sigma) pairs (backbone-only) "
              f"against the original ImageNet-1k head, on {len(targets)} targets")

        engines = launch_ssl_engines(args.num_engines,
                                     backbone_family=args.backbone_family,
                                     perturb_target="backbone",
                                     input_mode="official_resize")

        base_by_name = {}
        for name, info in targets.items():
            base_by_name[name] = score(info["handler"], ray.get(engines[0].predict.remote(
                info["items"], info["input_mode"], info["mask"])), info["items"])
            print(f"[{name}] BASE (unmodified): {base_by_name[name]:.4f}")

        class _Args:
            pass
        a = _Args()
        a.num_engines = len(engines)
        a.max_top_k = k
        a.top_k_list = [k]

        eval_sets = [{"name": name, "items": info["items"], "logit_mask": info["mask"],
                      "input_mode": info["input_mode"], "base": base_by_name[name]}
                     for name, info in targets.items()]
        ensemble = run_ensemble_multi(a, engines, top_k_perturbs, eval_sets, wandb_run)
        for name in targets:
            acc = ensemble[name][k]["accuracy"]
            gain = acc - base_by_name[name] * 100
            print(f"[{name}] RandOpt-perturbed-backbone ensemble (K={k}): "
                  f"{acc:.2f}% [{gain:+.2f}pp vs base]")
            if wandb_run:
                wandb_run.log({f"imagenet_forget/{name}/randopt_ensemble_accuracy": acc,
                               f"imagenet_forget/{name}/randopt_gain_over_base": gain})

    else:  # ft
        engines = launch_ssl_engines(1, backbone_family=args.backbone_family,
                                     input_mode="official_resize")
        base_by_name = {}
        for name, info in targets.items():
            base_by_name[name] = score(info["handler"], ray.get(engines[0].predict.remote(
                info["items"], info["input_mode"], info["mask"])), info["items"])
        ray.get(engines[0].load_backbone_state_dict.remote(args.ft_backbone_path))
        for name, info in targets.items():
            ft_acc = score(info["handler"], ray.get(engines[0].predict.remote(
                info["items"], info["input_mode"], info["mask"])), info["items"])
            gain = (ft_acc - base_by_name[name]) * 100
            print(f"[{name}] BASE: {base_by_name[name]:.4f} | FT-backbone + "
                  f"original ImageNet head: {ft_acc:.4f} [{gain:+.2f}pp vs base]")
            if wandb_run:
                wandb_run.log({f"imagenet_forget/{name}/base_accuracy": base_by_name[name],
                               f"imagenet_forget/{name}/ft_backbone_accuracy": ft_acc,
                               f"imagenet_forget/{name}/ft_gain_over_base": gain})

    if wandb_run:
        wandb_run.finish()
    ray.shutdown()


if __name__ == "__main__":
    main(parse_args())
