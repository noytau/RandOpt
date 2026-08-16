"""Reverse-direction forgetting check: does adapting the DINOv3 backbone to
clean ImageNet -- via RandOpt (perturb-and-select) or via full gradient FT --
transfer to (or damage) its ability on ExDark, a domain it was never adapted
toward?

Mirrors scripts/eval_backbone_on_imagenet.py's role exactly, but in the
opposite direction: that script takes an ExDark-adapted backbone and checks
it against the ORIGINAL frozen ImageNet-1k head; this script takes an
ImageNet-adapted backbone and checks it against a freshly-fit ExDark head.
ExDark has no shared registry-based head/logit-mask the way the ImageNet
family does (it's a private 12-class label space, see
scripts/randopt_selfcontained.py's _SELFCONTAINED_REGISTRY) -- so a real
(non-toy) ExDark head must be fit on the UNADAPTED base backbone first
(scripts/fit_head.py --dataset exdark), then reused unchanged across base/
randopt/ft here, the same way the original head is reused unchanged in the
other direction.

perturb_target="backbone" (vision/ssl_engine.py) perturbs ONLY the backbone:
noise is generated per-parameter from (seed, that parameter's own shape),
identical regardless of what head is attached, so replaying a (seed, sigma)
pair here reproduces the EXACT SAME backbone each ImageNet-trained RandOpt
ensemble member had -- only the head reading its features differs.

Usage:
    python scripts/eval_backbone_on_exdark.py --mode base \\
        --head_path checkpoints/exdark_dinov3_head.pth
    python scripts/eval_backbone_on_exdark.py --mode randopt \\
        --head_path checkpoints/exdark_dinov3_head.pth \\
        --results_json results/dinov3-all6-N2000-K50-tr1k-te1k/results.json
    python scripts/eval_backbone_on_exdark.py --mode ft \\
        --head_path checkpoints/exdark_dinov3_head.pth \\
        --ft_backbone_path checkpoints/dinov3_ft_all_imagenet_backbone.pth
"""
import argparse
import json
import os
import sys

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class, score  # noqa: E402
from scripts.randopt_selfcontained import _SELFCONTAINED_REGISTRY  # noqa: E402
from scripts.randopt_shift import run_ensemble_multi  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["base", "randopt", "ft"])
    p.add_argument("--dataset", default="exdark", choices=list(_SELFCONTAINED_REGISTRY),
                    help="which self-contained dataset's own test split to "
                         "evaluate on -- exdark is the only one wired up so far")
    p.add_argument("--head_path", required=True,
                    help="a real (non-toy) head fit on the UNADAPTED base "
                         "backbone via scripts/fit_head.py --dataset "
                         "<dataset> -- reused unchanged across all 3 modes")
    p.add_argument("--results_json", default=None,
                    help="[randopt mode] results.json with top_k_perturbs, "
                         "from a RandOpt run trained/scored on clean "
                         "ImageNet (e.g. scripts/randopt_shift.py's output)")
    p.add_argument("--ft_backbone_path", default=None,
                    help="[ft mode] saved fine-tuned backbone state_dict, "
                         "from scripts/ft_matched_baseline.py's --backbone_out")
    p.add_argument("--backbone_family", default="dinov3", choices=["dinov2", "dinov3"])
    p.add_argument("--test_samples", type=int, default=0,
                    help="class-balanced sample from the dataset's own test "
                         "split; 0 = full split")
    p.add_argument("--num_engines", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    args = p.parse_args()
    if args.mode == "randopt" and not args.results_json:
        p.error("--mode randopt requires --results_json")
    if args.mode == "ft" and not args.ft_backbone_path:
        p.error("--mode ft requires --ft_backbone_path")
    return args


def _datasets_root():
    return os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")


def main(args):
    reg = _SELFCONTAINED_REGISTRY[args.dataset]
    rng = np.random.default_rng(args.global_seed)

    from data_handlers import get_dataset_handler
    handler = get_dataset_handler(args.dataset)
    manifest = reg["manifest"] or handler.default_train_path
    test_items = sample_per_class(handler.load_data(manifest, split="test"),
                                  args.test_samples, rng)
    print(f"[{args.dataset}] eval target (own test): {manifest} | "
          f"n={len(test_items)} | head: {args.head_path}")

    tag = f"{args.dataset}-forget-check-{args.mode}"
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
                                     num_classes=reg["num_classes"],
                                     head_path=args.head_path,
                                     input_mode=reg["input_mode"])
        acc = score(handler, ray.get(engines[0].predict.remote(
            test_items, reg["input_mode"], None)), test_items)
        print(f"[{args.dataset}] BASE (unmodified backbone + {args.dataset} head): {acc:.4f}")
        if wandb_run:
            wandb_run.log({f"{args.dataset}_forget/base_accuracy": acc})

    elif args.mode == "randopt":
        with open(args.results_json) as f:
            results = json.load(f)
        top_k_perturbs = [tuple(pr) for pr in results["top_k_perturbs"]]
        k = len(top_k_perturbs)
        print(f"replaying {k} ImageNet-trained RandOpt (seed, sigma) pairs "
              f"(backbone-only) against the {args.dataset} head")

        engines = launch_ssl_engines(args.num_engines,
                                     backbone_family=args.backbone_family,
                                     perturb_target="backbone",
                                     num_classes=reg["num_classes"],
                                     head_path=args.head_path,
                                     input_mode=reg["input_mode"])

        base_acc = score(handler, ray.get(engines[0].predict.remote(
            test_items, reg["input_mode"], None)), test_items)
        print(f"[{args.dataset}] BASE (unmodified): {base_acc:.4f}")

        class _Args:
            pass
        a = _Args()
        a.num_engines = len(engines)
        a.max_top_k = k
        a.top_k_list = [k]

        eval_sets = [{"name": args.dataset, "items": test_items, "logit_mask": None,
                      "input_mode": reg["input_mode"], "base": base_acc}]
        ensemble = run_ensemble_multi(a, engines, top_k_perturbs, eval_sets, wandb_run)
        acc = ensemble[args.dataset][k]["accuracy"]
        gain = acc - base_acc * 100
        print(f"[{args.dataset}] ImageNet-trained-RandOpt-backbone ensemble "
              f"(K={k}): {acc:.2f}% [{gain:+.2f}pp vs base]")
        if wandb_run:
            wandb_run.log({f"{args.dataset}_forget/randopt_ensemble_accuracy": acc,
                           f"{args.dataset}_forget/randopt_gain_over_base": gain})

    else:  # ft
        engines = launch_ssl_engines(1, backbone_family=args.backbone_family,
                                     num_classes=reg["num_classes"],
                                     head_path=args.head_path,
                                     input_mode=reg["input_mode"])
        base_acc = score(handler, ray.get(engines[0].predict.remote(
            test_items, reg["input_mode"], None)), test_items)
        ray.get(engines[0].load_backbone_state_dict.remote(args.ft_backbone_path))
        ft_acc = score(handler, ray.get(engines[0].predict.remote(
            test_items, reg["input_mode"], None)), test_items)
        gain = (ft_acc - base_acc) * 100
        print(f"[{args.dataset}] BASE: {base_acc:.4f} | ImageNet-trained-FT-backbone + "
              f"{args.dataset} head: {ft_acc:.4f} [{gain:+.2f}pp vs base]")
        if wandb_run:
            wandb_run.log({f"{args.dataset}_forget/base_accuracy": base_acc,
                           f"{args.dataset}_forget/ft_backbone_accuracy": ft_acc,
                           f"{args.dataset}_forget/ft_gain_over_base": gain})

    if wandb_run:
        wandb_run.finish()
    ray.shutdown()


if __name__ == "__main__":
    main(parse_args())
