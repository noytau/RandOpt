"""RandOpt on self-contained datasets — each with its OWN train/val/test
split (exdark first; fgvc_aircraft/stanford_dogs/domainnet_clipart/
domainnet_sketch are meant to be one more registry line + handler each,
later) — as opposed to scripts/randopt_shift.py's protocol, where scoring is
ALWAYS on clean ImageNet and only the final eval target varies.

Protocol here is the "honest" train/test split:
  - Perturbation SCORING (the "train" phase) runs on the dataset's OWN train
    split.
  - The FINAL evaluation (base_test_accuracy + the top-K majority-vote
    ensemble) runs on the SAME dataset's own test split.
  - There is no frozen ImageNet-1k head to reuse here (these datasets are
    outside ImageNet's label space) — RandOpt perturbs around a dedicated
    N-way head instead, fit ONCE ahead of time by scripts/fit_head.py
    (`vision/head_probe.py`). This script requires that checkpoint to
    already exist (checkpoints/<dataset>_head.pth by default) and fails
    fast with instructions if it's missing, rather than silently mixing
    raw-CUDA head-fitting into the same process as the Ray engines below.
  - No logit_mask anywhere: every dataset here has its own private label
    space (no shared head to restrict), so there's nothing to mask.

--dataset is a comma-separated plusarg (same style as randopt_shift.py's),
resolved through _SELFCONTAINED_REGISTRY. Unlike randopt_shift.py, naming
multiple datasets does NOT share one scoring phase across them (each scores
on its own train split) — it just runs each dataset's full protocol in turn,
one after another, in the same process.

run_sampling/run_ensemble_multi are imported from randopt_shift.py UNCHANGED
— both were already fully dataset-agnostic (schema-driven over items/labels,
no ImageNet-specific logic) — this script only supplies where the scoring
set and the head come from.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class, score  # noqa: E402
from scripts.randopt_shift import run_ensemble_multi, run_sampling  # noqa: E402

# One entry per self-contained dataset: its own manifest (None = the
# handler's own default_train_path — a single manifest file holding all 3
# splits, per scripts/data_prep/make_new_dataset_manifests.py), how many
# classes its own private label space has, which transform its images need
# (all 5 new datasets are raw variable-size JPEGs -> "official_resize"), and
# score_split -- which manifest split RandOpt's scoring set is drawn from.
# exdark's score_split is "train_holdout" (carved out of official train by
# make_new_dataset_manifests.py), NOT plain "train": the linear head
# (scripts/fit_head.py) is fit on the rest of "train", so scoring on the
# same images the head already fit to would just re-measure memorization,
# not real generalization. Defaults to "train" for any future dataset that
# doesn't need this split (e.g. one whose head comes pretrained already).
_SELFCONTAINED_REGISTRY = {
    "exdark": {"manifest": None, "num_classes": 12, "input_mode": "official_resize",
               "score_split": "train_holdout"},
}


def _parse_dataset_list(raw: str):
    names = []
    for token in (t.strip() for t in raw.split(",")):
        if token not in _SELFCONTAINED_REGISTRY:
            raise ValueError(f"unknown --dataset entry '{token}' -- valid: "
                              f"{list(_SELFCONTAINED_REGISTRY)}")
        names.append(token)
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                    help="comma-separated self-contained targets, any of: " +
                         ", ".join(_SELFCONTAINED_REGISTRY))
    p.add_argument("--backbone_family", default="dinov2", choices=["dinov2", "dinov3"])
    p.add_argument("--backbone_name", default=None)
    p.add_argument("--weights_path", default=None)
    p.add_argument("--head_dir", default="checkpoints",
                    help="where fitted head checkpoints live, written by "
                         "scripts/fit_head.py (<head_dir>/<dataset>_head.pth)")
    p.add_argument("--population_size", type=int, default=30)
    p.add_argument("--sigma_values", default="0.0005,0.001,0.002")
    p.add_argument("--top_k_ratios", default="0.05,0.1,0.2")
    p.add_argument("--num_engines", type=int, default=1)
    p.add_argument("--perturb_target", default="all",
                    choices=["all", "head", "last_n_blocks"])
    p.add_argument("--last_n_blocks", type=int, default=0)
    p.add_argument("--train_samples", type=int, default=0,
                    help="scoring-set size, class-balanced sample from the "
                         "dataset's own train split; 0 = full split")
    p.add_argument("--test_samples", type=int, default=0,
                    help="eval-set size from the dataset's own test split; "
                         "0 = full split")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    args = p.parse_args()
    args.dataset_targets = _parse_dataset_list(args.dataset)
    args.sigma_list = [float(s) for s in args.sigma_values.split(",")]
    ratios = [float(r) for r in args.top_k_ratios.split(",")]
    args.top_k_list = sorted({max(1, int(r * args.population_size))
                              for r in ratios}, reverse=True)
    args.max_top_k = args.top_k_list[0]
    return args


def _head_path(dataset: str, head_dir: str) -> str:
    path = os.path.join(head_dir, f"{dataset}_head.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no fitted head for '{dataset}' at {path} -- run "
            f"`python scripts/fit_head.py --dataset {dataset}` first. "
            f"RandOpt needs a trained center to perturb around; datasets "
            f"outside ImageNet-1k's label space have no pretrained head to "
            f"reuse the way imagenet_a/r/sketch/es do.")
    return path


def main(args):
    fam = "" if args.backbone_family == "dinov2" else f"-{args.backbone_family}"
    dataset_tag = "-".join(args.dataset_targets)
    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_name or (
            f"randopt-ssl{fam}-self-{dataset_tag}-N{args.population_size}")
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config=vars(args))

    from data_handlers import get_dataset_handler

    rng = np.random.default_rng(args.global_seed)
    ray_tmp = os.path.join(os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets"),
                           "ray_tmp")
    os.makedirs(ray_tmp, exist_ok=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=ray_tmp)
    from vision import launch_ssl_engines

    all_results = {}
    for dataset in args.dataset_targets:
        reg = _SELFCONTAINED_REGISTRY[dataset]
        handler = get_dataset_handler(dataset)
        manifest = reg["manifest"] or handler.default_train_path
        head_path = _head_path(dataset, args.head_dir)
        # run_sampling (reused from randopt_shift.py) reads args.train_input_mode
        # directly rather than taking it as a parameter -- this dataset's own
        # train split needs its own registry input_mode, not clean ImageNet's.
        args.train_input_mode = reg["input_mode"]

        score_split = reg.get("score_split", "train")
        train_items = sample_per_class(
            handler.load_data(manifest, split=score_split), args.train_samples, rng)
        test_items = sample_per_class(
            handler.load_data(manifest, split="test"), args.test_samples, rng)
        print(f"[{dataset}] scoring set (own {score_split}): {len(train_items)} | "
              f"eval set (own test): {len(test_items)} | head: {head_path}")

        engines = launch_ssl_engines(args.num_engines,
                                     backbone_family=args.backbone_family,
                                     backbone_name=args.backbone_name,
                                     weights_path=args.weights_path,
                                     head_path=head_path,
                                     num_classes=reg["num_classes"],
                                     perturb_target=args.perturb_target,
                                     last_n_blocks=args.last_n_blocks,
                                     inference_batch_size=args.batch_size,
                                     input_mode=reg["input_mode"])
        # launch_ssl_engines silently reduces engine count if fewer GPUs are
        # actually visible to Ray than requested (e.g. Geoffry's env script
        # defaults CUDA_VISIBLE_DEVICES to a single GPU) -- run_sampling/
        # run_ensemble_multi (imported from randopt_shift.py) index engines[i]
        # up to args.num_engines, so that must track the REAL engine count,
        # not the originally requested one, or they IndexError.
        args.num_engines = len(engines)
        n_scope = ray.get(engines[0].count_perturb_params.remote())
        print(f"[{dataset}] perturb scope: {args.perturb_target} = "
              f"{n_scope / 1e6:.1f}M params")

        t0 = time.time()
        base_train = score(handler, ray.get(engines[0].predict.remote(
            train_items, reg["input_mode"], None)), train_items)
        base_test = score(handler, ray.get(engines[0].predict.remote(
            test_items, reg["input_mode"], None)), test_items)
        print(f"[{dataset}] BASE: train_reward={base_train:.4f} "
              f"test_accuracy={base_test:.4f} ({time.time() - t0:.0f}s)")
        if wandb_run:
            wandb_run.log({f"{dataset}/base/train_reward": base_train,
                           f"{dataset}/base/test_accuracy": base_test})

        # scoring is always on this dataset's OWN train split -> logit_mask
        # is always None (private label space, nothing to restrict)
        perf, best_sigma = run_sampling(args, engines, handler, train_items,
                                        None, wandb_run)
        ray.get([e.reset_to_base_weights.remote() for e in engines])

        top = sorted(perf.items(), key=lambda kv: kv[1], reverse=True)
        top_k_perturbs = [k for k, _ in top[:args.max_top_k]]
        top_k_rewards = [v for _, v in top[:args.max_top_k]]
        print(f"[{dataset}] top-{args.max_top_k} train rewards: "
              f"{['%.3f' % r for r in top_k_rewards[:5]]}...")

        eval_sets = [{"name": dataset, "items": test_items, "logit_mask": None,
                      "input_mode": reg["input_mode"], "base": base_test}]
        ensemble = run_ensemble_multi(args, engines, top_k_perturbs, eval_sets,
                                      wandb_run)

        exp_dir = args.experiment_dir or (
            f"results/randopt-ssl{fam}-self-{dataset}-N{args.population_size}")
        os.makedirs(exp_dir, exist_ok=True)
        result = {
            "dataset": dataset,
            "base_train_reward": base_train,
            "base_test_accuracy": base_test,
            "best_sigma": best_sigma,
            "ensemble_results": {str(k): v for k, v in ensemble[dataset].items()},
            "top_k_perturbs": top_k_perturbs,
            "top_k_train_rewards": top_k_rewards,
            "head_path": head_path,
            "config": vars(args),
        }
        with open(os.path.join(exp_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"[{dataset}] saved {exp_dir}/results.json")
        all_results[dataset] = result

        for e in engines:
            ray.kill(e)

    if wandb_run:
        wandb_run.finish()
    ray.shutdown()
    return all_results


if __name__ == "__main__":
    main(parse_args())
