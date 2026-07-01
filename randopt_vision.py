#!/usr/bin/env python3
"""
RandOpt for vision models (DINOv2 + CIFAR-10).

Applies the Neural Thickets algorithm to SSL vision encoders:
randomly perturbs pretrained DINOv2 weights, evaluates each perturbation
on a small train set, selects top-K by accuracy, and majority-votes at test time.

Usage (local):
    python randopt_vision.py \
        --dataset cifar10 \
        --model_name facebook/dinov2-small \
        --num_engines 1 \
        --cuda_devices 0 \
        --population_size 20 \
        --train_samples 500 \
        --sigma_values "0.0001,0.001,0.01,0.1" \
        --wandb_project randopt
"""
import argparse
from collections import Counter
from datetime import datetime
import gc
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import ray
import torch

from data_handlers import get_dataset_handler, list_datasets
from vision import launch_vision_engines, cleanup_vision_engines

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="RandOpt for vision (DINOv2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="cifar10",
                        help="Vision dataset name (currently: cifar10)")
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="Override default train data path (root dir for torchvision datasets)")
    parser.add_argument("--test_data_path", type=str, default=None,
                        help="Override default test data path")
    parser.add_argument("--train_samples", type=int, default=500,
                        help="Number of train samples used for perturbation scoring")
    parser.add_argument("--test_samples", type=int, default=None,
                        help="Max test samples (None = all)")
    parser.add_argument("--model_name", type=str, default="facebook/dinov2-base",
                        help="HuggingFace DINOv2 model ID (e.g. facebook/dinov2-small, facebook/dinov2-base)")
    parser.add_argument("--num_classes", type=int, default=10,
                        help="Number of output classes")
    parser.add_argument("--linear_init_path", type=str, default=None,
                        help="Path to a pretrained linear head .pt file (from vision/train_linear_probe.py). "
                             "If not set, uses random init.")
    parser.add_argument("--perturb_target", type=str, default="all",
                        choices=["all", "classifier"],
                        help="Which weights to perturb: 'all' (backbone+classifier) or 'classifier' (head only).")
    parser.add_argument("--inference_batch_size", type=int, default=64,
                        help="Images per GPU forward pass inside VisionEngine")
    parser.add_argument("--sigma_values", type=str, default="0.0001,0.001,0.01,0.1",
                        help="Comma-separated noise scales to search over")
    parser.add_argument("--population_size", type=int, default=50,
                        help="Total number of perturbations to evaluate")
    parser.add_argument("--top_k_ratios", type=str, default="0.01,0.05,0.1",
                        help="Fractions of population_size to use for ensemble")
    parser.add_argument("--num_engines", type=int, default=1,
                        help="Number of parallel VisionEngine actors (= GPUs to use)")
    parser.add_argument("--cuda_devices", type=str, default="0",
                        help="Comma-separated CUDA device indices visible to this process")
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--experiment_dir", type=str, default="results/vision",
                        help="Directory to save results")
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Resume from previous run dir (skip sampling, run ensemble only)")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="W&B project name. Leave unset to disable logging.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="W&B run name. Defaults to dataset_model_timestamp.")

    args = parser.parse_args()

    args.sigma_list = [float(s.strip()) for s in args.sigma_values.split(",")]
    ratios = [float(r.strip()) for r in args.top_k_ratios.split(",")]
    args.top_k_list = sorted(set(max(1, int(r * args.population_size)) for r in ratios), reverse=True)
    args.max_top_k = args.top_k_list[0]

    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.manual_seed(args.global_seed)

    return args


# ---------------------------------------------------------------------------
# W&B
# ---------------------------------------------------------------------------

def init_wandb(args):
    if not WANDB_AVAILABLE or not args.wandb_project:
        return
    run_name = args.wandb_run_name or (
        f"{args.dataset}_{args.model_name.split('/')[-1]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={k: v for k, v in vars(args).items() if k not in ("sigma_list", "top_k_list")},
        resume="allow",
    )


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_data(handler, args):
    train_path = args.train_data_path or handler.default_train_path
    test_path  = args.test_data_path  or handler.default_test_path

    print(f"Loading {handler.name} data...")
    train_datas = handler.load_data(train_path, split="train", max_samples=args.train_samples)
    test_datas  = handler.load_data(test_path,  split="test",  max_samples=args.test_samples)
    print(f"  Train: {len(train_datas)} | Test: {len(test_datas)}")
    return train_datas, test_datas


def stack_images(datas: List[Dict]) -> torch.Tensor:
    """Stack image_tensor fields into a single (N, C, H, W) tensor."""
    return torch.stack([d["image_tensor"] for d in datas])


# ---------------------------------------------------------------------------
# Algorithm phases
# ---------------------------------------------------------------------------

def evaluate_base_model(engines, handler, train_images, test_images, train_datas, test_datas):
    """Evaluate the unperturbed model on train and test sets."""
    print(f"\n{'='*60}\nBASE MODEL EVALUATION\n{'='*60}")

    train_preds = ray.get(engines[0].forward.remote(train_images))
    base_train_reward = handler.postprocess_outputs(train_preds, train_datas)
    print(f"Train accuracy: {base_train_reward*100:.2f}%")

    test_preds = ray.get(engines[0].forward.remote(test_images))
    base_test_accuracy = handler.postprocess_outputs(test_preds, test_datas)
    print(f"Test  accuracy: {base_test_accuracy*100:.2f}% ({int(base_test_accuracy*len(test_datas))}/{len(test_datas)})")

    if WANDB_AVAILABLE and wandb.run:
        wandb.log({
            "base/train_reward": base_train_reward,
            "base/test_accuracy": base_test_accuracy,
        })

    return base_train_reward, base_test_accuracy


def run_sampling(args, engines, handler, train_images, train_datas):
    """Perturbation sampling: evaluate population_size (seed, sigma) pairs."""
    print(f"\n{'='*60}\nPERTURBATION SAMPLING\n{'='*60}")
    print(f"Budget: {args.population_size} | Sigmas: {args.sigma_list}")

    rng = np.random.default_rng(seed=args.global_seed)
    perf: Dict[Tuple[int, float], float] = {}

    all_seeds  = rng.choice(2**31, size=args.population_size, replace=False).tolist()
    all_sigmas = rng.choice(args.sigma_list, size=args.population_size).tolist()
    seed_idx = 0

    samples_evaluated, batch_idx = 0, 0

    while samples_evaluated < args.population_size:
        batch_size = min(args.num_engines, args.population_size - samples_evaluated)
        batch = [(int(all_seeds[seed_idx + i]), float(all_sigmas[seed_idx + i])) for i in range(batch_size)]
        seed_idx += batch_size

        # Perturb
        ray.get([engines[i].perturb_weights.remote(seed, sigma) for i, (seed, sigma) in enumerate(batch)])

        # Infer on train set
        preds_list = ray.get([engines[i].forward.remote(train_images) for i in range(batch_size)])

        # Restore
        ray.get([engines[i].restore_weights.remote(seed, sigma) for i, (seed, sigma) in enumerate(batch)])

        # Compute rewards (top-1 accuracy on train)
        rewards = []
        for i, (seed, sigma) in enumerate(batch):
            r = handler.postprocess_outputs(preds_list[i], train_datas)
            perf[(seed, sigma)] = r
            rewards.append(r)

        samples_evaluated += batch_size
        batch_idx += 1
        print(f"  Batch {batch_idx} | {samples_evaluated}/{args.population_size} | {['%.3f' % r for r in rewards]}")

        if WANDB_AVAILABLE and wandb.run:
            wandb.log({
                "sampling/samples_evaluated": samples_evaluated,
                "sampling/batch_mean_reward": float(np.mean(rewards)),
                "sampling/batch_max_reward":  float(np.max(rewards)),
            }, step=samples_evaluated)

    print(f"\nSampling done.")

    # Per-sigma summary
    print(f"\n{'='*60}\nSAMPLING COMPLETE\n{'='*60}")
    sigma_rewards: Dict[float, List[float]] = {s: [] for s in args.sigma_list}
    for (_, sigma), reward in perf.items():
        sigma_rewards[sigma].append(reward)

    sigma_log = {}
    for sigma in args.sigma_list:
        rs = sigma_rewards[sigma]
        if rs:
            mean_r = float(np.mean(rs))
            print(f"  σ={sigma}: mean={mean_r:.4f}, n={len(rs)}")
            sigma_log[f"sigma/{sigma}/mean_reward"] = mean_r

    if WANDB_AVAILABLE and wandb.run and sigma_log:
        wandb.log(sigma_log)

    best_sigma = max(args.sigma_list, key=lambda s: np.mean(sigma_rewards[s]) if sigma_rewards[s] else 0)
    print(f"\n★ Best sigma: {best_sigma}")
    if WANDB_AVAILABLE and wandb.run:
        wandb.log({"sampling/best_sigma": best_sigma})

    return perf, best_sigma


def run_ensemble_evaluation(args, engines, handler, test_images, test_datas, top_k_perturbs, base_test):
    """Majority-vote ensemble over top-K perturbed models on the test set."""
    max_k = min(args.max_top_k, len(top_k_perturbs))
    num_samples = len(test_datas)
    eval_k_values = [k for k in args.top_k_list if k <= max_k]

    print(f"\n{'='*60}\nENSEMBLE EVALUATION\n{'='*60}")
    print(f"K values: {eval_k_values} | Test samples: {num_samples}")

    # all_preds[model_idx][sample_idx] = class_id (int)
    all_preds: List[List[int]] = [None] * max_k

    total_batches = (max_k + args.num_engines - 1) // args.num_engines

    for batch_idx in range(total_batches):
        start = batch_idx * args.num_engines
        end   = min(start + args.num_engines, max_k)
        batch_perturbs = top_k_perturbs[start:end]

        if batch_idx % 10 == 0 or batch_idx == total_batches - 1:
            print(f"  Batch {batch_idx + 1}/{total_batches} ({len(batch_perturbs)} models)...", flush=True)

        ray.get([engines[i].perturb_weights.remote(int(s), sig) for i, (s, sig) in enumerate(batch_perturbs)])

        batch_preds = ray.get([engines[i].forward.remote(test_images) for i in range(len(batch_perturbs))])

        ray.get([engines[i].restore_weights.remote(int(s), sig) for i, (s, sig) in enumerate(batch_perturbs)])

        for local_idx, global_idx in enumerate(range(start, end)):
            all_preds[global_idx] = batch_preds[local_idx]

        del batch_preds
        gc.collect()

    print(f"\nGeneration completed. Running majority voting...")

    ensemble_results = {}
    for k_value in eval_k_values:
        correct = 0
        for idx, data in enumerate(test_datas):
            votes = [all_preds[m][idx] for m in range(k_value)]
            final = Counter(votes).most_common(1)[0][0]
            if final == data["class_id"]:
                correct += 1

        acc = correct / num_samples * 100
        ensemble_results[k_value] = {"accuracy": acc, "correct": correct}
        print(f"  K={k_value}: {acc:.2f}% ({correct}/{num_samples}) [+{acc - base_test*100:.2f}% vs base]")

        if WANDB_AVAILABLE and wandb.run:
            wandb.log({
                f"ensemble/k{k_value}/accuracy": acc,
                f"ensemble/k{k_value}/gain_over_base": acc - base_test * 100,
            })

    del all_preds
    gc.collect()

    return ensemble_results


def save_results(args, logging_dir, model_saves_dir, base_train_reward, base_test_accuracy,
                 top_k_perturbs, top_k_rewards, ensemble_results, perf, best_sigma):
    print(f"\n=== Saving Results ===")

    seeds_info = {
        "base_model": args.model_name,
        "best_sigma": best_sigma,
        "top_k_models": [
            {"rank": i + 1, "seed": int(seed), "sigma": float(sigma), "train_reward": float(reward)}
            for i, ((seed, sigma), reward) in enumerate(zip(top_k_perturbs, top_k_rewards))
        ],
    }
    with open(f"{model_saves_dir}/top_k_seeds.json", "w") as f:
        json.dump(seeds_info, f, indent=4)

    sigma_rewards: Dict[float, List[float]] = {s: [] for s in args.sigma_list}
    for (_, sigma), reward in perf.items():
        sigma_rewards[sigma].append(reward)
    sigma_stats = {
        str(s): {"mean": float(np.mean(sigma_rewards[s])) if sigma_rewards[s] else 0.0,
                 "count": len(sigma_rewards[s])}
        for s in args.sigma_list
    }

    results = {
        "dataset": args.dataset,
        "model": args.model_name,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "base_train_reward": base_train_reward,
        "base_test_accuracy": base_test_accuracy,
        "sigma_stats": sigma_stats,
        "best_sigma": best_sigma,
        "ensemble_results": {str(k): v for k, v in ensemble_results.items()},
        "top_k_perturbs": [(int(s), float(sig)) for s, sig in top_k_perturbs],
        "top_k_train_rewards": [float(r) for r in top_k_rewards],
    }
    with open(f"{logging_dir}/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {logging_dir}/")

    if WANDB_AVAILABLE and wandb.run:
        wandb.log({"results/logging_dir": logging_dir})
        wandb.finish()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    handler = get_dataset_handler(args.dataset)
    is_resume = args.resume_dir is not None

    init_wandb(args)

    print(f"{'='*60}")
    print(f"RandOpt Vision — {handler.name.upper()} {'[RESUME]' if is_resume else ''}")
    print(f"{'='*60}")
    print(f"Model      : {args.model_name}")
    print(f"Population : {args.population_size} | Top-K: {args.top_k_list} | Engines: {args.num_engines}")
    print(f"Sigmas     : {args.sigma_list}")
    print(f"Linear init: {args.linear_init_path or 'random'}")

    if os.environ.get("RAY_ADDRESS"):
        ray.init(address="auto", ignore_reinit_error=True)
    else:
        ray.init(ignore_reinit_error=True)

    # Setup directories
    if is_resume:
        logging_dir = f"{args.experiment_dir}/{args.dataset}_resume_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        logging_dir = f"{args.experiment_dir}/{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_saves_dir = f"{logging_dir}/model_saves"
    os.makedirs(model_saves_dir, exist_ok=True)

    with open(f"{logging_dir}/args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load data
    train_datas, test_datas = load_data(handler, args)
    train_images = stack_images(train_datas)
    test_images  = stack_images(test_datas)
    print(f"  Train tensor: {tuple(train_images.shape)} | Test tensor: {tuple(test_images.shape)}")

    # Launch engines
    engines = launch_vision_engines(
        num_engines=args.num_engines,
        model_name=args.model_name,
        num_classes=args.num_classes,
        linear_init_path=args.linear_init_path,
        inference_batch_size=args.inference_batch_size,
        perturb_target=args.perturb_target,
    )
    print(f"Launched {len(engines)} VisionEngine(s).")

    try:
        if not is_resume:
            base_train_reward, base_test_accuracy = evaluate_base_model(
                engines, handler, train_images, test_images, train_datas, test_datas)

            perf, best_sigma = run_sampling(
                args, engines, handler, train_images, train_datas)

            # Selection
            print(f"\n{'='*60}\nSELECTION\n{'='*60}")
            sorted_perturbs = sorted(perf.items(), key=lambda x: x[1], reverse=True)
            top_k_perturbs = [(seed, sigma) for (seed, sigma), _ in sorted_perturbs[:args.max_top_k]]
            top_k_rewards  = [reward for _, reward in sorted_perturbs[:args.max_top_k]]

            print(f"Selected top-{args.max_top_k} from {args.population_size} perturbations")
            for i, ((seed, sigma), reward) in enumerate(sorted_perturbs[:10]):
                print(f"  {i+1}. seed={seed}, σ={sigma}: {reward:.4f}")
        else:
            with open(f"{args.resume_dir}/model_saves/top_k_seeds.json") as f:
                saved = json.load(f)
            best_sigma     = saved["best_sigma"]
            top_k_perturbs = [(m["seed"], m["sigma"]) for m in saved["top_k_models"]]
            top_k_rewards  = [m["train_reward"] for m in saved["top_k_models"]]

            with open(f"{args.resume_dir}/results.json") as f:
                prev = json.load(f)
            base_train_reward  = prev["base_train_reward"]
            base_test_accuracy = prev["base_test_accuracy"]
            perf = {(s, sig): r for (s, sig), r in zip(top_k_perturbs, top_k_rewards)}

            print(f"Resumed from: {args.resume_dir} ({len(top_k_perturbs)} models)")

        # Ensemble evaluation
        ensemble_results = run_ensemble_evaluation(
            args, engines, handler, test_images, test_datas,
            top_k_perturbs, base_test_accuracy)

        save_results(args, logging_dir, model_saves_dir, base_train_reward,
                     base_test_accuracy, top_k_perturbs, top_k_rewards,
                     ensemble_results, perf, best_sigma)

    finally:
        cleanup_vision_engines(engines)


if __name__ == "__main__":
    args = parse_args()
    main(args)
