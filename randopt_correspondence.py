"""RandOpt for semantic correspondence on SSL vision models.

No linear head. Reward = PCK@0.1 computed from raw DINOv2 patch embeddings.

Algorithm:
  1. Load SPair-71k image pairs
  2. Evaluate base DINOv2: PCK@0.1 on train pairs
  3. For each (seed, sigma): perturb backbone → get_patch_features → PCK → restore
  4. Top-K by train PCK → ensemble on test (average logit/sim, take argmax per kpt)
  5. Log all metrics to W&B

Usage:
  python randopt_correspondence.py \
    --dataset spair71k \
    --model_name facebook/dinov2-base \
    --num_engines 1 --cuda_devices 0 \
    --population_size 500 \
    --sigma_values 0.00001,0.0001,0.001,0.01 \
    --train_samples 500 \
    --test_samples 1000
"""
import argparse
import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import ray
import torch

from data_handlers import get_dataset_handler
from data_handlers.spair71k import compute_pck, GRID
from vision import launch_vision_engines


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="spair71k")
    p.add_argument("--model_name", default="facebook/dinov2-base")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--num_engines", type=int, default=1)
    p.add_argument("--cuda_devices", default="0")
    p.add_argument("--population_size", type=int, default=500)
    p.add_argument("--sigma_values", default="0.00001,0.0001,0.001,0.01")
    p.add_argument("--top_k_ratios", default="0.01,0.05,0.1")
    p.add_argument("--train_samples", type=int, default=500)
    p.add_argument("--test_samples", type=int, default=1000)
    p.add_argument("--inference_batch_size", type=int, default=32)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def stack_images(datas: List[Dict], key: str = "image_tensor") -> torch.Tensor:
    return torch.stack([d[key] for d in datas])


def eval_pck(engines, datas: List[Dict], batch_size: int = 32) -> float:
    """Compute mean PCK@0.1 over all pairs using the current engine weights."""
    engine = engines[0]
    src_imgs = stack_images(datas, "image_tensor")
    tgt_imgs = stack_images(datas, "image_tensor_tgt")

    src_feats = ray.get(engine.get_patch_features.remote(src_imgs))  # (N, P, D)
    tgt_feats = ray.get(engine.get_patch_features.remote(tgt_imgs))

    scores = []
    for i, d in enumerate(datas):
        pck = compute_pck(src_feats[i], tgt_feats[i], d["kpts_src"], d["kpts_tgt"],
                          bbox_thresh=d.get("bbox_thresh", 1.6))
        scores.append(pck)
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # W&B
    wandb_run = None
    if args.wandb_project:
        import wandb
        run_name = args.wandb_run_name or f"corr-{args.dataset}-n{args.population_size}"
        wandb_run = wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Data
    handler = get_dataset_handler(args.dataset)
    data_dir = args.data_dir or handler.default_train_path
    print(f"Loading {args.dataset} data...")
    train_datas = handler.load_data(data_dir, split="train", max_samples=args.train_samples)
    test_datas  = handler.load_data(data_dir, split="test",  max_samples=args.test_samples)
    print(f"  Train pairs: {len(train_datas)} | Test pairs: {len(test_datas)}")

    # Engines (backbone only, no classifier for correspondence)
    ray.init(ignore_reinit_error=True)
    engines = launch_vision_engines(
        num_engines=args.num_engines,
        model_name=args.model_name,
        num_classes=1,           # dummy — classifier not used
        linear_init_path=None,
        perturb_target="all",
    )

    sigma_values = [float(s) for s in args.sigma_values.split(",")]
    top_k_ratios = [float(r) for r in args.top_k_ratios.split(",")]

    # --- Phase 1: Base model eval ---
    print("\n=== Base model evaluation ===")
    base_train_pck = eval_pck(engines, train_datas)
    base_test_pck  = eval_pck(engines, test_datas)
    print(f"Base train PCK@0.1: {base_train_pck*100:.2f}%")
    print(f"Base test  PCK@0.1: {base_test_pck*100:.2f}%")
    if wandb_run:
        wandb_run.log({"base/train_pck": base_train_pck, "base/test_pck": base_test_pck})

    # --- Phase 2: Perturbation sampling ---
    print(f"\n=== Sampling {args.population_size} perturbations ===")
    rng = np.random.default_rng(42)
    all_seeds  = rng.integers(0, 2**31, size=args.population_size)
    all_sigmas = rng.choice(sigma_values, size=args.population_size)

    results: List[Tuple[int, float, float]] = []  # (seed, sigma, train_pck)
    seed_idx, samples_evaluated = 0, 0

    src_imgs   = stack_images(train_datas, "image_tensor")
    tgt_imgs   = stack_images(train_datas, "image_tensor_tgt")

    while samples_evaluated < args.population_size:
        batch_size = min(args.num_engines, args.population_size - samples_evaluated)
        batch = [(int(all_seeds[seed_idx + i]), float(all_sigmas[seed_idx + i]))
                 for i in range(batch_size)]
        seed_idx += batch_size

        ray.get([engines[i].perturb_weights.remote(s, sig) for i, (s, sig) in enumerate(batch)])

        # Patch features per engine
        batch_pcks = []
        for i, (seed, sigma) in enumerate(batch):
            sf = ray.get(engines[i].get_patch_features.remote(src_imgs))
            tf = ray.get(engines[i].get_patch_features.remote(tgt_imgs))
            pcks = [compute_pck(sf[j], tf[j], train_datas[j]["kpts_src"],
                                train_datas[j]["kpts_tgt"],
                                bbox_thresh=train_datas[j].get("bbox_thresh", 1.6))
                    for j in range(len(train_datas))]
            batch_pcks.append(float(np.mean(pcks)))

        ray.get([engines[i].restore_weights.remote(s, sig) for i, (s, sig) in enumerate(batch)])

        for i, (seed, sigma) in enumerate(batch):
            results.append((seed, sigma, batch_pcks[i]))

        samples_evaluated += batch_size
        print(f"  {samples_evaluated}/{args.population_size} | "
              f"mean={np.mean(batch_pcks)*100:.2f}% max={np.max(batch_pcks)*100:.2f}%")
        if wandb_run:
            wandb_run.log({
                "sampling/batch_mean_pck": float(np.mean(batch_pcks)),
                "sampling/batch_max_pck":  float(np.max(batch_pcks)),
                "sampling/samples_evaluated": samples_evaluated,
            })

    # Sigma analysis
    for sigma in sigma_values:
        s_pcks = [r[2] for r in results if abs(r[1] - sigma) < 1e-9]
        if s_pcks and wandb_run:
            wandb_run.log({f"sigma/{sigma}/mean_pck": float(np.mean(s_pcks))})

    results.sort(key=lambda x: x[2], reverse=True)
    best_sigma = results[0][1]
    print(f"\nBest sigma: {best_sigma} | Best train PCK: {results[0][2]*100:.2f}%")
    if wandb_run:
        wandb_run.log({"sampling/best_sigma": best_sigma})

    # --- Phase 3: Ensemble evaluation ---
    print("\n=== Ensemble evaluation ===")
    src_test_imgs = stack_images(test_datas, "image_tensor")
    tgt_test_imgs = stack_images(test_datas, "image_tensor_tgt")

    ensemble_results = {}
    for ratio in top_k_ratios:
        K = max(1, int(ratio * args.population_size))
        top_k = results[:K]
        print(f"\n  Top-K={K} ensemble...")

        # Accumulate similarity matrices: (N, P_src, P_tgt) — average across ensemble
        N = len(test_datas)
        P = GRID * GRID
        sim_accum = torch.zeros(N, P, P)

        total_batches = (K + args.num_engines - 1) // args.num_engines
        for b in range(total_batches):
            batch_perturbs = top_k[b * args.num_engines : (b + 1) * args.num_engines]
            ray.get([engines[i].perturb_weights.remote(int(s), sig)
                     for i, (s, sig, _) in enumerate(batch_perturbs)])

            for i, (seed, sigma, _) in enumerate(batch_perturbs):
                sf = ray.get(engines[i].get_patch_features.remote(src_test_imgs))  # (N,P,D)
                tf = ray.get(engines[i].get_patch_features.remote(tgt_test_imgs))
                # sim: (N, P, P) via batched matmul
                sim_accum += torch.bmm(sf, tf.transpose(1, 2))

            ray.get([engines[i].restore_weights.remote(int(s), sig)
                     for i, (s, sig, _) in enumerate(batch_perturbs)])

        sim_accum /= K

        # PCK from averaged similarity
        pcks = []
        for j, d in enumerate(test_datas):
            pck = compute_pck(
                sim_accum[j], None,
                d["kpts_src"], d["kpts_tgt"],
                bbox_thresh=d.get("bbox_thresh", 1.6),
                precomputed_sim=True,
            )
            pcks.append(pck)

        ensemble_pck = float(np.mean(pcks))
        gain = ensemble_pck - base_test_pck
        ensemble_results[K] = {"pck": ensemble_pck, "gain": gain}
        print(f"    K={K}: test PCK={ensemble_pck*100:.2f}% | gain={gain*100:+.2f}pp vs base {base_test_pck*100:.2f}%")
        if wandb_run:
            wandb_run.log({
                f"ensemble/k{K}/pck":           ensemble_pck,
                f"ensemble/k{K}/gain_over_base": gain,
            })

    # Save results
    exp_dir = args.experiment_dir or f"results/corr_{args.dataset}_n{args.population_size}"
    os.makedirs(exp_dir, exist_ok=True)
    out = {
        "base_train_pck": base_train_pck,
        "base_test_pck":  base_test_pck,
        "best_sigma": best_sigma,
        "ensemble_results": {str(k): v for k, v in ensemble_results.items()},
        "top_k_perturbs": [(int(s), float(sig)) for s, sig, _ in results[:max(
            int(r * args.population_size) for r in top_k_ratios
        )]],
    }
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {exp_dir}/results.json")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
