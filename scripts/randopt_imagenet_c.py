"""E1 Task 4: RandOpt on ImageNet-C around a backbone-only DINOv2 center (kNN eval).

Protocol (spec: experiments/E1_imagenet_c.md):
  1. Load one corruption/severity via the imagenet_c handler:
       train split (train_per_class per class) — the ONLY data touched
                                                 during selection
       test split  (test_per_class per class)  — touched once, at the end
     (the val split is reserved for larger-tier hyperparameter selection and
     is never loaded here)
  2. Inside train, build the scoring sets: gallery = first image of each
     class, scoring queries = a seeded subsample of the rest (capped at
     --n_score_queries). Each perturbation is scored by kNN top-1 of the
     scoring queries against the gallery (both re-embedded under the
     perturbed weights — features move with the weights).
  3. Cells are "scope:sigma" tokens. Per-perturbation seeds/scores are dumped
     to results.json for cross-job pooling (use a different --global_seed per
     job).
  4. Selection: pool ALL cells, rank by scoring accuracy, take top-K.
  5. Ensemble: each top-K model predicts every test image via kNN
     (gallery = FULL train split); majority vote across models;
     report accuracy vs the unperturbed base.

Selection honesty: only ENSEMBLE numbers are quotable (selection on train,
reporting on the untouched test split). Do NOT quote best-single-perturbation
scoring maxima — they are inflated by the winner's curse. The train/test split
design itself is queued for review with Nimrod (TASKS.md) before any larger
tier.

Single engine per job — fan cells out across cluster jobs.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import ray
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handlers import get_dataset_handler
from vision import launch_vision_engines


def parse_cell(tok: str) -> Tuple[str, int, str, float]:
    """'last4:0.003' -> (target, n_blocks, label, sigma); 'all:...' supported."""
    scope, sig = tok.strip().split(":")
    if scope == "all":
        return ("all", 0, "all", float(sig))
    if scope.startswith("last"):
        return ("last_n_blocks", int(scope[4:]), scope, float(sig))
    raise ValueError(f"unknown scope in cell token: {tok}")


def build_sets(args) -> Dict:
    """Load train/test and carve the kNN sets (all tensors CPU, 224x224).

    Returns dict with:
      score_gallery / score_queries  — disjoint subsets of the train split
                                       (selection signal; queries are a seeded
                                       subsample capped at n_score_queries)
      full_gallery                   — the whole train split (used for the
                                       base/ensemble kNN on test)
      test                           — test images + labels
    """
    handler = get_dataset_handler("imagenet_c")
    handler.train_per_class = args.train_per_class
    handler.val_per_class = args.val_per_class
    handler.test_per_class = args.test_per_class
    path = os.path.join(args.data_root, args.corruption, str(args.severity))

    train = handler.load_data(path, split="train")
    test = handler.load_data(path, split="test")

    # first image of each class -> gallery; the rest -> scoring-query pool
    seen, g_idx, q_idx = set(), [], []
    for i, d in enumerate(train):
        (q_idx if d["class_id"] in seen else g_idx).append(i)
        seen.add(d["class_id"])
    if args.n_score_queries and len(q_idx) > args.n_score_queries:
        rng = np.random.default_rng(args.global_seed)
        q_idx = [int(i) for i in rng.permutation(q_idx)[:args.n_score_queries]]

    def pack(items):
        return (torch.stack([d["image_tensor"] for d in items]),
                [d["class_id"] for d in items])

    sets = {
        "score_gallery": pack([train[i] for i in g_idx]),
        "score_queries": pack([train[i] for i in q_idx]),
        "full_gallery": pack(train),
        "test": pack(test),
    }
    print(f"Sets: score_gallery={len(g_idx)} score_queries={len(q_idx)} "
          f"full_gallery={len(train)} test={len(test)}")
    return sets


def score_current_weights(engine, sets) -> float:
    """kNN top-1 of the scoring queries for whatever weights the engine holds
    (the selection signal)."""
    g_imgs, g_labels = sets["score_gallery"]
    q_imgs, q_labels = sets["score_queries"]
    res = ray.get(engine.eval_global.remote(g_imgs, g_labels,
                                            [(q_imgs, q_labels)]))
    return res[0]["knn_top1"]


def main(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_run_name or f"e1-randopt-{args.corruption}-s{args.severity}"
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config=vars(args))

    sets = build_sets(args)
    test_imgs, test_labels = sets["test"]
    fg_imgs, fg_labels = sets["full_gallery"]

    engine = launch_vision_engines(
        num_engines=1, model_name=args.model_name, num_classes=1,
        linear_init_path=None, inference_batch_size=args.inference_batch_size,
        perturb_target="all")[0]

    # ---- base: unperturbed center ----
    base_score = score_current_weights(engine, sets)
    base_preds = ray.get(engine.knn_predict.remote(fg_imgs, fg_labels, test_imgs))
    base_test = float(np.mean([p == l for p, l in zip(base_preds, test_labels)]))
    print(f"BASE  score={base_score*100:.2f}  test={base_test*100:.2f}  "
          f"(headroom check: need >=5pp)")
    if wandb_run:
        wandb_run.log({"base/score_accuracy": base_score,
                       "base/test_accuracy": base_test})

    # ---- perturbation scan ----
    cells = [parse_cell(tok) for tok in args.cells.split(",")]
    rng = np.random.default_rng(args.global_seed)
    all_perts = []      # {"cell", "seed", "sigma", "target", "nb", "score"}
    cell_summaries = []
    for (target, nb, label, sigma) in cells:
        ray.get(engine.set_perturb_scope.remote(target, nb))
        pcount = ray.get(engine.count_perturb_params.remote())
        seeds = rng.integers(0, 2**31, size=args.population_size)
        scores = []
        t0 = time.time()
        print(f"\n=== cell {label}:{sigma} ({pcount:,} params, "
              f"N={args.population_size}) ===")
        for j, seed in enumerate(seeds):
            ray.get(engine.perturb_weights.remote(int(seed), float(sigma)))
            s = score_current_weights(engine, sets)
            ray.get(engine.restore_weights.remote(int(seed), float(sigma)))
            scores.append(s)
            all_perts.append({"cell": f"{label}:{sigma}", "seed": int(seed),
                              "sigma": sigma, "target": target, "nb": nb,
                              "score": s})
            if (j + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {j+1}/{len(seeds)}  [{el:.0f}s, {el/(j+1):.1f}s/pert]",
                      flush=True)
        arr = np.array(scores)
        summ = {"cell": f"{label}:{sigma}", "params": int(pcount),
                "n": len(seeds), "mean": float(arr.mean()),
                "p90": float(np.percentile(arr, 90)), "max": float(arr.max())}
        cell_summaries.append(summ)
        print(f"  mean {100*(summ['mean']-base_score):+6.2f}pp  "
              f"max {100*(summ['max']-base_score):+6.2f}pp vs base-score")
        if wandb_run:
            wandb_run.log({f"cell/{label}/s{sigma}/mean_shift":
                           summ["mean"] - base_score,
                           f"cell/{label}/s{sigma}/max_shift":
                           summ["max"] - base_score})

    # ---- selection (pooled over all cells) + majority-vote ensemble ----
    order = np.argsort([-p["score"] for p in all_perts])
    k_values = sorted({int(k) for k in args.top_k_list.split(",")
                       if int(k) <= len(all_perts)})
    max_k = max(k_values)
    top = [all_perts[i] for i in order[:max_k]]
    print(f"\nTop-{max_k} pooled perturbations (by scoring acc): "
          + ", ".join(f"{p['cell']}@{p['score']*100:.2f}" for p in top[:5])
          + " ...")

    all_preds = []
    for p in top:
        ray.get(engine.set_perturb_scope.remote(p["target"], p["nb"]))
        ray.get(engine.perturb_weights.remote(p["seed"], p["sigma"]))
        all_preds.append(ray.get(engine.knn_predict.remote(
            fg_imgs, fg_labels, test_imgs)))
        ray.get(engine.restore_weights.remote(p["seed"], p["sigma"]))

    ensemble_results = {}
    for k in k_values:
        votes = [Counter(col).most_common(1)[0][0]
                 for col in zip(*all_preds[:k])]
        acc = float(np.mean([v == l for v, l in zip(votes, test_labels)]))
        ensemble_results[k] = acc
        print(f"ENSEMBLE k={k:>3d}: test={acc*100:.2f}  "
              f"gain={100*(acc-base_test):+.2f}pp")
        if wandb_run:
            wandb_run.log({f"ensemble/k{k}/accuracy": acc,
                           f"ensemble/k{k}/gain_over_base": acc - base_test})

    exp_dir = args.experiment_dir or f"results/e1-randopt-{args.corruption}-s{args.severity}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump({"config": vars(args),
                   "base": {"score": base_score, "test_accuracy": base_test},
                   "cells": cell_summaries,
                   "ensemble": ensemble_results,
                   "perturbations": all_perts}, f, indent=2)
    print(f"Saved {exp_dir}/results.json")
    if wandb_run:
        wandb_run.summary["best_ensemble_gain"] = (
            max(ensemble_results.values()) - base_test)
        wandb_run.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="facebook/dinov2-giant")
    p.add_argument("--data_root", default="data/imagenet_c")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--train_per_class", type=int, default=25)
    p.add_argument("--val_per_class", type=int, default=10)
    p.add_argument("--test_per_class", type=int, default=15)
    p.add_argument("--n_score_queries", type=int, default=1000,
                   help="cap on scoring queries carved from train (cost control)")
    p.add_argument("--cells", default="all:0.0003,all:0.001,all:0.003")
    p.add_argument("--population_size", type=int, default=300)
    p.add_argument("--top_k_list", default="1,5,10,25")
    p.add_argument("--inference_batch_size", type=int, default=32)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
