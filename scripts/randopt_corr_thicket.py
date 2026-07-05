"""E1: individual-gain / thicket-existence study for SPair-71k correspondence.

Asks the fundamental question BEFORE any selection or ensembling:
  can a single perturbation beat the base model on held-out data?

For each (scope, sigma) cell we draw N perturbations and evaluate each on TWO
disjoint held-out splits A and B. Per cell we report:
  - base PCK on A and B
  - distribution of individual PCK on B (mean, p50, p90, max)
  - best-by-A perturbation's PCK on B  <-- unbiased "one expert" held-out gain
  - Spearman rho(PCK_A, PCK_B)          <-- selection-generalization signal (logged)

Ranking on A and reporting on B avoids the winner's-curse: max PCK over N is
inflated by eval noise, but the A-winner's B-score is an honest held-out number.

No selection/ensemble logic here by design.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import ray
import torch

# Ensure repo root is importable when run as scripts/randopt_corr_thicket.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handlers import get_dataset_handler
from data_handlers.spair71k import compute_pck, GRID
from vision import launch_vision_engines


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho without scipy: Pearson correlation of ranks."""
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def parse_scope(tok: str) -> Tuple[str, int, str]:
    tok = tok.strip()
    if tok == "all":
        return ("all", 0, "all")
    if tok.startswith("last"):
        n = int(tok[4:])
        return ("last_n_blocks", n, f"last{n}")
    raise ValueError(f"unknown scope token: {tok}")


def stack(datas: List[Dict], key: str) -> torch.Tensor:
    return torch.stack([d[key] for d in datas])


def eval_pck(engine, src_imgs, tgt_imgs, datas) -> float:
    """Mean PCK@0.1 over a set of pairs using current engine weights."""
    sf = ray.get(engine.get_patch_features.remote(src_imgs))
    tf = ray.get(engine.get_patch_features.remote(tgt_imgs))
    scores = [compute_pck(sf[i], tf[i], d["kpts_src"], d["kpts_tgt"],
                          bbox_thresh=d.get("bbox_thresh", 1.6))
              for i, d in enumerate(datas)]
    return float(np.mean(scores))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="spair71k")
    p.add_argument("--model_name", default="facebook/dinov2-base")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--nA", type=int, default=250, help="held-out split A size")
    p.add_argument("--nB", type=int, default=250, help="held-out split B size (disjoint)")
    p.add_argument("--population_size", type=int, default=100, help="N perturbations per cell")
    p.add_argument("--sigma_values", default="0.00003,0.0001,0.0003,0.001,0.003")
    p.add_argument("--scopes", default="all,last2,last1")
    p.add_argument("--inference_batch_size", type=int, default=64)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main(args):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_run_name or f"thicket-{args.dataset}"
        wandb_run = wandb.init(project=args.wandb_project, name=name, config=vars(args))

    handler = get_dataset_handler(args.dataset)
    data_dir = args.data_dir or handler.default_train_path
    print("Loading held-out splits A and B (disjoint)...")
    dataA = handler.load_data(data_dir, split="test", max_samples=args.nA, start_index=0)
    dataB = handler.load_data(data_dir, split="test", max_samples=args.nB, start_index=args.nA)
    print(f"  A: {len(dataA)} pairs | B: {len(dataB)} pairs")

    srcA, tgtA = stack(dataA, "image_tensor"), stack(dataA, "image_tensor_tgt")
    srcB, tgtB = stack(dataB, "image_tensor"), stack(dataB, "image_tensor_tgt")

    ray.init(ignore_reinit_error=True)
    engines = launch_vision_engines(
        num_engines=1, model_name=args.model_name, num_classes=1,
        linear_init_path=None, inference_batch_size=args.inference_batch_size,
        perturb_target="all",
    )
    engine = engines[0]

    # Base (no perturbation)
    base_A = eval_pck(engine, srcA, tgtA, dataA)
    base_B = eval_pck(engine, srcB, tgtB, dataB)
    print(f"\nBase PCK  A={base_A*100:.2f}%  B={base_B*100:.2f}%\n")
    if wandb_run:
        wandb_run.log({"base/pck_A": base_A, "base/pck_B": base_B})

    sigmas = [float(s) for s in args.sigma_values.split(",")]
    scopes = [parse_scope(t) for t in args.scopes.split(",")]
    rng = np.random.default_rng(args.global_seed)

    table = []  # cells for final summary + JSON
    step = 0
    for (target, n, label) in scopes:
        ray.get(engine.set_perturb_scope.remote(target, n))
        pcount = ray.get(engine.count_perturb_params.remote())
        print(f"=== scope={label} ({pcount:,} params) ===")
        for sigma in sigmas:
            seeds = rng.integers(0, 2**31, size=args.population_size)
            pcks_A, pcks_B = [], []
            t0 = time.time()
            for seed in seeds:
                ray.get(engine.perturb_weights.remote(int(seed), float(sigma)))
                pcks_A.append(eval_pck(engine, srcA, tgtA, dataA))
                pcks_B.append(eval_pck(engine, srcB, tgtB, dataB))
                ray.get(engine.restore_weights.remote(int(seed), float(sigma)))
            aA, aB = np.array(pcks_A), np.array(pcks_B)
            best_by_A = int(aA.argmax())
            one_expert_B = float(aB[best_by_A])          # unbiased held-out gain
            rho = spearman(aA, aB)
            cell = {
                "scope": label, "n_blocks": n, "params": int(pcount), "sigma": sigma,
                "base_B": base_B,
                "mean_B": float(aB.mean()), "p50_B": float(np.percentile(aB, 50)),
                "p90_B": float(np.percentile(aB, 90)), "max_B": float(aB.max()),
                "one_expert_B": one_expert_B,
                "one_expert_gain": one_expert_B - base_B,
                "rho_AB": rho,
            }
            table.append(cell)
            print(f"  sigma={sigma:<8} base_B={base_B*100:5.2f}  "
                  f"mean={aB.mean()*100:5.2f}  p90={np.percentile(aB,90)*100:5.2f}  "
                  f"max={aB.max()*100:5.2f}  1expert={one_expert_B*100:5.2f} "
                  f"(gain {100*(one_expert_B-base_B):+.2f}pp)  rho={rho:+.2f}  "
                  f"[{time.time()-t0:.0f}s]")
            if wandb_run:
                wandb_run.log({
                    f"thicket/{label}/s{sigma}/mean_B": aB.mean(),
                    f"thicket/{label}/s{sigma}/p90_B": float(np.percentile(aB, 90)),
                    f"thicket/{label}/s{sigma}/max_B": aB.max(),
                    f"thicket/{label}/s{sigma}/one_expert_gain": one_expert_B - base_B,
                    f"thicket/{label}/s{sigma}/rho_AB": rho,
                    "step": step,
                })
            step += 1

    # Summary table
    print("\n===== SCOPE x SIGMA : one-expert held-out gain (pp) =====")
    print(f"base_B = {base_B*100:.2f}%")
    for cell in sorted(table, key=lambda c: c["one_expert_gain"], reverse=True):
        flag = "  <-- THICKET" if cell["one_expert_gain"] >= 0.05 else ""
        print(f"  {cell['scope']:6s} sigma={cell['sigma']:<8} "
              f"gain={100*cell['one_expert_gain']:+6.2f}pp  rho={cell['rho_AB']:+.2f}{flag}")

    best = max(table, key=lambda c: c["one_expert_gain"])
    verdict = ("THICKET FOUND (hyp a)" if best["one_expert_gain"] >= 0.05
               else "no individual gain -> reachable ceiling ~ base (hyp b)")
    print(f"\nVerdict: {verdict} | best cell: scope={best['scope']} "
          f"sigma={best['sigma']} gain={100*best['one_expert_gain']:+.2f}pp")

    exp_dir = args.experiment_dir or f"results/corr_thicket_{args.dataset}"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump({"base_A": base_A, "base_B": base_B, "cells": table,
                   "verdict": verdict}, f, indent=2)
    print(f"\nSaved {exp_dir}/results.json")
    if wandb_run:
        wandb_run.summary["best_one_expert_gain"] = best["one_expert_gain"]
        wandb_run.summary["verdict"] = verdict
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
