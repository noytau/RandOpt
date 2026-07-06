"""E4: does task-adaptation move DINOv2 from a needle regime into a thicket?

E1 found no thicket for correspondence around the SSL init (random search).
E2 showed a needle: gradient reaches a +10.5pp model inside the searched ball,
so better models EXIST but occupy negligible measure. Open question (user's):
maybe a *thicket* exists too — not at the init, but around the task-ADAPTED
point that only directed movement reaches. The paper's core claim is that
task-adapted models (instruct-LLMs) sit in thickets while raw pretrained ones
may not; a few gradient steps make DINOv2 a little task-adapted.

This experiment runs the E1 thicket protocol (random isotropic perturbation,
rank-on-A / report-on-B, Spearman rho) at several points ALONG the gradient
trajectory — 0, 50, 150, 300 steps of adaptation — and asks whether thicket
density RISES with adaptation. Crucially, at each center the one-expert gain is
measured relative to THAT center's own held-out PCK (local density), not the
original init.

  gain stays ~0 at every level      -> needle all the way; adaptation doesn't
                                       create a reachable-by-random thicket.
  gain rises with adaptation steps  -> THICKET EMERGES with task-adaptation.
                                       Explains the negative result mechanistically
                                       and confirms the paper's claim on SSL.
  gain high only at 0 steps         -> (not expected) init special; adaptation
                                       destroys structure.

Design notes:
  - Reuses E2's standalone eval plumbing (parity-checked to base_B=58.37), NOT
    Ray, so the gradient trajectory is bit-identical to grad-reach-last1
    (softargmax, lr=3e-5, same seed/batch order).
  - Trajectory is walked INCREMENTALLY (base->50->150->300); at each checkpoint
    the scope weights are snapshotted, the thicket sweep perturbs around that
    snapshot (exact seeded add/subtract restore), then gradient resumes.
  - Perturbation scheme replicates VisionEngine exactly (per-param Generator
    seeded with `seed`), so thicket density here is comparable to E1's.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handlers import get_dataset_handler
from data_handlers.spair71k import GRID
from scripts.grad_reachability import (
    NORM_MEAN, NORM_STD, softargmax_loss, patch_features, scope_param_names,
    eval_pck_full, delta_norm, stack, make_eval_inputs,
)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def perturb(scope_params, seed: int, sigma: float, sign: float) -> None:
    """In-place seeded Gaussian, replicating VisionEngine.perturb_weights.

    sign=+1 perturb, sign=-1 restore (exact inverse from the same seed).
    """
    for p in scope_params:
        gen = torch.Generator(device=p.device); gen.manual_seed(seed)
        noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
        p.data.add_(sign * sigma * noise)


def thicket_sweep(backbone, scope_params, scope_names, base_center, sigmas, npop,
                  srcA, tgtA, evA, srcB, tgtB, evB, mean, std, batch, rng):
    """E1 protocol around the CURRENT weights (== base_center snapshot).

    Returns center_B and a list of per-sigma cells with one-expert gain + rho.
    """
    center_B = eval_pck_full(backbone, srcB, tgtB, evB, mean, std, batch)
    center_A = eval_pck_full(backbone, srcA, tgtA, evA, mean, std, batch)
    cells = []
    for sigma in sigmas:
        seeds = rng.integers(0, 2**31, size=npop)
        pcks_A, pcks_B = [], []
        for seed in seeds:
            perturb(scope_params, int(seed), float(sigma), +1.0)
            pcks_A.append(eval_pck_full(backbone, srcA, tgtA, evA, mean, std, batch))
            pcks_B.append(eval_pck_full(backbone, srcB, tgtB, evB, mean, std, batch))
            perturb(scope_params, int(seed), float(sigma), -1.0)  # exact restore
        # guard against any fp drift: hard-restore the snapshot
        with torch.no_grad():
            params = dict(backbone.named_parameters())
            for n in scope_names:
                params[n].data.copy_(base_center[n])
        aA, aB = np.array(pcks_A), np.array(pcks_B)
        best_by_A = int(aA.argmax())
        one_expert_B = float(aB[best_by_A])
        cells.append({
            "sigma": sigma, "center_B": center_B,
            "mean_B": float(aB.mean()), "p90_B": float(np.percentile(aB, 90)),
            "max_B": float(aB.max()), "one_expert_B": one_expert_B,
            "one_expert_gain": one_expert_B - center_B,
            "rho_AB": spearman(aA, aB),
        })
        print(f"    sigma={sigma:<8} center_B={center_B*100:5.2f}  "
              f"mean {100*(aB.mean()-center_B):+5.2f}  "
              f"1expert {100*(one_expert_B-center_B):+5.2f}pp  "
              f"rho={spearman(aA, aB):+.2f}", flush=True)
    return center_A, center_B, cells


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="facebook/dinov2-base")
    p.add_argument("--scope", default="last1")
    p.add_argument("--levels", default="0,50,150,300", help="adaptation step checkpoints")
    p.add_argument("--sigmas", default="0.0003,0.001,0.003,0.01")
    p.add_argument("--npop", type=int, default=300, help="perturbations per (level,sigma)")
    p.add_argument("--lr", type=float, default=3e-5, help="E2 winner")
    p.add_argument("--tau", type=float, default=0.05)
    p.add_argument("--pairs_per_step", type=int, default=8)
    p.add_argument("--n_train", type=int, default=800)
    p.add_argument("--nA", type=int, default=200)
    p.add_argument("--nB", type=int, default=200)
    p.add_argument("--inference_batch_size", type=int, default=64)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.global_seed)

    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project,
                               name=args.wandb_run_name or "recenter-thicket",
                               config=vars(args))

    from transformers import Dinov2Model
    backbone = Dinov2Model.from_pretrained(args.model_name).to(device)
    backbone.eval()
    mean = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    names = scope_param_names(backbone, args.scope)
    name_set = set(names)
    for n, p in backbone.named_parameters():
        p.requires_grad_(n in name_set)
    scope_params = [p for n, p in backbone.named_parameters() if n in name_set]
    d_scope = sum(p.numel() for p in scope_params)
    print(f"scope={args.scope}: {d_scope:,} params "
          f"(sigma=1e-3 ball radius ~= {1e-3 * d_scope**0.5:.2f})")

    handler = get_dataset_handler("spair71k")
    dd = handler.default_train_path
    dataT = handler.load_data(dd, split="train", max_samples=args.n_train)
    dataA = handler.load_data(dd, split="test", max_samples=args.nA, start_index=0)
    dataB = handler.load_data(dd, split="test", max_samples=args.nB, start_index=args.nA)
    srcA, tgtA = stack(dataA, "image_tensor"), stack(dataA, "image_tensor_tgt")
    srcB, tgtB = stack(dataB, "image_tensor"), stack(dataB, "image_tensor_tgt")
    evA, evB = make_eval_inputs(dataA), make_eval_inputs(dataB)

    idx = torch.arange(GRID * GRID, device=device)
    coords = torch.stack([idx // GRID, idx % GRID], dim=-1).float()

    base_B = eval_pck_full(backbone, srcB, tgtB, evB, mean, std, args.inference_batch_size)
    print(f"\nBase PCK B={base_B*100:.2f}%  (sweep=58.37 — parity check)\n")

    levels = sorted(int(x) for x in args.levels.split(","))
    sigmas = sorted([float(s) for s in args.sigmas.split(",")], reverse=True)
    opt = torch.optim.AdamW(scope_params, lr=args.lr, weight_decay=0.0)
    rng_data = np.random.default_rng(args.global_seed)
    order = rng_data.permutation(len(dataT)); pos = 0

    def train_steps(k):
        nonlocal pos, order
        for _ in range(k):
            if pos + args.pairs_per_step > len(order):
                order = rng_data.permutation(len(dataT)); pos = 0
            batch = [dataT[i] for i in order[pos:pos + args.pairs_per_step]]
            pos += args.pairs_per_step
            imgs = torch.cat([stack(batch, "image_tensor"),
                              stack(batch, "image_tensor_tgt")])
            feats = patch_features(backbone, imgs, mean, std, len(imgs), grad=True)
            fs, ft = feats[:len(batch)], feats[len(batch):]
            loss = 0.0
            for i, d in enumerate(batch):
                src_idx = [r * GRID + c for r, c in d["kpts_src"]]
                sim = fs[i][src_idx] @ ft[i].T
                loss = loss + softargmax_loss(sim, d["kpts_tgt"], d["bbox_thresh"],
                                              args.tau, coords)
            loss = loss / len(batch)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()

    orig_base = {n: dict(backbone.named_parameters())[n].data.clone() for n in names}
    results = []
    prev = 0
    rng_pert = np.random.default_rng(args.global_seed + 7)
    for lvl in levels:
        if lvl > prev:
            train_steps(lvl - prev); prev = lvl
        backbone.eval()
        dw = delta_norm(backbone, orig_base, names)  # how far this center is from the init
        # snapshot this adapted center for exact restore during the sweep
        center = {n: dict(backbone.named_parameters())[n].data.clone() for n in names}
        print(f"=== adaptation level: {lvl} steps  (||dw||={dw:.2f} from init) ===")
        cA, cB, cells = thicket_sweep(
            backbone, scope_params, names, center, sigmas, args.npop,
            srcA, tgtA, evA, srcB, tgtB, evB, mean, std,
            args.inference_batch_size, rng_pert)
        best = max(cells, key=lambda c: c["one_expert_gain"])
        results.append({"level": lvl, "dw_from_init": dw, "center_A": cA, "center_B": cB,
                        "best_one_expert_gain": best["one_expert_gain"],
                        "best_sigma": best["sigma"], "cells": cells})
        print(f"  -> center_B={cB*100:.2f}%  best thicket gain={100*best['one_expert_gain']:+.2f}pp "
              f"(sigma={best['sigma']}, rho={best['rho_AB']:+.2f})\n")
        if wandb_run:
            wandb_run.log({"level": lvl, "center_B": cB,
                           "best_thicket_gain": best["one_expert_gain"]})

    print("===== THICKET-EMERGENCE CURVE =====")
    print(f"{'level':>6} {'center_B%':>10} {'best_gain_pp':>13} {'best_sigma':>11}")
    for r in results:
        print(f"{r['level']:>6} {r['center_B']*100:>10.2f} "
              f"{100*r['best_one_expert_gain']:>+13.2f} {r['best_sigma']:>11}")
    gains = [100 * r["best_one_expert_gain"] for r in results]
    rising = len(gains) >= 2 and gains[-1] - gains[0] >= 2.0 and gains[-1] >= 2.0
    verdict = ("THICKET EMERGES with adaptation" if rising
               else "NO emergent thicket — needle persists across adaptation")
    print(f"\nVerdict: {verdict}")

    exp_dir = args.experiment_dir or "results/recenter_thicket"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump({"base_B": base_B, "scope": args.scope, "levels": results,
                   "verdict": verdict}, f, indent=2)
    print(f"Saved {exp_dir}/results.json")
    if wandb_run:
        wandb_run.summary["verdict"] = verdict
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
