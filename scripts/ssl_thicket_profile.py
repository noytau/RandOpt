"""E3: multi-task thicket PROFILE of the DINOv2 SSL neighborhood.

The PCK sweep measured one axis and found a needle (E2 proved better models
exist nearby but have negligible measure). This experiment asks the broader
question: is the SSL neighborhood a thicket along ANY functional axis, and
which ones?

Hypothesis (H1): thickets exist along the axes the init was ADAPTED for.
DINOv2's pretext is discriminative (instance/patch self-distillation), so its
neighborhood should be dense with experts for global-discrimination readouts
(kNN classification, retrieval mAP) and needle-like for spatial correspondence
(measured: nothing). Decision rules are fixed in advance:

  experts on global tasks, none on PCK  -> H1 supported (adaptation-local thickets)
  no experts on any axis                -> no thicket at all; run per-task gradient
                                           controls to classify plateau vs needle
  rho(A,B) per task                     -> high+positive = real thicket;
                                           high+negative = brittle axis; ~0 = noise

Design:
  - Each perturbation is scored on the full task panel from shared forwards:
      pck      SPair-71k correspondence (patch tokens)          [spatial]
      cub_knn  CUB-200 kNN top-1        (CLS tokens)            [global]
      cub_map  CUB-200 retrieval mAP    (same sims, free)       [global, smooth]
      air_knn  FGVC-Aircraft kNN top-1                          [global]
      air_map  FGVC-Aircraft retrieval mAP                      [global, smooth]
  - Same honesty protocol as the sweep, PER TASK: two disjoint held-out query
    splits A/B; rank on A, report the winner's B score (unbiased one-expert
    gain); Spearman rho(A,B) as the selection-generalization signal.
  - NEW readout: cross-task Spearman of B-gains across perturbations — does a
    draw that helps CUB hurt PCK? Maps whether the axes trade off or are
    independent.
  - Cells are "scope:sigma" tokens (not a full cross product) because the
    sigma->cliff differs per scope; include a small-sigma cell per scope as the
    noise yardstick, and the panel re-measures the cliff per task via mean_B.
  - A dataset that fails to load (cluster download flakiness) is skipped with a
    warning; the rest of the panel proceeds.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handlers import get_dataset_handler
from vision import launch_vision_engines


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def parse_cell(tok: str) -> Tuple[str, int, str, float]:
    """'last1:0.003' -> (target, n_blocks, label, sigma)"""
    scope, sig = tok.strip().split(":")
    if scope == "all":
        return ("all", 0, "all", float(sig))
    if scope.startswith("last"):
        n = int(scope[4:])
        return ("last_n_blocks", n, scope, float(sig))
    raise ValueError(f"unknown scope in cell token: {tok}")


def to_448(t: torch.Tensor) -> torch.Tensor:
    """Resize a (3,H,W) tensor to (3,448,448) if needed (CUB loads native size)."""
    if t.shape[-2:] == (448, 448):
        return t
    return F.interpolate(t.unsqueeze(0), size=(448, 448), mode="bicubic",
                         align_corners=False).squeeze(0).clamp(0, 1)


def stack448(datas: List[Dict]) -> torch.Tensor:
    return torch.stack([to_448(d["image_tensor"]) for d in datas])


# ----------------------------------------------------------------------
# Task panel loaders
# ----------------------------------------------------------------------

def load_spair(nA: int, nB: int):
    handler = get_dataset_handler("spair71k")
    dataA = handler.load_data(handler.default_train_path, split="test",
                              max_samples=nA, start_index=0)
    dataB = handler.load_data(handler.default_train_path, split="test",
                              max_samples=nB, start_index=nA)
    def pack(datas):
        return {
            "src": torch.stack([d["image_tensor"] for d in datas]),
            "tgt": torch.stack([d["image_tensor_tgt"] for d in datas]),
            "kpts_src": [d["kpts_src"] for d in datas],
            "kpts_tgt": [d["kpts_tgt"] for d in datas],
            "bbox": [d.get("bbox_thresh", 1.6) for d in datas],
        }
    return pack(dataA), pack(dataB)


def load_classification(name: str, gallery_per_class: int, nqA: int, nqB: int,
                        seed: int):
    """Balanced gallery from train split; shuffled disjoint A/B queries from test.

    CUB/Aircraft test items are ordered by class, so the queries MUST be
    shuffled (seeded) before splitting or A would cover only a few classes.
    """
    handler = get_dataset_handler(name)
    train = handler.load_data(handler.default_train_path, split="train")
    test = handler.load_data(handler.default_test_path, split="test")

    by_class: Dict[int, List[Dict]] = {}
    for d in train:
        by_class.setdefault(d["class_id"], []).append(d)
    gallery = []
    for cid in sorted(by_class):
        gallery.extend(by_class[cid][:gallery_per_class])

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(test))
    qA = [test[i] for i in order[:nqA]]
    qB = [test[i] for i in order[nqA:nqA + nqB]]

    return {
        "g_imgs": stack448(gallery),
        "g_labels": [d["class_id"] for d in gallery],
        "qA_imgs": stack448(qA), "qA_labels": [d["class_id"] for d in qA],
        "qB_imgs": stack448(qB), "qB_labels": [d["class_id"] for d in qB],
    }


# ----------------------------------------------------------------------
# Panel evaluation: all task scores for the CURRENT engine weights
# ----------------------------------------------------------------------

def eval_panel(engine, spairA, spairB, cls_sets: Dict[str, dict]) -> Dict[str, float]:
    """Returns {task_split: score}, e.g. {'pck_A':…, 'cub_knn_B':…}."""
    out = {}
    for split, s in (("A", spairA), ("B", spairB)):
        out[f"pck_{split}"] = ray.get(engine.eval_pck.remote(
            s["src"], s["tgt"], s["kpts_src"], s["kpts_tgt"], s["bbox"]))
    for name, c in cls_sets.items():
        resA, resB = ray.get(engine.eval_global.remote(
            c["g_imgs"], c["g_labels"],
            [(c["qA_imgs"], c["qA_labels"]), (c["qB_imgs"], c["qB_labels"])]))
        out[f"{name}_knn_A"] = resA["knn_top1"]; out[f"{name}_map_A"] = resA["map"]
        out[f"{name}_knn_B"] = resB["knn_top1"]; out[f"{name}_map_B"] = resB["map"]
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="facebook/dinov2-base")
    p.add_argument("--cells", default=("last1:0.0003,last1:0.001,last1:0.003,last1:0.01,"
                                       "last2:0.0003,last2:0.001,"
                                       "all:0.0001,all:0.0003,all:0.001"))
    p.add_argument("--tasks", default="pck,cub,air",
                   help="subset of pck,cub,air (a failed download drops its task)")
    p.add_argument("--population_size", type=int, default=300)
    p.add_argument("--sp_nA", type=int, default=200)
    p.add_argument("--sp_nB", type=int, default=200)
    p.add_argument("--gallery_per_class", type=int, default=3,
                   help="CUB uses this; Aircraft uses 2x (100 classes vs 200)")
    p.add_argument("--n_queries", type=int, default=500, help="per split A/B")
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
        name = args.wandb_run_name or "ssl-thicket-profile"
        wandb_run = wandb.init(project=args.wandb_project, name=name, config=vars(args))

    want = [t.strip() for t in args.tasks.split(",")]

    print("Loading task panel...")
    spairA = spairB = None
    if "pck" in want:
        spairA, spairB = load_spair(args.sp_nA, args.sp_nB)
        print(f"  spair71k: A={len(spairA['bbox'])} B={len(spairB['bbox'])} pairs")

    cls_sets: Dict[str, dict] = {}
    for name, ds, gpc in (("cub", "cub200", args.gallery_per_class),
                          ("air", "fgvc_aircraft", args.gallery_per_class * 2)):
        if name not in want:
            continue
        try:
            cls_sets[name] = load_classification(ds, gpc, args.n_queries,
                                                 args.n_queries, args.global_seed)
            c = cls_sets[name]
            print(f"  {ds}: gallery={len(c['g_labels'])} qA={len(c['qA_labels'])} "
                  f"qB={len(c['qB_labels'])}")
        except Exception as e:  # cluster download flakiness must not kill the run
            print(f"  WARNING: {ds} failed to load ({e}) — dropping task '{name}'")
    if spairA is None and not cls_sets:
        print("No tasks loaded — aborting."); sys.exit(1)

    ray.init(ignore_reinit_error=True)
    engine = launch_vision_engines(
        num_engines=1, model_name=args.model_name, num_classes=1,
        linear_init_path=None, inference_batch_size=args.inference_batch_size,
        perturb_target="all")[0]

    base = eval_panel(engine, spairA, spairB, cls_sets)
    tasks = sorted({k[:-2] for k in base})  # 'pck', 'cub_knn', …
    print("\nBase panel:")
    for t in tasks:
        print(f"  {t:>8s}: A={base[t+'_A']*100:5.2f}  B={base[t+'_B']*100:5.2f}")
    if wandb_run:
        wandb_run.log({f"base/{k}": v for k, v in base.items()})

    cells = [parse_cell(tok) for tok in args.cells.split(",")]
    rng = np.random.default_rng(args.global_seed)
    results = []
    for (target, nb, label, sigma) in cells:
        ray.get(engine.set_perturb_scope.remote(target, nb))
        pcount = ray.get(engine.count_perturb_params.remote())
        seeds = rng.integers(0, 2**31, size=args.population_size)
        scores = {k: [] for t in tasks for k in (f"{t}_A", f"{t}_B")}
        t0 = time.time()
        print(f"\n=== cell {label} sigma={sigma} ({pcount:,} params, "
              f"N={args.population_size}) ===")
        for j, seed in enumerate(seeds):
            ray.get(engine.perturb_weights.remote(int(seed), float(sigma)))
            panel = eval_panel(engine, spairA, spairB, cls_sets)
            ray.get(engine.restore_weights.remote(int(seed), float(sigma)))
            for k, v in panel.items():
                scores[k].append(v)
            if (j + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {j+1}/{len(seeds)}  [{el:.0f}s, {el/(j+1):.1f}s/pert]",
                      flush=True)

        cell = {"cell": f"{label}:{sigma}", "scope": label, "sigma": sigma,
                "params": int(pcount), "n": len(seeds), "tasks": {}}
        gains_B = {}
        for t in tasks:
            aA = np.array(scores[f"{t}_A"]); aB = np.array(scores[f"{t}_B"])
            bB = base[f"{t}_B"]
            best_by_A = int(aA.argmax())
            one_expert = float(aB[best_by_A])
            cell["tasks"][t] = {
                "base_B": bB, "mean_B": float(aB.mean()),
                "p90_B": float(np.percentile(aB, 90)), "max_B": float(aB.max()),
                "one_expert_B": one_expert, "one_expert_gain": one_expert - bB,
                "rho_AB": spearman(aA, aB),
            }
            gains_B[t] = aB - bB
            r = cell["tasks"][t]
            print(f"  {t:>8s}: mean {100*(r['mean_B']-bB):+6.2f}  "
                  f"1expert {100*r['one_expert_gain']:+6.2f}pp  "
                  f"rho={r['rho_AB']:+.2f}")
            if wandb_run:
                wandb_run.log({f"profile/{label}/s{sigma}/{t}/one_expert_gain":
                               r["one_expert_gain"],
                               f"profile/{label}/s{sigma}/{t}/rho_AB": r["rho_AB"],
                               f"profile/{label}/s{sigma}/{t}/mean_shift":
                               r["mean_B"] - bB})
        # cross-task structure of the SAME perturbations (B-gains)
        cell["cross_task_rho"] = {
            f"{t1}~{t2}": spearman(gains_B[t1], gains_B[t2])
            for i, t1 in enumerate(tasks) for t2 in tasks[i+1:]}
        results.append(cell)

    # -------- profile matrix --------
    print("\n===== THICKET PROFILE : one-expert held-out gain (pp) =====")
    header = "cell".ljust(16) + "".join(t.rjust(11) for t in tasks)
    print(header)
    for cell in results:
        row = cell["cell"].ljust(16)
        for t in tasks:
            row += f"{100*cell['tasks'][t]['one_expert_gain']:+10.2f} "
        print(row)
    print("\n(rho_AB per cell/task and cross-task correlations in results.json)")

    exp_dir = args.experiment_dir or "results/ssl_thicket_profile"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump({"base": base, "cells": results}, f, indent=2)
    print(f"Saved {exp_dir}/results.json")
    if wandb_run:
        best = {t: max(c["tasks"][t]["one_expert_gain"] for c in results)
                for t in tasks}
        for t, g in best.items():
            wandb_run.summary[f"best_gain/{t}"] = g
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
