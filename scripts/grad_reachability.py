"""E2: gradient-reachability control for the SPair-71k thicket result.

The thicket sweep (scripts/randopt_corr_thicket.py) found NO expert among ~36k
random isotropic perturbations. That is compatible with two opposite worlds:

  PLATEAU — no better correspondence model exists near the SSL init at all.
  NEEDLE  — better models exist nearby but occupy negligible measure, so
            isotropic Gaussian sampling essentially never lands on them.

This script distinguishes them: replace random search with gradient ascent on a
differentiable PCK surrogate, under otherwise IDENTICAL conditions — same frozen
setup (no head), same scope filter (encoder.layer.{i}, matching VisionEngine
_perturb_params), same eval (bicubic 448->224, ImageNet norm, CLS dropped,
L2-normalized patch tokens, compute_pck with per-pair bbox threshold), same
held-out A/B protocol (train on trn split, rank checkpoints on A, report B).

  gradient also fails  -> no ascent direction -> TRUE PLATEAU (strong negative)
  gradient improves    -> better models exist; random search can't hit them
                          -> NEEDLE (thickets need measure, not just existence)

||delta_w|| is logged at every eval and converted to "equivalent sigma"
(= ||dw|| / sqrt(d_in_scope)) so every point on the curve can be placed inside
or outside the sweep's searched radius.

Losses (both are standard in the correspondence literature):
  softargmax — temperature softmax over the similarity row -> expected patch
               coordinate -> distance to GT keypoint, normalized by the pair's
               PCK bbox threshold. Distance-aware ("close counts"), directly
               aligned with PCK@0.1.
  infonce    — cross-entropy of sim/tau against the GT target patch index.
               Stricter (exact-patch) but well-conditioned.
"""
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handlers import get_dataset_handler
from data_handlers.spair71k import compute_pck, GRID

# Same preprocessing constants as vision/engine.py (VisionEngine)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
INPUT_SIZE = 224


# ----------------------------------------------------------------------
# Losses (module-level, unit-testable without a model)
# ----------------------------------------------------------------------

def softargmax_loss(sim: torch.Tensor, kpts_tgt: List[Tuple[int, int]],
                    bbox_thresh: float, tau: float,
                    coords: torch.Tensor) -> torch.Tensor:
    """Soft-argmax PCK surrogate for one pair.

    Args:
        sim: (K, P) cosine similarity rows for the K source keypoints
        kpts_tgt: K ground-truth (row, col) target keypoints
        bbox_thresh: this pair's PCK threshold in patch units
        tau: softmax temperature
        coords: (P, 2) float patch-center coordinates [(row, col), ...]
    Returns:
        scalar — mean GT distance of the expected coordinate, in units of the
        pair's PCK threshold (loss < 1 ~ "would count as correct").
    """
    w = F.softmax(sim / tau, dim=-1)                        # (K, P)
    pred = w @ coords                                       # (K, 2)
    tgt = torch.tensor(kpts_tgt, dtype=pred.dtype, device=pred.device)
    dist = torch.linalg.vector_norm(pred - tgt, dim=-1)     # (K,)
    return (dist / max(bbox_thresh, 1e-6)).mean()


def infonce_loss(sim: torch.Tensor, kpts_tgt: List[Tuple[int, int]],
                 tau: float) -> torch.Tensor:
    """Contrastive correspondence loss for one pair: CE(sim/tau, GT patch idx)."""
    target = torch.tensor([r * GRID + c for r, c in kpts_tgt],
                          dtype=torch.long, device=sim.device)
    return F.cross_entropy(sim / tau, target)


# ----------------------------------------------------------------------
# Model plumbing — mirrors VisionEngine exactly
# ----------------------------------------------------------------------

def preprocess(images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    x = F.interpolate(images, size=(INPUT_SIZE, INPUT_SIZE),
                      mode="bicubic", align_corners=False)
    return (x - mean) / std


def patch_features(backbone, images: torch.Tensor, mean, std,
                   batch_size: int, grad: bool) -> torch.Tensor:
    """(N,3,H,W) [0,1] -> (N, P, D) L2-normalized patch tokens (CLS dropped)."""
    device = mean.device
    feats = []
    for i in range(0, len(images), batch_size):
        batch = preprocess(images[i:i + batch_size].to(device), mean, std)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            tokens = backbone(pixel_values=batch).last_hidden_state[:, 1:, :]
            tokens = F.normalize(tokens, dim=-1)
        feats.append(tokens if grad else tokens.cpu())
    return torch.cat(feats, dim=0)


def scope_param_names(backbone, scope: str) -> List[str]:
    """Same selection logic as VisionEngine._perturb_params (no head here)."""
    if scope == "all":
        return [n for n, _ in backbone.named_parameters()]
    assert scope.startswith("last"), f"unknown scope: {scope}"
    n_blocks = int(scope[4:])
    num_layers = backbone.config.num_hidden_layers
    keep = set(range(num_layers - n_blocks, num_layers))
    names = []
    for name, _ in backbone.named_parameters():
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layer" \
                and int(parts[2]) in keep:
            names.append(name)
    return names


def eval_pck_full(backbone, src, tgt, ev, mean, std, batch_size) -> float:
    """Identical to VisionEngine.eval_pck: features -> CPU -> compute_pck."""
    kpts_src, kpts_tgt, bbox = ev
    backbone.eval()
    sf = patch_features(backbone, src, mean, std, batch_size, grad=False)
    tf = patch_features(backbone, tgt, mean, std, batch_size, grad=False)
    scores = [compute_pck(sf[i], tf[i], kpts_src[i], kpts_tgt[i], bbox_thresh=bbox[i])
              for i in range(len(kpts_src))]
    return float(sum(scores) / len(scores)) if scores else 0.0


def delta_norm(backbone, base: Dict[str, torch.Tensor], names: List[str]) -> float:
    params = dict(backbone.named_parameters())
    sq = sum(float(((params[n].data - base[n]) ** 2).sum()) for n in names)
    return sq ** 0.5


# ----------------------------------------------------------------------
# Data helpers (same shapes as randopt_corr_thicket.py)
# ----------------------------------------------------------------------

def stack(datas: List[Dict], key: str) -> torch.Tensor:
    return torch.stack([d[key] for d in datas])


def make_eval_inputs(datas):
    return ([d["kpts_src"] for d in datas],
            [d["kpts_tgt"] for d in datas],
            [d.get("bbox_thresh", 1.6) for d in datas])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="spair71k")
    p.add_argument("--model_name", default="facebook/dinov2-base")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--scope", default="last1", help="all | lastN (matches sweep scopes)")
    p.add_argument("--n_train", type=int, default=800, help="trn-split pairs for the loss")
    p.add_argument("--nA", type=int, default=250, help="held-out split A (checkpoint ranking)")
    p.add_argument("--nB", type=int, default=250, help="held-out split B (reported)")
    p.add_argument("--losses", default="softargmax,infonce")
    p.add_argument("--lrs", default="3e-6,3e-5,3e-4")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--eval_every", type=int, default=25)
    p.add_argument("--pairs_per_step", type=int, default=8)
    p.add_argument("--tau", type=float, default=0.05, help="softmax temperature")
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
        name = args.wandb_run_name or f"grad-reach-{args.scope}"
        wandb_run = wandb.init(project=args.wandb_project, name=name, config=vars(args))

    from transformers import Dinov2Model
    backbone = Dinov2Model.from_pretrained(args.model_name).to(device)
    backbone.eval()  # no dropout/BN in DINOv2; keep eval semantics like the sweep
    mean = torch.tensor(NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=device).view(1, 3, 1, 1)

    # Scope: freeze everything, unfreeze the sweep-matching param set
    names = scope_param_names(backbone, args.scope)
    name_set = set(names)
    for n, p in backbone.named_parameters():
        p.requires_grad_(n in name_set)
    trainable = [p for n, p in backbone.named_parameters() if n in name_set]
    d_scope = sum(p.numel() for p in trainable)
    print(f"scope={args.scope}: {d_scope:,} trainable params "
          f"(sweep sigma=1e-3 ball radius ~= {1e-3 * d_scope ** 0.5:.2f})")

    # Base snapshot for exact reset between runs + ||delta_w|| tracking
    base = {n: p.data.clone() for n, p in backbone.named_parameters() if n in name_set}

    handler = get_dataset_handler(args.dataset)
    data_dir = args.data_dir or handler.default_train_path
    print("Loading data: trn (loss) + test A/B (held-out, same as sweep)...")
    dataT = handler.load_data(data_dir, split="train", max_samples=args.n_train)
    dataA = handler.load_data(data_dir, split="test", max_samples=args.nA, start_index=0)
    dataB = handler.load_data(data_dir, split="test", max_samples=args.nB, start_index=args.nA)
    print(f"  train: {len(dataT)} | A: {len(dataA)} | B: {len(dataB)}")

    srcA, tgtA = stack(dataA, "image_tensor"), stack(dataA, "image_tensor_tgt")
    srcB, tgtB = stack(dataB, "image_tensor"), stack(dataB, "image_tensor_tgt")
    evA, evB = make_eval_inputs(dataA), make_eval_inputs(dataB)

    # Patch-center coordinate grid for the soft-argmax loss
    idx = torch.arange(GRID * GRID, device=device)
    coords = torch.stack([idx // GRID, idx % GRID], dim=-1).float()  # (P, 2)

    # Parity check vs the sweep: must reproduce base_B ~= 58.37% (report.html §7)
    base_A = eval_pck_full(backbone, srcA, tgtA, evA, mean, std, args.inference_batch_size)
    base_B = eval_pck_full(backbone, srcB, tgtB, evB, mean, std, args.inference_batch_size)
    print(f"\nBase PCK  A={base_A*100:.2f}%  B={base_B*100:.2f}%  "
          f"(sweep reported base_B=58.37% — should match)\n")
    if wandb_run:
        wandb_run.log({"base/pck_A": base_A, "base/pck_B": base_B})

    losses = [t.strip() for t in args.losses.split(",")]
    lrs = [float(t) for t in args.lrs.split(",")]
    rng = np.random.default_rng(args.global_seed)

    runs = []
    for loss_name in losses:
        for lr in lrs:
            tag = f"{loss_name}/lr{lr:g}"
            # Exact reset to the SSL init
            with torch.no_grad():
                for n, p in backbone.named_parameters():
                    if n in name_set:
                        p.data.copy_(base[n])
            opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.0)

            curve = [{"step": 0, "pck_A": base_A, "pck_B": base_B,
                      "dw": 0.0, "sigma_equiv": 0.0, "train_loss": None}]
            order = rng.permutation(len(dataT))
            pos, t0 = 0, time.time()
            print(f"--- {tag} ---")
            for step in range(1, args.steps + 1):
                if pos + args.pairs_per_step > len(order):
                    order = rng.permutation(len(dataT))
                    pos = 0
                batch_ids = order[pos:pos + args.pairs_per_step]
                pos += args.pairs_per_step
                batch = [dataT[i] for i in batch_ids]

                imgs = torch.cat([stack(batch, "image_tensor"),
                                  stack(batch, "image_tensor_tgt")])
                feats = patch_features(backbone, imgs, mean, std,
                                       len(imgs), grad=True)
                fs, ft = feats[:len(batch)], feats[len(batch):]

                loss = 0.0
                for i, d in enumerate(batch):
                    src_idx = [r * GRID + c for r, c in d["kpts_src"]]
                    sim = fs[i][src_idx] @ ft[i].T  # (K, P)
                    if loss_name == "softargmax":
                        loss = loss + softargmax_loss(sim, d["kpts_tgt"],
                                                      d["bbox_thresh"], args.tau, coords)
                    elif loss_name == "infonce":
                        loss = loss + infonce_loss(sim, d["kpts_tgt"], args.tau)
                    else:
                        raise ValueError(f"unknown loss: {loss_name}")
                loss = loss / len(batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

                if step % args.eval_every == 0 or step == args.steps:
                    pA = eval_pck_full(backbone, srcA, tgtA, evA, mean, std,
                                       args.inference_batch_size)
                    pB = eval_pck_full(backbone, srcB, tgtB, evB, mean, std,
                                       args.inference_batch_size)
                    dw = delta_norm(backbone, base, names)
                    s_eq = dw / d_scope ** 0.5
                    curve.append({"step": step, "pck_A": pA, "pck_B": pB,
                                  "dw": dw, "sigma_equiv": s_eq,
                                  "train_loss": float(loss)})
                    print(f"  step={step:4d} loss={float(loss):.4f}  "
                          f"A={pA*100:5.2f}  B={pB*100:5.2f} "
                          f"(gain {100*(pB-base_B):+.2f}pp)  "
                          f"|dw|={dw:.2f} (sigma_eq={s_eq:.1e})  "
                          f"[{time.time()-t0:.0f}s]")
                    if wandb_run:
                        wandb_run.log({f"grad/{tag}/loss": float(loss),
                                       f"grad/{tag}/pck_A": pA,
                                       f"grad/{tag}/pck_B": pB,
                                       f"grad/{tag}/dw": dw, "step": step})

            # Honest checkpoint selection: rank on A, report B (mirrors the sweep)
            best = max(curve[1:], key=lambda c: c["pck_A"])
            runs.append({"loss": loss_name, "lr": lr, "curve": curve,
                         "best_by_A_step": best["step"],
                         "best_by_A_pck_B": best["pck_B"],
                         "gain_B": best["pck_B"] - base_B,
                         "sigma_equiv_at_best": best["sigma_equiv"]})
            print(f"  => best-by-A step={best['step']}  "
                  f"B={best['pck_B']*100:.2f}  "
                  f"gain={100*(best['pck_B']-base_B):+.2f}pp  "
                  f"sigma_eq={best['sigma_equiv']:.1e}\n")

    # Verdict: gain thresholds relative to the sweep's ~0.8pp noise floor
    print("===== GRADIENT REACHABILITY : held-out gain on B (pp) =====")
    print(f"base_B = {base_B*100:.2f}%  |  sweep best (random, 36k draws) = +0.34pp")
    for r in sorted(runs, key=lambda r: r["gain_B"], reverse=True):
        print(f"  {r['loss']:>10s} lr={r['lr']:<8g} gain={100*r['gain_B']:+6.2f}pp  "
              f"sigma_eq={r['sigma_equiv_at_best']:.1e}  (step {r['best_by_A_step']})")

    best = max(runs, key=lambda r: r["gain_B"])
    g = best["gain_B"] * 100
    if g >= 2.0:
        verdict = ("NEEDLE — gradient finds better models nearby that isotropic "
                   "random search could not (thickets need measure, not existence)")
    elif g <= 0.8:
        verdict = ("PLATEAU — even gradient ascent cannot improve the SSL init on "
                   "this task without added capacity (strong negative)")
    else:
        verdict = "MARGINAL — above sweep best but within/near noise; needs more steps or LRs"
    print(f"\nVerdict: {verdict}")
    print(f"Best: {best['loss']} lr={best['lr']:g} gain={g:+.2f}pp "
          f"sigma_eq={best['sigma_equiv_at_best']:.1e}")

    exp_dir = args.experiment_dir or "results/grad_reach"
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump({"base_A": base_A, "base_B": base_B, "d_scope": d_scope,
                   "scope": args.scope, "runs": runs, "verdict": verdict}, f, indent=2)
    print(f"\nSaved {exp_dir}/results.json")
    if wandb_run:
        wandb_run.summary["best_gain_B"] = best["gain_B"]
        wandb_run.summary["verdict"] = verdict
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
