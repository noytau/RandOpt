"""E1 Task 1: kNN classification baseline on ImageNet-C (frozen DINOv2 backbone).

Two modes, per the method of work (CLAUDE.md):
  POC (default, --sweep_k empty): ONE fixed config — k (default 20, the
      DINO/DINOv2 protocol value), tau=0.07 — gallery = train features,
      evaluated directly on test. The resulting test top-1 is E1's base /
      headroom check.
  Sweep (--sweep_k "1,3,5,..."): evaluate each k on the VAL split, pick the
      best, then evaluate test once with that k (larger-tier hyperparameter
      selection; test stays single-touch).

Consumes the shared feature cache (vision/features.py) — after the one-time
extraction this script does no backbone forwards; a full k sweep is seconds.
Vote math (weighted kNN, temperature tau) is identical to
VisionEngine.eval_global / knn_predict for parity with the RandOpt rung.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.features import extract_split_features


def knn_top1(gallery: torch.Tensor, g_labels: torch.Tensor,
             queries: torch.Tensor, q_labels: torch.Tensor,
             k: int = 20, tau: float = 0.07, chunk: int = 2048) -> float:
    """Weighted-vote kNN top-1 (same math as VisionEngine.eval_global)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = F.normalize(gallery.float(), dim=-1).to(device)
    q = F.normalize(queries.float(), dim=-1).to(device)
    gl = g_labels.to(device)
    ql = q_labels.to(device)
    num_classes = int(gl.max()) + 1
    correct = 0
    for i in range(0, len(q), chunk):
        sim = q[i:i + chunk] @ g.T                       # (C, G)
        topv, topi = sim.topk(min(k, sim.shape[1]), dim=1)
        w = (topv / tau).exp()
        votes = torch.zeros(sim.shape[0], num_classes, device=device)
        votes.scatter_add_(1, gl[topi], w)
        correct += (votes.argmax(dim=1) == ql[i:i + chunk]).sum().item()
    return correct / len(q)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="facebook/dinov2-giant")
    p.add_argument("--data_root", default="data/imagenet_c")
    p.add_argument("--cache_dir", default="results/features")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--k", type=int, default=20, help="POC fixed k (DINOv2 protocol)")
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--sweep_k", default="",
                   help="comma list; if set, select k on VAL then test once")
    p.add_argument("--batch_size", type=int, default=64, help="feature extraction")
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main(args):
    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_run_name or f"e1-knn-{args.corruption}-s{args.severity}"
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config=vars(args))

    feats = {}
    splits = ["train", "test"] + (["val"] if args.sweep_k else [])
    for split in splits:
        feats[split] = extract_split_features(
            args.model_name, args.corruption, args.severity, split,
            data_root=args.data_root, cache_dir=args.cache_dir,
            batch_size=args.batch_size)
        print(f"  {split}: {feats[split]['cls'].shape[0]} features")

    gallery, g_labels = feats["train"]["cls"], feats["train"]["labels"]
    results = {"config": vars(args), "val_sweep": {}}

    if args.sweep_k:
        ks = sorted({int(x) for x in args.sweep_k.split(",")})
        best_k, best_val = None, -1.0
        for k in ks:
            acc = knn_top1(gallery, g_labels,
                           feats["val"]["cls"], feats["val"]["labels"],
                           k=k, tau=args.tau)
            results["val_sweep"][k] = acc
            print(f"val  k={k:>4d}: top1={acc*100:.2f}")
            if wandb_run:
                wandb_run.log({"knn/val_top1": acc, "knn/k": k})
            if acc > best_val:
                best_k, best_val = k, acc
        chosen_k = best_k
        print(f"chosen k={chosen_k} (val top1={best_val*100:.2f})")
    else:
        chosen_k = args.k
        print(f"POC mode: fixed k={chosen_k} (no val selection)")

    test_acc = knn_top1(gallery, g_labels,
                        feats["test"]["cls"], feats["test"]["labels"],
                        k=chosen_k, tau=args.tau)
    print(f"TEST top1={test_acc*100:.2f} (k={chosen_k}, tau={args.tau}) "
          f"— this is E1's base/headroom number")
    results.update({"chosen_k": chosen_k, "test_top1": test_acc})
    if wandb_run:
        wandb_run.log({"base/test_accuracy": test_acc,
                       "knn/test_top1": test_acc, "knn/chosen_k": chosen_k})

    exp_dir = args.experiment_dir or (
        f"results/e1-knn-{args.corruption}-s{args.severity}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {exp_dir}/results.json")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
