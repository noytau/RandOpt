"""E1 Task 2: linear-classifier baseline on ImageNet-C (frozen DINOv2 backbone).

Trains nn.Linear(embed_dim -> 1000) on cached `pooler` features (see
vision/features.py) — no backbone forwards, so a run takes minutes even on CPU.

Two modes, per the method of work (CLAUDE.md):
  POC (default, --sweep_lr empty): ONE fixed config (AdamW, --lr, --epochs,
      CE), trained on the train split, evaluated ONCE on test at the end.
      No val usage.
  Sweep (--sweep_lr "3e-4,1e-3,..."): train once per LR, pick the best by VAL
      accuracy, then evaluate test once with the winner (larger tier).

The trained head is saved as a state dict loadable by
VisionEngine(linear_init_path=...) — it later doubles as the task-adapted
center option for RandOpt and as the FT baseline's head init.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision.features import extract_split_features


def train_head(train_x, train_y, lr: float, epochs: int, batch_size: int,
               device: str, seed: int = 0) -> nn.Linear:
    """AdamW + CrossEntropy on precomputed features; returns the trained head."""
    torch.manual_seed(seed)
    num_classes = int(train_y.max()) + 1
    head = nn.Linear(train_x.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x = train_x.float().to(device)
    y = train_y.to(device)
    n = len(y)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(head(x[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        yield ep, total / n, head


def head_top1(head: nn.Linear, feats: torch.Tensor, labels: torch.Tensor,
              device: str, chunk: int = 8192) -> float:
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels), chunk):
            logits = head(feats[i:i + chunk].float().to(device))
            correct += (logits.argmax(1).cpu() == labels[i:i + chunk]).sum().item()
    return correct / len(labels)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="facebook/dinov2-giant")
    p.add_argument("--data_root", default="data/imagenet_c")
    p.add_argument("--cache_dir", default="results/features")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3, help="POC fixed LR")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--train_batch", type=int, default=1024)
    p.add_argument("--sweep_lr", default="",
                   help="comma list; if set, select LR on VAL then test once")
    p.add_argument("--batch_size", type=int, default=64, help="feature extraction")
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main(args):
    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_run_name or f"e1-probe-{args.corruption}-s{args.severity}"
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config=vars(args))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    feats = {}
    splits = ["train", "test"] + (["val"] if args.sweep_lr else [])
    for split in splits:
        feats[split] = extract_split_features(
            args.model_name, args.corruption, args.severity, split,
            data_root=args.data_root, cache_dir=args.cache_dir,
            batch_size=args.batch_size)
        print(f"  {split}: {feats[split]['pooler'].shape[0]} features")

    tr_x, tr_y = feats["train"]["pooler"], feats["train"]["labels"]
    lrs = ([float(x) for x in args.sweep_lr.split(",")] if args.sweep_lr
           else [args.lr])
    results = {"config": vars(args), "lr_sweep": {}}

    best = None  # (val_acc, lr, head)
    for lr in lrs:
        head = None
        for ep, loss, head in train_head(tr_x, tr_y, lr, args.epochs,
                                         args.train_batch, device):
            if (ep + 1) % 5 == 0 or ep == args.epochs - 1:
                print(f"lr={lr:g} epoch {ep+1}/{args.epochs} loss={loss:.4f}")
            if wandb_run:
                wandb_run.log({f"probe/train_loss_lr{lr:g}": loss, "epoch": ep})
        if args.sweep_lr:
            val_acc = head_top1(head, feats["val"]["pooler"],
                                feats["val"]["labels"], device)
            results["lr_sweep"][lr] = val_acc
            print(f"lr={lr:g}: VAL top1={val_acc*100:.2f}")
            if wandb_run:
                wandb_run.log({"probe/val_top1": val_acc, "probe/lr": lr})
            if best is None or val_acc > best[0]:
                best = (val_acc, lr, head)
        else:
            best = (None, lr, head)
            print(f"POC mode: fixed lr={lr:g}, {args.epochs} epochs "
                  f"(no val selection)")

    _, chosen_lr, head = best
    test_acc = head_top1(head, feats["test"]["pooler"],
                         feats["test"]["labels"], device)
    print(f"TEST top1={test_acc*100:.2f} (lr={chosen_lr:g})")
    results.update({"chosen_lr": chosen_lr, "test_top1": test_acc})
    if wandb_run:
        wandb_run.log({"probe/test_top1": test_acc, "probe/chosen_lr": chosen_lr})

    exp_dir = args.experiment_dir or (
        f"results/e1-probe-{args.corruption}-s{args.severity}")
    os.makedirs(exp_dir, exist_ok=True)
    head_path = f"{exp_dir}/head.pt"
    torch.save(head.cpu().state_dict(), head_path)
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {exp_dir}/results.json and {head_path} "
          f"(loadable via VisionEngine linear_init_path)")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
