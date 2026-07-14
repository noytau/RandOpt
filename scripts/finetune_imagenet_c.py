"""E1 Task 3: finetune baseline on ImageNet-C (DINOv2-giant, full backbone + head).

User-specified config (2026-07-14): train ALL transformer layers (+head),
AdamW lr=1e-5 with cosine annealing to 0 over the whole run, max batch found
at runtime via OOM backoff (halve until it fits), bf16 autocast.

Head is initialized from Task 2's trained probe head (--head_init) — a random
head would destroy the backbone in the first steps.

POC (per method of work): ONE fixed config, no val usage, test evaluated once
at the end — both with the trained head (top-1) and with the kNN readout of
the finetuned backbone (k=20 on re-embedded train gallery), so the FT rung is
comparable to the kNN rung too. Logs ||Δw|| and its equivalent-σ
(= ||Δw|| / sqrt(d)) so FT's weight movement is comparable to RandOpt's
search radius.
"""
import argparse
import contextlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_handlers import get_dataset_handler

_NORM_MEAN = [0.485, 0.456, 0.406]  # = VisionEngine._NORM_MEAN
_NORM_STD = [0.229, 0.224, 0.225]


def amp():
    """bf16 autocast on GPU; no-op on CPU (local dry-runs)."""
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def load_split_tensors(args, split):
    handler = get_dataset_handler("imagenet_c")
    items = handler.load_data(
        os.path.join(args.data_root, args.corruption, str(args.severity)),
        split=split)
    x = torch.stack([d["image_tensor"] for d in items])
    y = torch.tensor([d["class_id"] for d in items], dtype=torch.long)
    return x, y


def preprocess(batch, mean, std):
    batch = F.interpolate(batch, size=(224, 224), mode="bicubic",
                          align_corners=False)
    return (batch - mean) / std


def find_max_batch(model, head, mean, std, device, start: int) -> int:
    """Halve the batch size until one fwd+bwd step fits in memory."""
    if not torch.cuda.is_available():
        return min(start, 16)  # CPU dry-run
    bs = start
    while bs >= 4:
        try:
            x = torch.rand(bs, 3, 224, 224, device=device)
            with amp():
                out = model(pixel_values=preprocess(x, mean, std))
                loss = head(out.pooler_output).float().mean()
            loss.backward()
            model.zero_grad(set_to_none=True)
            head.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            print(f"[batch-probe] batch={bs} fits")
            return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[batch-probe] batch={bs} OOM -> halving")
            bs //= 2
    raise RuntimeError("no batch size >= 4 fits")


@torch.no_grad()
def eval_test(model, head, test_x, test_y, mean, std, device, bs):
    """Head top-1 AND kNN readout features for the test split."""
    correct, cls_feats = 0, []
    for i in range(0, len(test_y), bs):
        b = test_x[i:i + bs].to(device)
        with amp():
            out = model(pixel_values=preprocess(b, mean, std))
            logits = head(out.pooler_output)
        correct += (logits.float().argmax(1).cpu() == test_y[i:i + bs]).sum().item()
        cls_feats.append(out.last_hidden_state[:, 0, :].float().cpu())
    return correct / len(test_y), torch.cat(cls_feats)


@torch.no_grad()
def embed_cls(model, x, mean, std, device, bs):
    feats = []
    for i in range(0, len(x), bs):
        with amp():
            out = model(pixel_values=preprocess(x[i:i + bs].to(device), mean, std))
        feats.append(out.last_hidden_state[:, 0, :].float().cpu())
    return torch.cat(feats)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="facebook/dinov2-giant")
    p.add_argument("--data_root", default="data/imagenet_c")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--severity", type=int, default=3)
    p.add_argument("--head_init", default=None,
                   help="probe head .pt (default results/e1-probe-<corr>-s<sev>/head.pt)")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=64,
                   help="starting point for the max-batch OOM backoff")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save_model", action="store_true",
                   help="also save the finetuned backbone (~4.5GB)")
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_run_name", default=None)
    return p.parse_args()


def main(args):
    from transformers import Dinov2Model
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_run_name or f"e1-ft-{args.corruption}-s{args.severity}"
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config=vars(args))

    train_x, train_y = load_split_tensors(args, "train")
    test_x, test_y = load_split_tensors(args, "test")
    print(f"train={len(train_y)} test={len(test_y)}")

    model = Dinov2Model.from_pretrained(args.model_name).to(device)
    head_path = args.head_init or (
        f"results/e1-probe-{args.corruption}-s{args.severity}/head.pt")
    sd = torch.load(head_path, map_location="cpu")
    head = nn.Linear(sd["weight"].shape[1], sd["weight"].shape[0]).to(device)
    head.load_state_dict(sd)
    print(f"head init <- {head_path}")

    mean = torch.tensor(_NORM_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(_NORM_STD, device=device).view(1, 3, 1, 1)
    base = {n: p.detach().clone() for n, p in model.named_parameters()}
    d_scope = sum(p.numel() for p in model.parameters()) \
        + sum(p.numel() for p in head.parameters())

    model.train()
    bs = find_max_batch(model, head, mean, std, device, args.batch_size)
    steps_total = math.ceil(len(train_y) / bs) * args.epochs
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_total)
    loss_fn = nn.CrossEntropyLoss()
    print(f"full-model FT: {d_scope:,} trainable params, batch={bs}, "
          f"{steps_total} steps, lr={args.lr} cosine")

    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        perm = torch.randperm(len(train_y))
        for i in range(0, len(train_y), bs):
            idx = perm[i:i + bs]
            xb = train_x[idx].to(device)
            yb = train_y[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with amp():
                out = model(pixel_values=preprocess(xb, mean, std))
                loss = loss_fn(head(out.pooler_output).float(), yb)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                print(f"ep{ep} step {step}/{steps_total} loss={loss.item():.4f} "
                      f"lr={sched.get_last_lr()[0]:.2e} [{el:.0f}s]", flush=True)
                if wandb_run:
                    wandb_run.log({"ft/train_loss": loss.item(),
                                   "ft/lr": sched.get_last_lr()[0],
                                   "step": step})

    model.eval()
    with torch.no_grad():
        dw2 = sum(((p.detach() - base[n]).float() ** 2).sum().item()
                  for n, p in model.named_parameters())
    dw = math.sqrt(dw2)
    sigma_equiv = dw / math.sqrt(d_scope)

    head_acc, test_cls = eval_test(model, head, test_x, test_y, mean, std,
                                   device, bs)
    gal_cls = embed_cls(model, train_x, mean, std, device, bs)
    g = F.normalize(gal_cls, dim=-1).to(device)
    q = F.normalize(test_cls, dim=-1).to(device)
    gl = train_y.to(device)
    knn_correct = 0
    for i in range(0, len(test_y), 2048):
        sim = q[i:i + 2048] @ g.T
        topv, topi = sim.topk(20, dim=1)
        w = (topv / 0.07).exp()
        votes = torch.zeros(sim.shape[0], int(gl.max()) + 1, device=device)
        votes.scatter_add_(1, gl[topi], w)
        knn_correct += (votes.argmax(1).cpu() == test_y[i:i + 2048]).sum().item()
    knn_acc = knn_correct / len(test_y)

    print(f"TEST head top1={head_acc*100:.2f}  knn-readout top1={knn_acc*100:.2f}  "
          f"||dw||={dw:.3f}  sigma_equiv={sigma_equiv:.2e}")
    if wandb_run:
        wandb_run.log({"ft/test_top1": head_acc, "ft/test_knn_top1": knn_acc,
                       "ft/delta_w": dw, "ft/sigma_equiv": sigma_equiv,
                       "ft/batch_size": bs})

    exp_dir = args.experiment_dir or (
        f"results/e1-ft-{args.corruption}-s{args.severity}")
    os.makedirs(exp_dir, exist_ok=True)
    torch.save(head.cpu().state_dict(), f"{exp_dir}/head_ft.pt")
    if args.save_model:
        model.save_pretrained(f"{exp_dir}/backbone_ft")
    with open(f"{exp_dir}/results.json", "w") as f:
        json.dump({"config": vars(args), "batch_size": bs,
                   "test_top1": head_acc, "test_knn_top1": knn_acc,
                   "delta_w": dw, "sigma_equiv": sigma_equiv}, f, indent=2)
    print(f"Saved {exp_dir}/results.json")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main(parse_args())
