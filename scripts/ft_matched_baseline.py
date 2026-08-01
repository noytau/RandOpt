"""Protocol-matched fine-tuning baseline for the RandOpt null.

Fine-tunes the released DINOv2 backbone + ImageNet linear head on the SAME
class-balanced clean ImageNet-train draw that RandOpt used as its scoring set
(seed 42 -> identical images), then evaluates on (a) a disjoint clean holdout
and (b) the SAME class-balanced ImageNet-C test draw RandOpt voted on.

This is the gradient-based counterpart to selection: same model, same data
budget, same eval. If FT also lands ~0 vs base, RandOpt's null is the task
ceiling, not a method failure.

Configs mirroring the RandOpt table:
    --train_samples 1000 --test_samples 5000     (tr1k / eval5k)
    --train_samples 5000 --test_samples 1000     (tr5k / eval1k)

Runs as a plain single-GPU script (no Ray): needs ~24 GB GPU for full-model
fp32 AdamW on ViT-g (A6000/large-GPU servers; NOT the 11 GB 2080 Tis).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class, score  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", default="data/imagenet/data.json")
    p.add_argument("--test_manifest", default="data/imagenet_c/data.json")
    p.add_argument("--train_samples", type=int, default=1000)
    p.add_argument("--test_samples", type=int, default=5000)
    p.add_argument("--clean_eval_samples", type=int, default=0,
                   help="clean holdout size; 0 = complement pattern "
                        "(5k when training on 1k, 1k when training on 5k)")
    p.add_argument("--backbone_family", default="dinov2")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    return p.parse_args()


def evaluate(engine, items, input_mode, handler):
    engine.backbone.eval(); engine.head.eval()
    with torch.no_grad():
        preds = engine.predict(items, input_mode)
    return score(handler, preds, items)


def main(args):
    tag = (f"ft-d2-tr-intrain{args.train_samples//1000}k-"
           f"te-ic{args.test_samples//1000}k")
    import wandb
    run = wandb.init(project=args.wandb_project,
                     name=args.wandb_name or tag, config=vars(args))

    from data_handlers import get_dataset_handler
    from data_handlers.imagenet_c import load_image_batch
    from vision.ssl_engine import SSLEngineImpl

    handler = get_dataset_handler("imagenet_c")
    rng = np.random.default_rng(args.global_seed)

    all_train = handler.load_data(args.train_manifest, split="train")
    # identical draw to the RandOpt runs (same seed, same sampler)
    train_items = sample_per_class(all_train, args.train_samples, rng)
    test_items = sample_per_class(
        handler.load_data(args.test_manifest, split="test"),
        args.test_samples, rng)

    # clean holdout: complement of the training draw, class-balanced
    train_paths = {d["image_path"] for d in train_items}
    leftover = [d for d in all_train if d["image_path"] not in train_paths]
    n_clean = args.clean_eval_samples or (
        5000 if args.train_samples <= 1000 else 1000)
    clean_eval = sample_per_class(leftover, n_clean,
                                  np.random.default_rng(args.global_seed + 1))
    print(f"FT train: {len(train_items)} | clean holdout: {len(clean_eval)} "
          f"| IC test: {len(test_items)}")

    engine = SSLEngineImpl(backbone_family=args.backbone_family)
    device = engine.device

    base = {
        "base/train_acc": evaluate(engine, train_items, "official_resize", handler),
        "base/clean_eval_acc": evaluate(engine, clean_eval, "official_resize", handler),
        "base/ic_acc": evaluate(engine, test_items, "presized224", handler),
    }
    print("BASE:", {k: round(v, 4) for k, v in base.items()})
    run.log(base)

    w0 = {n: p.detach().clone() for n, p in engine.backbone.named_parameters()}
    w0.update({f"head.{n}": p.detach().clone()
               for n, p in engine.head.named_parameters()})

    params = (list(engine.backbone.parameters())
              + list(engine.head.parameters()))
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    steps_per_epoch = (len(train_items) + args.batch_size - 1) // args.batch_size
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch)
    transform = engine.transforms["official_resize"]
    labels_all = [int(d["class_id"]) for d in train_items]

    engine.backbone.train(); engine.head.train()
    order = np.arange(len(train_items))
    for epoch in range(args.epochs):
        rng.shuffle(order)
        losses = []
        for i in range(0, len(order), args.batch_size):
            idx = order[i:i + args.batch_size]
            batch = load_image_batch([train_items[j] for j in idx],
                                     transform).to(device)
            labels = torch.tensor([labels_all[j] for j in idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f_out = engine.backbone.forward_features(batch)
                feat = torch.cat([f_out["x_norm_clstoken"],
                                  f_out["x_norm_patchtokens"].mean(dim=1)], dim=1)
                logits = engine.head(feat.float())
                loss = F.cross_entropy(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f}")
        run.log({"ft/epoch": epoch + 1, "ft/train_loss": float(np.mean(losses))})

    # weight displacement vs base -> comparable to RandOpt's sigma*sqrt(P)
    with torch.no_grad():
        sq, n_par = 0.0, 0
        cur = {n: p for n, p in engine.backbone.named_parameters()}
        cur.update({f"head.{n}": p for n, p in engine.head.named_parameters()})
        for n, p in cur.items():
            sq += (p.detach() - w0[n]).float().pow(2).sum().item()
            n_par += p.numel()
    delta_w = sq ** 0.5

    final = {
        "ft/train_acc": evaluate(engine, train_items, "official_resize", handler),
        "ft/clean_eval_acc": evaluate(engine, clean_eval, "official_resize", handler),
        "ft/ic_acc": evaluate(engine, test_items, "presized224", handler),
        "ft/delta_w": delta_w,
        "ft/sigma_equiv": delta_w / (n_par ** 0.5),
    }
    final["ft/ic_gain_over_base"] = (final["ft/ic_acc"] - base["base/ic_acc"]) * 100
    final["ft/clean_gain_over_base"] = (
        final["ft/clean_eval_acc"] - base["base/clean_eval_acc"]) * 100
    print("FINAL:", {k: round(v, 4) for k, v in final.items()})
    run.log(final)
    run.finish()


if __name__ == "__main__":
    main(parse_args())
