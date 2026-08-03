"""Protocol-matched fine-tuning + base-eval baseline for the RandOpt comparison.

Fine-tunes the released DINOv2 backbone + ImageNet linear head on a
class-balanced clean ImageNet-train draw (same seed/sampler, same
--train_manifest, as scripts/randopt_shift.py's scoring set), then evaluates
BOTH the frozen base model and the fine-tuned model on the same --dataset
target(s) randopt_shift.py evaluates on — base/FT/RandOpt are always
comparable because they resolve every eval target through the identical
registry (manifest, split, input_mode, logit mask).

--dataset is the same comma-separated plusarg as randopt_shift.py (imported
directly, not duplicated): any of imagenet/imagenet_c/imagenet_a/r/sketch/es,
or 'all' for the 4 shift datasets. The logit mask is applied identically to
base and FT eval for masked targets (imagenet_a/r/es) — applying it to one
method and not the other would invalidate the comparison (spec S4.2).

This is the gradient-based counterpart to selection: same model, same data
budget, same eval. If FT also lands ~0 vs base on some target, RandOpt's
null there is the task ceiling, not a method failure.

Runs as a plain single-GPU script (no Ray): needs ~24 GB GPU for full-model
fp32 AdamW on ViT-g (A6000/large-GPU servers; NOT the 11 GB 2080 Tis).
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class, score  # noqa: E402
from scripts.randopt_shift import (  # noqa: E402
    _DATASET_REGISTRY, _parse_dataset_list, load_logit_mask)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                    help="comma-separated eval targets (like randopt_shift.py), "
                         "any of: " + ", ".join(_DATASET_REGISTRY) + ", or 'all'")
    p.add_argument("--train_manifest", default="data/imagenet/data.json",
                    help="clean ImageNet manifest for fine-tuning — never one "
                         "of the eval targets")
    p.add_argument("--train_samples", type=int, default=1000,
                    help="class-balanced fine-tuning draw size; 0 = full split")
    p.add_argument("--test_samples", type=int, default=1000,
                    help="class-balanced subsample size per eval target; "
                         "0 = full split")
    p.add_argument("--test_manifest", default=None,
                    help="override the manifest path for a single eval "
                         "target (only applies when --dataset names exactly "
                         "one dataset)")
    p.add_argument("--backbone_family", default="dinov2")
    p.add_argument("--weights_path", default=None)
    p.add_argument("--head_path", default=None)
    p.add_argument("--ft_scope", default="all", choices=["all", "head", "last_n_blocks"],
                    help="which params AdamW updates — same scope semantics as "
                         "RandOpt's perturb_target, so FT and RandOpt optimize/"
                         "perturb the identical param subset for a fair "
                         "comparison. 'all' (default) full-model FT needs ~24GB "
                         "for ViT-g; DINOv3's ViT-7B needs last_n_blocks to fit "
                         "a single GPU (full fp32 AdamW would need ~108GB).")
    p.add_argument("--ft_last_n_blocks", type=int, default=0,
                    help="only used when --ft_scope last_n_blocks")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    args = p.parse_args()
    args.dataset_targets = _parse_dataset_list(args.dataset)
    return args


def evaluate(engine, items, input_mode, handler, logit_mask=None):
    engine.backbone.eval(); engine.head.eval()
    with torch.no_grad():
        preds = engine.predict(items, input_mode, logit_mask)
    return score(handler, preds, items)


def load_targets(args, rng):
    """One eval-set entry per --dataset target, resolved through the SAME
    registry randopt_shift.py uses — manifest, split, input_mode, logit mask
    all identical to what the RandOpt runs on these targets see."""
    from data_handlers import get_dataset_handler
    targets = {}
    for name in args.dataset_targets:
        reg = _DATASET_REGISTRY[name]
        h = get_dataset_handler(name)
        manifest = (args.test_manifest if (len(args.dataset_targets) == 1 and args.test_manifest)
                   else (reg["manifest"] or h.default_train_path))
        mask = load_logit_mask(name)
        items = sample_per_class(h.load_data(manifest, split=reg["split"]),
                                 args.test_samples, rng)
        targets[name] = {"handler": h, "mask": mask, "items": items,
                         "input_mode": reg["input_mode"]}
        print(f"eval target ({name}, split={reg['split']}, "
              f"input_mode={reg['input_mode']}): {manifest} | "
              f"logit_mask: {'None (full 1000-way)' if mask is None else f'{int((mask == 0).sum())}-class subset'} "
              f"| n={len(items)}")
    return targets


def main(args):
    dataset_tag = "-".join(args.dataset_targets)
    tag = f"ft-d2-tr{args.train_samples // 1000}k-{dataset_tag}"
    import wandb
    run = wandb.init(project=args.wandb_project, name=args.wandb_name or tag,
                     config=vars(args))

    from data_handlers import get_dataset_handler
    from data_handlers.imagenet_c import load_image_batch
    from vision.ssl_engine import SSLEngineImpl

    rng = np.random.default_rng(args.global_seed)

    # any handler's compute_reward/extract_answer is identical (schema-driven);
    # used for the clean-ImageNet fine-tuning draw + its reward compute
    train_handler = get_dataset_handler(args.dataset_targets[0])
    train_items = sample_per_class(
        train_handler.load_data(args.train_manifest, split="train"),
        args.train_samples, rng)
    print(f"FT train (clean ImageNet, unmasked): {len(train_items)}")

    targets = load_targets(args, rng)

    engine = SSLEngineImpl(backbone_family=args.backbone_family,
                           weights_path=args.weights_path,
                           head_path=args.head_path)
    device = engine.device

    base = {"base/train_acc": evaluate(engine, train_items, "official_resize",
                                       train_handler)}
    for name, info in targets.items():
        base[f"base/{name}_acc"] = evaluate(
            engine, info["items"], info["input_mode"], info["handler"], info["mask"])
    print("BASE:", {k: round(v, 4) for k, v in base.items()})
    run.log(base)

    if args.ft_scope == "all":
        trainable = dict(engine._all_params())
    else:
        engine.set_perturb_scope(args.ft_scope, args.ft_last_n_blocks)
        trainable = dict(engine._perturb_params())
    for name, p in engine._all_params():
        p.requires_grad_(name in trainable)
    print(f"FT scope '{args.ft_scope}': {len(trainable)} trainable tensors "
          f"({sum(p.numel() for p in trainable.values())/1e6:.1f}M params)")

    # ft_scope=all stores weights+grad+optimizer moments in fp32 (16 bytes/
    # param total) -- 6.72B backbone params means ~107.5GB, doesn't fit a
    # 46GB GPU even before considering activations. bf16 storage halves
    # every one of those four tensors (8 bytes/param, ~53.8GB) -- confirmed
    # by a real test still NOT sufficient alone (OOM'd inside opt.step(),
    # allocating the 4th of 4 required tensors, 44.51/44.55GiB in use).
    # 8-bit optimizer moments (bitsandbytes) additionally cut exp_avg/
    # exp_avg_sq to ~1 byte/param each (~40.3GB) -- but a real test showed
    # the actual wall is elsewhere: it OOM'd mid-forward-pass, inside one
    # block's MLP, at ~40.8GB used, *before* the optimizer state was ever
    # touched. With every one of the 40 blocks trainable, autograd must
    # retain EVERY block's activations for backward (unlike last_n_blocks
    # scope, where the frozen early blocks' outputs don't require_grad and
    # their activations are never saved at all) -- that's tens of GB right
    # there, dwarfing the weight/optimizer costs above. Gradient
    # checkpointing addresses exactly this: recompute each block's forward
    # during backward instead of keeping all 40 blocks' activations
    # resident at once. Can't edit the pinned dinov3 repo's block-loop code
    # (CLAUDE.md: never vendor/edit the official clone) -- so each block's
    # own `forward` is monkey-patched at runtime instead, from here.
    model_dtype = torch.bfloat16 if args.ft_scope == "all" else torch.float32
    if model_dtype is torch.bfloat16:
        engine.backbone.to(model_dtype)
        engine.head.to(model_dtype)
    if args.ft_scope == "all":
        import torch.utils.checkpoint as ckpt

        def _checkpointed(fwd):
            def wrapped(*a, **kw):
                return ckpt.checkpoint(fwd, *a, use_reentrant=False, **kw)
            return wrapped

        for block in engine.backbone.blocks:
            block.forward = _checkpointed(block.forward)

    params = list(trainable.values())
    if args.ft_scope == "all":
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=args.lr, weight_decay=0.0)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
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
                                     transform).to(device, dtype=model_dtype)
            labels = torch.tensor([labels_all[j] for j in idx], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f_out = engine.backbone.forward_features(batch)
                feat = torch.cat([f_out["x_norm_clstoken"],
                                  f_out["x_norm_patchtokens"].mean(dim=1)], dim=1)
                logits = engine.head(feat.to(engine.head.weight.dtype))
                loss = criterion(logits, labels)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        print(f"epoch {epoch + 1}/{args.epochs} loss={np.mean(losses):.4f}")
        run.log({"ft/epoch": epoch + 1, "ft/train_loss": float(np.mean(losses))})

    final = {"ft/train_acc": evaluate(engine, train_items, "official_resize",
                                      train_handler)}
    for name, info in targets.items():
        acc = evaluate(engine, info["items"], info["input_mode"],
                       info["handler"], info["mask"])
        final[f"ft/{name}_acc"] = acc
        final[f"ft/{name}_gain_over_base"] = (acc - base[f"base/{name}_acc"]) * 100
    print("FINAL:", {k: round(v, 4) for k, v in final.items()})
    run.log(final)
    run.finish()


if __name__ == "__main__":
    main(parse_args())
