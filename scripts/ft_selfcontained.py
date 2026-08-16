"""Protocol-matched fine-tuning baseline for self-contained datasets (own
train/val/test split, e.g. exdark) — the gradient-based counterpart to
scripts/randopt_selfcontained.py, for a fair RandOpt-vs-FT comparison.

Same center, same data, same eval as randopt_selfcontained.py: starts from
the SAME fitted head (checkpoints/<dataset>_head.pth, from scripts/fit_head.py)
— not a fresh one — fine-tunes it (by default --ft_scope all: the whole
backbone + head, matching RandOpt's default perturb_target=all) via AdamW on
the SAME train_holdout split RandOpt scores perturbations on, then evaluates
both the frozen base and the fine-tuned model on the same test split. Base
numbers here should match randopt_selfcontained.py's BASE line exactly (same
head, same data) — if they don't, something about the two protocols has
drifted apart.

Mirrors scripts/ft_matched_baseline.py's role for the ImageNet shift suite
(same data budget, same eval, gradient descent instead of selection-over-
noise), but for a private-label-space dataset — reuses vision/head_probe.py's
training loop (already generic over which params are trainable) instead of
duplicating it.

Usage:
    python scripts/ft_selfcontained.py --dataset exdark
    python scripts/ft_selfcontained.py --dataset exdark --ft_scope all --epochs 5 --lr 1e-5
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class, score  # noqa: E402
from scripts.randopt_selfcontained import _head_path, _SELFCONTAINED_REGISTRY  # noqa: E402
from vision.head_probe import fit_linear_head  # noqa: E402
from vision.ssl_engine import SSLEngineImpl  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=list(_SELFCONTAINED_REGISTRY))
    p.add_argument("--backbone_family", default="dinov2", choices=["dinov2", "dinov3"])
    p.add_argument("--backbone_name", default=None)
    p.add_argument("--weights_path", default=None)
    p.add_argument("--head_dir", default="checkpoints")
    p.add_argument("--ft_scope", default="all", choices=["all", "head", "last_n_blocks"],
                    help="which params AdamW updates -- 'all' (default) matches "
                         "RandOpt's default perturb_target=all for a fair "
                         "comparison")
    p.add_argument("--ft_last_n_blocks", type=int, default=0)
    p.add_argument("--train_samples", type=int, default=0,
                    help="class-balanced sample from the dataset's own scoring "
                         "split (registry's score_split, e.g. exdark's "
                         "train_holdout); 0 = full split -- match whatever "
                         "RandOpt scored on for a fair comparison")
    p.add_argument("--test_samples", type=int, default=0, help="0 = full test split")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--backbone_out", default=None,
                    help="where to save the fine-tuned backbone's state_dict "
                         "(default: <head_dir>/<dataset>_ft_<ft_scope>_backbone.pth) "
                         "-- lets a later cross-eval (e.g. on clean ImageNet "
                         "with the original head) replay this exact backbone")
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    return p.parse_args()


def main(args):
    from data_handlers import get_dataset_handler
    reg = _SELFCONTAINED_REGISTRY[args.dataset]
    handler = get_dataset_handler(args.dataset)
    manifest = reg["manifest"] or handler.default_train_path
    head_path = _head_path(args.dataset, args.head_dir)
    score_split = reg.get("score_split", "train")

    tag = f"ft-self-{args.dataset}-{args.ft_scope}"
    wandb_run = None
    if args.wandb_project:
        import wandb
        wandb_run = wandb.init(project=args.wandb_project,
                               name=args.wandb_name or tag, config=vars(args))

    rng = np.random.default_rng(args.global_seed)
    train_items = sample_per_class(
        handler.load_data(manifest, split=score_split), args.train_samples, rng)
    test_items = sample_per_class(
        handler.load_data(manifest, split="test"), args.test_samples, rng)
    print(f"[{args.dataset}] FT train (own {score_split}): {len(train_items)} | "
          f"eval (own test): {len(test_items)} | head: {head_path}")

    # SAME starting center RandOpt perturbs around -- the already-fit head,
    # not a fresh one.
    engine = SSLEngineImpl(backbone_family=args.backbone_family,
                           backbone_name=args.backbone_name,
                           weights_path=args.weights_path,
                           num_classes=reg["num_classes"],
                           head_path=head_path,
                           input_mode=reg["input_mode"])

    def evaluate(items):
        engine.backbone.eval()
        engine.head.eval()
        with torch.no_grad():
            preds = engine.predict(items, reg["input_mode"], None)
        return score(handler, preds, items)

    base_train = evaluate(train_items)
    base_test = evaluate(test_items)
    print(f"[{args.dataset}] BASE: train_reward={base_train:.4f} "
          f"test_accuracy={base_test:.4f}")
    if wandb_run:
        wandb_run.log({"base/train_reward": base_train, "base/test_accuracy": base_test})

    # snapshot the trainable subset BEFORE fitting, to measure weight
    # displacement (delta_w / sigma_equiv) comparable to RandOpt's sigma --
    # same scope-selection logic fit_linear_head uses internally. CPU, not
    # GPU (matches ft_matched_baseline.py's already-fixed pattern) -- a
    # GPU-resident clone of all-scope's 6.72B params (~27GB) sitting
    # alongside fit_linear_head's own bf16+8bit-optimizer+checkpointing
    # footprint is exactly the kind of avoidable OOM that pattern was
    # fixed for elsewhere.
    if args.ft_scope == "all":
        trainable = dict(engine._all_params())
    else:
        engine.set_perturb_scope(args.ft_scope, args.ft_last_n_blocks)
        trainable = dict(engine._perturb_params())
    w0 = {n: p.detach().cpu().clone() for n, p in trainable.items()}

    fit_linear_head(engine, train_items, handler, reg["input_mode"],
                    lr=args.lr, epochs=args.epochs, batch_size=args.batch_size,
                    scope=args.ft_scope, last_n_blocks=args.ft_last_n_blocks,
                    seed=args.global_seed)

    with torch.no_grad():
        sq, n_par = 0.0, 0
        for n, p in trainable.items():
            sq += (p.detach().cpu().float() - w0[n].float()).pow(2).sum().item()
            n_par += p.numel()
    delta_w = sq ** 0.5
    sigma_equiv = delta_w / (n_par ** 0.5)

    ft_train = evaluate(train_items)
    ft_test = evaluate(test_items)
    gain = (ft_test - base_test) * 100
    print(f"[{args.dataset}] FT ({args.ft_scope}): train_reward={ft_train:.4f} "
          f"test_accuracy={ft_test:.4f} [{gain:+.2f}pp vs base] "
          f"delta_w={delta_w:.4f} sigma_equiv={sigma_equiv:.2e}")
    if wandb_run:
        wandb_run.log({
            "ft/train_reward": ft_train, "ft/test_accuracy": ft_test,
            "ft/gain_over_base": gain,
            "ft/delta_w": delta_w, "ft/sigma_equiv": sigma_equiv,
        })
        wandb_run.finish()

    backbone_out = args.backbone_out or os.path.join(
        args.head_dir, f"{args.dataset}_ft_{args.ft_scope}_backbone.pth")
    torch.save(engine.backbone.state_dict(), backbone_out)
    print(f"saved fine-tuned backbone: {backbone_out}")


if __name__ == "__main__":
    main(parse_args())
