"""RandOpt on the ImageNet distribution-shift suite (ImageNet-A/R/Sketch/ES),
plain ImageNet, and ImageNet-C — with an SSL model (DINOv2). Same loop as
scripts/randopt_imagenet_c.py (base eval -> perturbation sampling -> top-K
selection -> majority-vote ensemble), generalized to ANY combination of eval
targets via a comma-separated --dataset (--sigma_values/--top_k_ratios style)
resolved through _DATASET_REGISTRY.

Protocol (matches TASKS.md's established imagenet_c protocol — "score N
perturbations on class-balanced clean ImageNet-train draws; select top-K;
majority-vote on class-balanced [eval target] test draws"):
  - Perturbation SCORING (the "train" phase, incl. base_train_reward) is
    ALWAYS on clean ImageNet (--train_manifest, default data/imagenet/
    data.json) — never one of the --dataset eval targets. Candidates are
    selected purely from how well they do on ordinary ImageNet.
  - Only the FINAL evaluation (base_test_accuracy + the ensemble vote over
    the selected top-K) runs on each named --dataset target's own manifest.
    Naming multiple targets shares the (expensive) scoring phase across all
    of them in one job — only the (cheap) final-eval forward pass repeats
    per target, each model perturbed ONCE per ensemble batch and evaluated
    on every named target before being restored (no extra perturb/restore
    round-trips).
  - The logit mask (utils/logit_mask.py, from manifests/_meta/
    <dataset>_meta.json `subset_indices`) is applied ONLY on the eval side,
    per target (imagenet/imagenet_c/imagenet_sketch have no meta file or an
    explicit null subset -> unmasked full 1000-way; imagenet_a/r/es ->
    masked to their 200-class subset). Applying it to clean-ImageNet
    scoring would be wrong in the other direction: clean ImageNet spans all
    1000 classes, so masking to a 200-class subset there would zero out the
    correct answer for every image whose true label falls outside it.

scripts/randopt_imagenet_c.py is left untouched; this is a separate script so
the working ImageNet-C pipeline can't regress.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from typing import Dict, Tuple

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.randopt_imagenet_c import sample_per_class, score  # noqa: E402

# Registry of every FINAL-eval target randopt_shift.py knows how to resolve:
# manifest (None = the handler's own default_train_path), which split value
# selects the eval rows from it (most datasets use "test"; the clean-ImageNet
# val manifest labels every row "train" regardless of role — a leftover of
# how it was generated for the earlier SSL series), and which transform its
# images need. imagenet_c is the ONE dataset that ships pre-cropped 224x224
# images (Hendrycks' ImageNet-C protocol) -> "presized224" (normalize only);
# every other target here is raw variable-size JPEGs -> "official_resize"
# (Resize 256 -> CenterCrop 224). Getting this wrong isn't cosmetic: applying
# official_resize to an already-224x224 image means resizing UP to 256 then
# cropping back down, an unnecessary blur round-trip.
_DATASET_REGISTRY = {
    "imagenet": {"manifest": "data/imagenet_val10/data.json", "split": "train",
                 "input_mode": "official_resize"},
    "imagenet_c": {"manifest": None, "split": "test", "input_mode": "presized224"},
    "imagenet_a": {"manifest": None, "split": "test", "input_mode": "official_resize"},
    "imagenet_r": {"manifest": None, "split": "test", "input_mode": "official_resize"},
    "imagenet_sketch": {"manifest": None, "split": "test", "input_mode": "official_resize"},
    "imagenet_es": {"manifest": None, "split": "test", "input_mode": "official_resize"},
}
_SHIFT_DATASETS = ["imagenet_a", "imagenet_r", "imagenet_sketch", "imagenet_es"]


def _parse_dataset_list(raw: str):
    """Comma-separated eval targets (same style as --sigma_values/
    --top_k_ratios): any _DATASET_REGISTRY key, or 'all' as shorthand for
    the 4 shift datasets. Order is preserved, duplicates dropped."""
    names = []
    for token in (t.strip() for t in raw.split(",")):
        if token == "all":
            names.extend(_SHIFT_DATASETS)
        elif token in _DATASET_REGISTRY:
            names.append(token)
        else:
            raise ValueError(f"unknown --dataset entry '{token}' -- valid: "
                              f"{list(_DATASET_REGISTRY)}, or 'all'")
    seen = set()
    return [n for n in names if not (n in seen or seen.add(n))]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                    help="comma-separated FINAL-eval targets (like "
                         "--sigma_values), any of: " + ", ".join(_DATASET_REGISTRY) +
                         ", or 'all' (shorthand for the 4 shift datasets: " +
                         ", ".join(_SHIFT_DATASETS) + "). Scoring always runs "
                         "on --train_manifest regardless of this list -- the "
                         "expensive scoring phase (N perturbations) is shared "
                         "across every target named here in one job.")
    p.add_argument("--train_manifest", default="data/imagenet/data.json",
                    help="clean ImageNet manifest for perturbation scoring / "
                         "base_train_reward — never one of the eval targets")
    p.add_argument("--test_manifest", default=None,
                    help="override the manifest path for a single eval "
                         "target (only applies when --dataset names exactly "
                         "one dataset)")
    p.add_argument("--train_input_mode", default="official_resize",
                    choices=["presized224", "official_resize"],
                    help="official_resize for raw clean-ImageNet JPEGs "
                         "(variable size); presized224 if --train_manifest "
                         "ever points at a pre-cropped set")
    p.add_argument("--backbone_family", default="dinov2", choices=["dinov2", "dinov3"])
    p.add_argument("--backbone_name", default=None)
    p.add_argument("--weights_path", default=None)
    p.add_argument("--head_path", default=None)
    p.add_argument("--population_size", type=int, default=30)
    p.add_argument("--sigma_values", default="0.0005,0.001,0.002")
    p.add_argument("--top_k_ratios", default="0.05,0.1,0.2")
    p.add_argument("--num_engines", type=int, default=1)
    p.add_argument("--perturb_target", default="all",
                    choices=["all", "head", "last_n_blocks"])
    p.add_argument("--last_n_blocks", type=int, default=0)
    p.add_argument("--perturb_steps", type=int, default=1,
                    help="chain P perturbations (same sigma each step) before "
                         "scoring; only the final step of the chain is scored. "
                         "P=1 (default) is the original single-step behavior.")
    p.add_argument("--train_samples", type=int, default=500,
                    help="scoring-set size sampled (class-balanced) from train; "
                         "0 = full split")
    p.add_argument("--test_samples", type=int, default=0, help="0 = full test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    args = p.parse_args()
    args.dataset_targets = _parse_dataset_list(args.dataset)
    args.sigma_list = [float(s) for s in args.sigma_values.split(",")]
    ratios = [float(r) for r in args.top_k_ratios.split(",")]
    args.top_k_list = sorted({max(1, int(r * args.population_size))
                              for r in ratios}, reverse=True)
    args.max_top_k = args.top_k_list[0]
    return args


def _datasets_root():
    return os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")


def load_logit_mask(dataset: str):
    """None (no-op) for datasets with no manifests/_meta/<name>_meta.json at
    all (imagenet, imagenet_c — they predate the shift-dataset prep pipeline
    and were never given one: full 1000-way, correctly unmasked) or an
    explicit subset_indices=None entry (imagenet_sketch); a {0,-inf} mask
    tensor over the class subset otherwise (imagenet_a/r/es)."""
    meta_path = os.path.join(_datasets_root(), "manifests", "_meta",
                              f"{dataset}_meta.json")
    if not os.path.exists(meta_path):
        return None
    from utils.logit_mask import build_mask, load_subset_indices
    subset_indices = load_subset_indices(meta_path)
    if subset_indices is None:
        return None
    return build_mask(subset_indices)


def _as_seed_chain(seed_part):
    """Normalize a perturbation's seed component to a tuple of P seeds,
    applied in sequence and scored only at the end of the chain. Accepts
    either a single int seed (the original single-step format, still used by
    scripts/eval_backbone_on_imagenet.py and pre-existing results.json files)
    or an already-chained sequence -- so run_ensemble_multi works unchanged
    for both single-step and multi-step callers."""
    if isinstance(seed_part, (list, tuple)):
        return tuple(int(s) for s in seed_part)
    return (int(seed_part),)


def run_sampling(args, engines, handler, train_items, logit_mask, wandb_run):
    print(f"\n{'='*60}\nPERTURBATION SAMPLING\n{'='*60}")
    steps = getattr(args, "perturb_steps", 1)
    print(f"Budget: {args.population_size} | Sigmas: {args.sigma_list} | "
          f"perturb_steps: {steps}")

    # same sigma is applied at every step of a chain; only the chain's final
    # state is scored. steps=1 draws the exact same seeds (same rng calls, in
    # the same order) as the original single-step implementation.
    rng = np.random.default_rng(seed=args.global_seed)
    flat_seeds = rng.choice(2**31, size=args.population_size * steps,
                            replace=False).tolist()
    all_seed_chains = [tuple(int(s) for s in flat_seeds[i * steps:(i + 1) * steps])
                       for i in range(args.population_size)]
    all_sigmas = rng.choice(args.sigma_list, size=args.population_size).tolist()

    perf: Dict[Tuple[Tuple[int, ...], float], float] = {}
    done, batch_idx = 0, 0
    while done < args.population_size:
        n = min(args.num_engines, args.population_size - done)
        batch = [(all_seed_chains[done + i], float(all_sigmas[done + i]))
                 for i in range(n)]

        for step in range(steps):
            ray.get([engines[i].perturb_weights.remote(chain[step], sig)
                     for i, (chain, sig) in enumerate(batch)])
        preds = ray.get([engines[i].predict.remote(
                             train_items, args.train_input_mode, logit_mask)
                         for i in range(n)])
        for step in reversed(range(steps)):
            ray.get([engines[i].restore_weights.remote(chain[step], sig)
                     for i, (chain, sig) in enumerate(batch)])

        rewards = [score(handler, preds[i], train_items) for i in range(n)]
        for (chain, sig), r in zip(batch, rewards):
            perf[(chain, sig)] = r
        done += n
        batch_idx += 1
        print(f"  Batch {batch_idx} | {done}/{args.population_size} | "
              f"{['%.3f' % r for r in rewards]}")
        if wandb_run:
            wandb_run.log({"sampling/batch_mean_reward": float(np.mean(rewards)),
                           "sampling/batch_max_reward": float(np.max(rewards)),
                           "sampling/samples_evaluated": done})

    sigma_rewards = {s: [] for s in args.sigma_list}
    for (_, sig), r in perf.items():
        sigma_rewards[sig].append(r)
    for sig in args.sigma_list:
        if sigma_rewards[sig]:
            m = float(np.mean(sigma_rewards[sig]))
            print(f"  σ={sig}: mean={m:.4f}, n={len(sigma_rewards[sig])}")
            if wandb_run:
                wandb_run.log({f"sigma/{sig}/mean_reward": m})
    best_sigma = max(args.sigma_list,
                     key=lambda s: np.mean(sigma_rewards[s]) if sigma_rewards[s] else 0)
    print(f"\n★ Best sigma: {best_sigma}")
    if wandb_run:
        wandb_run.log({"sampling/best_sigma": best_sigma})
    return perf, best_sigma


def run_ensemble_multi(args, engines, top_k_perturbs, eval_sets, wandb_run):
    """Majority-vote ensemble over top_k_perturbs, evaluated on every entry in
    `eval_sets` — a list of dicts {name, items, logit_mask, input_mode, base}.
    Each selected model is perturbed ONCE per batch and predicts on every eval
    set before being restored, so adding eval sets costs one extra forward
    pass per batch, not extra perturb/restore round-trips.
    """
    max_k = min(args.max_top_k, len(top_k_perturbs))
    eval_ks = [k for k in args.top_k_list if k <= max_k]
    print(f"\n{'='*60}\nENSEMBLE EVALUATION\n{'='*60}")
    print(f"K values: {eval_ks} | eval sets: "
          f"{[(e['name'], len(e['items'])) for e in eval_sets]}")

    all_answers = {e["name"]: [None] * max_k for e in eval_sets}
    total_batches = (max_k + args.num_engines - 1) // args.num_engines
    for b in range(total_batches):
        start, end = b * args.num_engines, min((b + 1) * args.num_engines, max_k)
        batch = [(_as_seed_chain(sp), float(sig))
                 for sp, sig in top_k_perturbs[start:end]]
        print(f"  Batch {b + 1}/{total_batches} ({len(batch)} models)...",
              flush=True)
        max_steps = max(len(chain) for chain, _ in batch)
        for step in range(max_steps):
            ray.get([engines[i].perturb_weights.remote(chain[step], sig)
                     for i, (chain, sig) in enumerate(batch) if step < len(chain)])
        for e in eval_sets:
            preds = ray.get([engines[i].predict.remote(
                                 e["items"], e["input_mode"], e["logit_mask"])
                             for i in range(len(batch))])
            for local, global_idx in enumerate(range(start, end)):
                all_answers[e["name"]][global_idx] = preds[local]
        for step in reversed(range(max_steps)):
            ray.get([engines[i].restore_weights.remote(chain[step], sig)
                     for i, (chain, sig) in enumerate(batch) if step < len(chain)])

    results = {}
    for e in eval_sets:
        name, items, base = e["name"], e["items"], e["base"]
        answers = all_answers[name]
        results[name] = {}
        for k in eval_ks:
            correct = 0
            for idx, d in enumerate(items):
                votes = Counter(answers[m][idx] for m in range(k))
                if votes.most_common(1)[0][0] == d["ground_truth"]:
                    correct += 1
            acc = correct / len(items) * 100
            results[name][k] = {"accuracy": acc, "correct": correct}
            print(f"  [{name}] K={k}: {acc:.2f}% ({correct}/{len(items)}) "
                  f"[{acc - base*100:+.2f}% vs base]")
            if wandb_run:
                wandb_run.log({f"ensemble/{name}/k{k}/accuracy": acc,
                               f"ensemble/{name}/k{k}/gain_over_base": acc - base*100})
    return results


def main(args):
    fam = "" if args.backbone_family == "dinov2" else f"-{args.backbone_family}"
    dataset_targets = args.dataset_targets
    dataset_tag = "-".join(dataset_targets)
    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_name or (
            f"randopt-ssl{fam}-{dataset_tag}-N{args.population_size}")
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config=vars(args))

    from data_handlers import get_dataset_handler
    # any subclass's compute_reward/extract_answer is identical (schema-driven,
    # not class-specific) — used for clean-ImageNet scoring + reward compute
    handler = get_dataset_handler(dataset_targets[0])
    print(f"train_manifest (clean ImageNet, unmasked): {args.train_manifest}")

    train_items = handler.load_data(args.train_manifest, split="train")
    rng = np.random.default_rng(args.global_seed)
    train_items = sample_per_class(train_items, args.train_samples, rng)
    print(f"scoring set (clean ImageNet): {len(train_items)}")

    # One eval-set entry per --dataset target, all resolved and loaded the
    # same way via _DATASET_REGISTRY (manifest + which split selects its
    # eval rows) — so base-eval and ensemble-eval below need no per-dataset
    # special case, whether the target is a shift dataset, imagenet_c, or
    # clean imagenet itself.
    targets = {}
    for name in dataset_targets:
        reg = _DATASET_REGISTRY[name]
        h = get_dataset_handler(name)
        # a single --test_manifest override only makes sense with exactly
        # one target dataset — with multiple targets there's no single path
        # that could apply to all of them, so it's ignored then
        manifest = (args.test_manifest if (len(dataset_targets) == 1 and args.test_manifest)
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

    # Ray's default /tmp spill dir sits on Geoffry's root disk (~4GB free,
    # 99% full per CLAUDE.md's server table) — point it at $DATASETS_ROOT
    # (1.8TB free) so a larger run than this smoke test doesn't risk a
    # spill failure (observed "over 95% full" warnings during dev, harmless
    # only because this run's data was tiny).
    ray_tmp = os.path.join(_datasets_root(), "ray_tmp")
    os.makedirs(ray_tmp, exist_ok=True)
    ray.init(ignore_reinit_error=True, include_dashboard=False, _temp_dir=ray_tmp)
    from vision import launch_ssl_engines
    engines = launch_ssl_engines(args.num_engines,
                                 backbone_family=args.backbone_family,
                                 backbone_name=args.backbone_name,
                                 weights_path=args.weights_path,
                                 head_path=args.head_path,
                                 perturb_target=args.perturb_target,
                                 last_n_blocks=args.last_n_blocks,
                                 inference_batch_size=args.batch_size)
    n_scope = ray.get(engines[0].count_perturb_params.remote())
    print(f"perturb scope: {args.perturb_target} = {n_scope/1e6:.1f}M params")

    t0 = time.time()
    base_train = score(handler,
                       ray.get(engines[0].predict.remote(
                           train_items, args.train_input_mode, None)),  # clean ImageNet: never masked
                       train_items)
    base_by_name = {}
    for name, info in targets.items():
        base_by_name[name] = score(info["handler"],
                                   ray.get(engines[0].predict.remote(
                                       info["items"], info["input_mode"], info["mask"])),
                                   info["items"])
    print(f"BASE: train_reward={base_train:.4f} (clean ImageNet) | "
          + " | ".join(f"{name}={acc:.4f}" for name, acc in base_by_name.items())
          + f" ({time.time()-t0:.0f}s)")
    if wandb_run:
        log = {"base/train_reward": base_train}
        log.update({f"base/{name}_test_accuracy": acc for name, acc in base_by_name.items()})
        wandb_run.log(log)

    # scoring is always on clean ImageNet -> logit_mask=None, regardless of
    # which shift dataset the final eval below targets
    perf, best_sigma = run_sampling(args, engines, handler, train_items,
                                    None, wandb_run)

    ray.get([e.reset_to_base_weights.remote() for e in engines])

    top = sorted(perf.items(), key=lambda kv: kv[1], reverse=True)
    top_k_perturbs = [k for k, _ in top[:args.max_top_k]]
    top_k_rewards = [v for _, v in top[:args.max_top_k]]
    print(f"top-{args.max_top_k} train rewards (clean ImageNet): "
          f"{['%.3f' % r for r in top_k_rewards[:5]]}...")

    # only the final eval touches any --dataset target; every eval set is
    # perturbed only ONCE per ensemble batch regardless of how many targets
    eval_sets = [{"name": name, "items": info["items"], "logit_mask": info["mask"],
                  "input_mode": info["input_mode"], "base": base_by_name[name]}
                 for name, info in targets.items()]
    ensemble = run_ensemble_multi(args, engines, top_k_perturbs, eval_sets, wandb_run)

    exp_dir = args.experiment_dir or (
        f"results/randopt-ssl{fam}-{dataset_tag}-N{args.population_size}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({
            "dataset": dataset_targets,
            "base_train_reward": base_train,
            "base_test_accuracy": base_by_name,
            "best_sigma": best_sigma,
            "ensemble_results": {
                name: {str(k): v for k, v in per_k.items()}
                for name, per_k in ensemble.items()
            },
            "top_k_perturbs": top_k_perturbs,
            "top_k_train_rewards": top_k_rewards,
            "config": vars(args),
        }, f, indent=2)
    print(f"Saved {exp_dir}/results.json")
    if wandb_run:
        wandb_run.finish()
    ray.shutdown()


if __name__ == "__main__":
    main(parse_args())
