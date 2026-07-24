"""RandOpt on ImageNet-C with an SSL model (DINOv2) — the vision twin of
randopt.py: identical loop (base eval -> perturbation sampling -> top-K
selection -> majority-vote ensemble), identical W&B keys, with vLLM engines
swapped for vision.SSLEngine actors.

Simplifications inherited from classification (vs the LLM path):
  - no sampling params (temperature/max_tokens) — inference is one forward
  - scoring = handler.compute_reward on label strings = plain accuracy
  - ensemble votes need no answer extraction — a predicted label IS its vote

Drift hygiene: engines restore after every perturbation (near-exact, ~1e-7);
reset_to_base_weights() runs between phases for a bit-exact clean slate.
"""
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default="data/imagenet_c/data.json")
    p.add_argument("--train_manifest", default=None,
                   help="manifest for the perturbation-scoring set (its 'train' "
                        "split); defaults to --manifest. Lets scoring run on a "
                        "different dataset than the final test, e.g. clean "
                        "ImageNet train / ImageNet-C test")
    p.add_argument("--test_manifest", default=None,
                   help="manifest for the ensemble test set (its 'test' "
                        "split); defaults to --manifest")
    p.add_argument("--train_input_mode", default="presized224",
                   choices=["presized224", "official_resize"],
                   help="transform for scoring images: presized224 = normalize "
                        "only (ImageNet-C protocol), official_resize = Resize "
                        "256 -> CenterCrop 224 (raw clean-ImageNet JPEGs)")
    p.add_argument("--test_input_mode", default="presized224",
                   choices=["presized224", "official_resize"])
    p.add_argument("--backbone_family", default="dinov2",
                   choices=["dinov2", "dinov3"])
    p.add_argument("--backbone_name", default=None,
                   help="hub entrypoint; default = family default "
                        "(dinov2_vitg14_reg / dinov3_vit7b16)")
    p.add_argument("--weights_path", default=None,
                   help="gated backbone .pth (dinov3); default = ingest path")
    p.add_argument("--head_path", default=None,
                   help="classifier head .pth (dinov3); default = ingest path")
    p.add_argument("--population_size", type=int, default=30)
    p.add_argument("--sigma_values", default="0.0005,0.001,0.002")
    p.add_argument("--top_k_ratios", default="0.05,0.1,0.2")
    p.add_argument("--num_engines", type=int, default=1)
    p.add_argument("--perturb_target", default="all",
                   choices=["all", "head", "last_n_blocks"])
    p.add_argument("--last_n_blocks", type=int, default=0)
    p.add_argument("--train_samples", type=int, default=500,
                   help="scoring-set size sampled from the train split "
                        "(25k full = ~10min/perturbation on a 2080 Ti)")
    p.add_argument("--test_samples", type=int, default=0, help="0 = full test")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--global_seed", type=int, default=42)
    p.add_argument("--experiment_dir", default=None)
    p.add_argument("--wandb_project", default="randopt")
    p.add_argument("--wandb_name", default=None)
    args = p.parse_args()
    args.train_manifest = args.train_manifest or args.manifest
    args.test_manifest = args.test_manifest or args.manifest
    args.sigma_list = [float(s) for s in args.sigma_values.split(",")]
    ratios = [float(r) for r in args.top_k_ratios.split(",")]
    args.top_k_list = sorted({max(1, int(r * args.population_size))
                              for r in ratios}, reverse=True)
    args.max_top_k = args.top_k_list[0]
    return args


def sample_per_class(items: List[Dict], n_total: int, rng) -> List[Dict]:
    """Class-balanced random subsample: n_total/n_classes items drawn
    randomly from EVERY class, so any selected set represents all 1k
    classes equally. n_total of 0 (or >= len) keeps the full set."""
    if not n_total or n_total >= len(items):
        return items
    by_class = defaultdict(list)
    for d in items:
        by_class[d["class_id"]].append(d)
    per = max(1, n_total // len(by_class))
    picked = []
    for c in sorted(by_class):
        pool = by_class[c]
        idx = rng.choice(len(pool), size=min(per, len(pool)), replace=False)
        picked.extend(pool[i] for i in idx)
    return picked


def score(handler, preds: List[str], items: List[Dict]) -> float:
    """Mean reward over predictions (= accuracy for exact-match rewards)."""
    return float(np.mean([handler.compute_reward(p, d["ground_truth"])
                          for p, d in zip(preds, items)]))


def run_sampling(args, engines, handler, train_items, wandb_run):
    """Mirror of randopt.py::run_sampling — engine-parallel perturbation
    scoring on the train split."""
    print(f"\n{'='*60}\nPERTURBATION SAMPLING\n{'='*60}")
    print(f"Budget: {args.population_size} | Sigmas: {args.sigma_list}")

    rng = np.random.default_rng(seed=args.global_seed)
    all_seeds = rng.choice(2**31, size=args.population_size, replace=False).tolist()
    all_sigmas = rng.choice(args.sigma_list, size=args.population_size).tolist()

    perf: Dict[Tuple[int, float], float] = {}
    done, batch_idx = 0, 0
    while done < args.population_size:
        n = min(args.num_engines, args.population_size - done)
        batch = [(int(all_seeds[done + i]), float(all_sigmas[done + i]))
                 for i in range(n)]

        ray.get([engines[i].perturb_weights.remote(s, sig)
                 for i, (s, sig) in enumerate(batch)])
        preds = ray.get([engines[i].predict.remote(train_items,
                                                   args.train_input_mode)
                         for i in range(n)])
        ray.get([engines[i].restore_weights.remote(s, sig)
                 for i, (s, sig) in enumerate(batch)])

        rewards = [score(handler, preds[i], train_items) for i in range(n)]
        for (s, sig), r in zip(batch, rewards):
            perf[(s, sig)] = r
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


def run_ensemble(args, engines, handler, test_items, top_k_perturbs,
                 base_test, wandb_run):
    """Mirror of randopt.py::run_ensemble_evaluation — per-model test
    predictions, then majority vote per K. Labels vote as-is (no extraction)."""
    max_k = min(args.max_top_k, len(top_k_perturbs))
    eval_ks = [k for k in args.top_k_list if k <= max_k]
    print(f"\n{'='*60}\nENSEMBLE EVALUATION\n{'='*60}")
    print(f"K values: {eval_ks} | Test samples: {len(test_items)}")

    all_answers: List[List[str]] = [None] * max_k
    total_batches = (max_k + args.num_engines - 1) // args.num_engines
    for b in range(total_batches):
        start, end = b * args.num_engines, min((b + 1) * args.num_engines, max_k)
        batch = top_k_perturbs[start:end]
        print(f"  Batch {b + 1}/{total_batches} ({len(batch)} models)...",
              flush=True)
        ray.get([engines[i].perturb_weights.remote(s, sig)
                 for i, (s, sig) in enumerate(batch)])
        preds = ray.get([engines[i].predict.remote(test_items,
                                                   args.test_input_mode)
                         for i in range(len(batch))])
        ray.get([engines[i].restore_weights.remote(s, sig)
                 for i, (s, sig) in enumerate(batch)])
        for local, global_idx in enumerate(range(start, end)):
            all_answers[global_idx] = preds[local]

    results = {}
    for k in eval_ks:
        correct = 0
        for idx, d in enumerate(test_items):
            votes = Counter(all_answers[m][idx] for m in range(k))
            if votes.most_common(1)[0][0] == d["ground_truth"]:
                correct += 1
        acc = correct / len(test_items) * 100
        results[k] = {"accuracy": acc, "correct": correct}
        print(f"  K={k}: {acc:.2f}% ({correct}/{len(test_items)}) "
              f"[{acc - base_test*100:+.2f}% vs base]")
        if wandb_run:
            wandb_run.log({f"ensemble/k{k}/accuracy": acc,
                           f"ensemble/k{k}/gain_over_base": acc - base_test*100})
    return results


def main(args):
    fam = "" if args.backbone_family == "dinov2" else f"-{args.backbone_family}"
    wandb_run = None
    if args.wandb_project:
        import wandb
        name = args.wandb_name or (
            f"randopt-ssl{fam}-imagenet-c-N{args.population_size}")
        wandb_run = wandb.init(project=args.wandb_project, name=name,
                               config=vars(args))

    from data_handlers import get_dataset_handler
    handler = get_dataset_handler("imagenet_c")
    train_items = handler.load_data(args.train_manifest, split="train")
    test_items = handler.load_data(args.test_manifest, split="test")
    if args.train_manifest != args.test_manifest:
        print(f"scoring manifest: {args.train_manifest}\n"
              f"test manifest:    {args.test_manifest}")
    rng = np.random.default_rng(args.global_seed)
    train_items = sample_per_class(train_items, args.train_samples, rng)
    test_items = sample_per_class(test_items, args.test_samples, rng)
    print(f"scoring set: {len(train_items)} | test: {len(test_items)} "
          f"(class-balanced)")

    ray.init(ignore_reinit_error=True, include_dashboard=False)
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

    # base model (the anchor every gain is measured against)
    t0 = time.time()
    base_train = score(handler,
                       ray.get(engines[0].predict.remote(
                           train_items, args.train_input_mode)), train_items)
    base_test = score(handler,
                      ray.get(engines[0].predict.remote(
                          test_items, args.test_input_mode)), test_items)
    print(f"BASE: train_reward={base_train:.4f} test_accuracy={base_test:.4f} "
          f"({time.time()-t0:.0f}s)")
    if wandb_run:
        wandb_run.log({"base/train_reward": base_train,
                       "base/test_accuracy": base_test})

    perf, best_sigma = run_sampling(args, engines, handler, train_items, wandb_run)

    # bit-exact clean slate before the ensemble phase
    ray.get([e.reset_to_base_weights.remote() for e in engines])

    top = sorted(perf.items(), key=lambda kv: kv[1], reverse=True)
    top_k_perturbs = [k for k, _ in top[:args.max_top_k]]
    top_k_rewards = [v for _, v in top[:args.max_top_k]]
    print(f"top-{args.max_top_k} train rewards: "
          f"{['%.3f' % r for r in top_k_rewards[:5]]}...")

    ensemble = run_ensemble(args, engines, handler, test_items, top_k_perturbs,
                            base_test, wandb_run)

    exp_dir = args.experiment_dir or (
        f"results/randopt-ssl{fam}-imagenet-c-N{args.population_size}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump({
            "base_train_reward": base_train,
            "base_test_accuracy": base_test,
            "best_sigma": best_sigma,
            "ensemble_results": {str(k): v for k, v in ensemble.items()},
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
