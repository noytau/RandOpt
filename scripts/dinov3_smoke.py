"""DINOv3-7B compatibility smoke + base eval (RunAI, 1 GPU).

Verifies, in order, everything the RandOpt loop assumes about a new backbone
family — and pushes every number to W&B because the cluster-api outage blocks
job logs (job status alone says pass/fail; W&B carries the values):

  1. engine constructs (hub weights=<path> strict load, head strict load)
  2. forward_features contract (x_norm_clstoken / x_norm_patchtokens)
  3. blocks.{i} naming -> last_n_blocks scope counts params
  4. perturb -> restore round-trip drift is ~1 ulp at tiny sigma
  5. base accuracy: clean train manifest + IC test manifest (subsampled)
     -- clean should land near the published linear-probe figure
  6. timings per phase (predict img/s, perturb+restore secs) for job sizing
"""
import argparse
import os
import sys
import time

import numpy as np
import ray

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", default="data/imagenet/data.json")
    p.add_argument("--test_manifest", default="data/imagenet_c/data.json")
    p.add_argument("--eval_samples", type=int, default=2000)
    p.add_argument("--sigma_probe", type=float, default=1e-4)
    p.add_argument("--wandb_project", default="randopt")
    return p.parse_args()


def main(args):
    import wandb
    run = wandb.init(project=args.wandb_project, name="dinov3-smoke",
                     config=vars(args))

    from data_handlers import get_dataset_handler
    handler = get_dataset_handler("imagenet_c")
    rng = np.random.default_rng(42)
    train = handler.load_data(args.train_manifest, split="train")
    test = handler.load_data(args.test_manifest, split="test")
    train = [train[i] for i in rng.choice(len(train), args.eval_samples,
                                          replace=False)]
    test = [test[i] for i in rng.choice(len(test), args.eval_samples,
                                        replace=False)]

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    from vision import launch_ssl_engines
    t0 = time.time()
    engines = launch_ssl_engines(1, backbone_family="dinov3")
    e = engines[0]
    load_s = time.time() - t0

    n_all = ray.get(e.count_perturb_params.remote())
    ray.get(e.set_perturb_scope.remote("last_n_blocks", 4))
    n_last4 = ray.get(e.count_perturb_params.remote())
    assert n_last4 > 0, "last_n_blocks scope found no params — blocks naming?"
    ray.get(e.set_perturb_scope.remote("all", 0))

    # perturb/restore round trip at tiny sigma: predictions must be identical
    t0 = time.time()
    base_preds = ray.get(e.predict.remote(test[:200], "presized224"))
    ray.get(e.perturb_weights.remote(123, args.sigma_probe))
    ray.get(e.restore_weights.remote(123, args.sigma_probe))
    roundtrip_preds = ray.get(e.predict.remote(test[:200], "presized224"))
    drift_flips = sum(a != b for a, b in zip(base_preds, roundtrip_preds))
    perturb_s = time.time() - t0

    def acc(preds, items):
        return float(np.mean([handler.compute_reward(p, d["ground_truth"])
                              for p, d in zip(preds, items)]))

    t0 = time.time()
    clean_acc = acc(ray.get(e.predict.remote(train, "official_resize")), train)
    clean_s = time.time() - t0
    t0 = time.time()
    ic_acc = acc(ray.get(e.predict.remote(test, "presized224")), test)
    ic_s = time.time() - t0

    metrics = {
        "smoke/engine_load_s": load_s,
        "smoke/params_all_M": n_all / 1e6,
        "smoke/params_last4_M": n_last4 / 1e6,
        "smoke/roundtrip_pred_flips": drift_flips,
        "smoke/perturb_restore_plus_2x200pred_s": perturb_s,
        "smoke/base_clean_acc": clean_acc,
        "smoke/base_ic_acc": ic_acc,
        "smoke/clean_imgs_per_s": len(train) / clean_s,
        "smoke/ic_imgs_per_s": len(test) / ic_s,
    }
    print(metrics)
    run.log(metrics)
    run.finish()

    assert drift_flips == 0, f"{drift_flips}/200 predictions changed after restore"
    assert clean_acc > 0.80, f"clean acc {clean_acc} — head/transform broken?"
    ray.shutdown()


if __name__ == "__main__":
    main(parse_args())
