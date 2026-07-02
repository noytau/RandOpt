"""End-to-end smoke test for the correspondence pipeline.

Tests: DINOv2 load, get_patch_features, compute_pck, perturb/restore round-trip.
N=5 perturbations, 20 train pairs, 20 test pairs. Runtime: ~3-5 min.
"""
import os
import sys
import time
import numpy as np
import ray
import torch

sys.path.insert(0, "/storage/noy/RandOpt")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/storage/noy/.cache/huggingface"

from data_handlers.spair71k import SPair71kHandler, compute_pck, GRID
from vision import launch_vision_engines

DATA_DIR = "data/spair71k"
MODEL    = "facebook/dinov2-base"
N        = 5
PAIRS    = 20

print("=== SPair-71k Correspondence Smoke Test ===")
print(f"Model: {MODEL} | N={N} perturbations | {PAIRS} pairs\n")

# --- Data ---
print("[1/5] Loading data...")
t0 = time.time()
handler = SPair71kHandler()
train_data = handler.load_data(DATA_DIR, split="train", max_samples=PAIRS)
test_data  = handler.load_data(DATA_DIR, split="test",  max_samples=PAIRS)
print(f"  Train: {len(train_data)} pairs | Test: {len(test_data)} pairs ({time.time()-t0:.1f}s)")

# --- Engine ---
print("[2/5] Launching DINOv2 engine...")
t0 = time.time()
ray.init(ignore_reinit_error=True)
engines = launch_vision_engines(
    num_engines=1, model_name=MODEL,
    num_classes=1, linear_init_path=None, perturb_target="all",
)
print(f"  Engine ready ({time.time()-t0:.1f}s)")

# --- Patch features ---
print("[3/5] Base model PCK...")
t0 = time.time()
src_imgs = torch.stack([d["image_tensor"]     for d in train_data])
tgt_imgs = torch.stack([d["image_tensor_tgt"] for d in train_data])
sf = ray.get(engines[0].get_patch_features.remote(src_imgs))  # (N, 256, 768)
tf = ray.get(engines[0].get_patch_features.remote(tgt_imgs))
assert sf.shape == (PAIRS, GRID*GRID, 768), f"Unexpected shape: {sf.shape}"

pcks = [compute_pck(sf[i], tf[i], train_data[i]["kpts_src"],
                    train_data[i]["kpts_tgt"], train_data[i]["bbox_thresh"])
        for i in range(len(train_data))]
base_pck = float(np.mean(pcks))
print(f"  Base train PCK@0.1: {base_pck*100:.2f}% ({time.time()-t0:.1f}s)")
assert 0.3 < base_pck < 0.95, f"Base PCK out of expected range: {base_pck:.3f}"

# --- Perturb/restore round-trip ---
print("[4/5] Perturb / restore round-trip...")
t0 = time.time()
sf_base = ray.get(engines[0].get_patch_features.remote(src_imgs[:4]))
ray.get(engines[0].perturb_weights.remote(42, 0.0001))
sf_perturbed = ray.get(engines[0].get_patch_features.remote(src_imgs[:4]))
ray.get(engines[0].restore_weights.remote(42, 0.0001))
sf_restored = ray.get(engines[0].get_patch_features.remote(src_imgs[:4]))
max_diff = (sf_base - sf_restored).abs().max().item()
print(f"  Max weight restore diff: {max_diff:.2e} (should be ~0)")
assert max_diff < 1e-4, f"Restore not exact: {max_diff}"
perturb_diff = (sf_base - sf_perturbed).abs().max().item()
print(f"  Perturbation changed features by: {perturb_diff:.4f} (should be > 0)")
assert perturb_diff > 1e-6, "Perturbation had no effect!"
print(f"  Round-trip OK ({time.time()-t0:.1f}s)")

# --- Mini sampling loop ---
print(f"[5/5] Sampling {N} perturbations...")
t0 = time.time()
sigmas = [0.00001, 0.0001, 0.001]
rng = np.random.default_rng(42)
seeds  = rng.integers(0, 2**31, size=N)
sigs   = rng.choice(sigmas, size=N)
results = []
for i, (seed, sigma) in enumerate(zip(seeds, sigs)):
    ray.get(engines[0].perturb_weights.remote(int(seed), float(sigma)))
    sf_p = ray.get(engines[0].get_patch_features.remote(src_imgs))
    tf_p = ray.get(engines[0].get_patch_features.remote(tgt_imgs))
    pck = float(np.mean([compute_pck(sf_p[j], tf_p[j],
                         train_data[j]["kpts_src"], train_data[j]["kpts_tgt"],
                         train_data[j]["bbox_thresh"]) for j in range(len(train_data))]))
    ray.get(engines[0].restore_weights.remote(int(seed), float(sigma)))
    results.append((seed, sigma, pck))
    print(f"  [{i+1}/{N}] seed={seed} σ={sigma:.5f} PCK={pck*100:.2f}%")

best = max(results, key=lambda x: x[2])
print(f"\nBest perturbation: seed={best[0]} σ={best[1]} PCK={best[2]*100:.2f}% ({time.time()-t0:.1f}s)")
print(f"\n=== SMOKE TEST PASSED ===")
print(f"Base PCK: {base_pck*100:.2f}% | Best perturb PCK: {best[2]*100:.2f}%")
