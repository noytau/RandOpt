#!/usr/bin/env bash
# SSL twin of local_run.sh: full local RandOpt evaluation on ImageNet-C with
# DINOv2 (vision.SSLEngine). Same knob style (env-overridable), sizes scaled
# to ViT-g forward cost on 2080 Tis — the LLM script's N=5000 would be ~5 GPU-
# days here; N=200 is the full local tier (see runtime notes per arg below).
#
#   CUDA_DEVICES=1,2,4,5 POPULATION=100 bash scripts/local_run_ssl.sh
set -euo pipefail

cd "$(dirname "$0")/.."

# Geoffry: GPU 3 is DEAD — never include index 3 (CUDA_DEVICE_ORDER=PCI_BUS_ID
# comes from randopt_env.sh; source it before running).
CUDA_DEVICES="${CUDA_DEVICES:-1,2}"
MANIFEST="${MANIFEST:-data/imagenet_c/data.json}"
# scoring/test may come from different datasets (e.g. clean-ImageNet scoring,
# ImageNet-C test); both default to MANIFEST for the classic single-set run
TRAIN_MANIFEST="${TRAIN_MANIFEST:-$MANIFEST}"
TEST_MANIFEST="${TEST_MANIFEST:-$MANIFEST}"

POPULATION="${POPULATION:-200}"
SIGMAS="${SIGMAS:-0.0001,0.0002,0.0005,0.001}"
TOP_K_RATIOS="${TOP_K_RATIOS:-0.01,0.05,0.1}"
TRAIN_SAMPLES="${TRAIN_SAMPLES:-500}"
TEST_SAMPLES="${TEST_SAMPLES:-0}"          # 0 = full 15k test split
PERTURB_TARGET="${PERTURB_TARGET:-all}"
LAST_N_BLOCKS="${LAST_N_BLOCKS:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-randopt}"

NUM_ENGINES="$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

# distinct results dir + W&B run name per perturbation scope, so scope
# comparisons never collide
SCOPE_TAG="$PERTURB_TARGET"
[ "$PERTURB_TARGET" = "last_n_blocks" ] && SCOPE_TAG="last${LAST_N_BLOCKS}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:-results/randopt-ssl-local-N${POPULATION}-${SCOPE_TAG}}"

python3 -u scripts/randopt_imagenet_c.py \
  --manifest "$MANIFEST" \
  --train_manifest "$TRAIN_MANIFEST" \
  --test_manifest "$TEST_MANIFEST" \
  --population_size "$POPULATION" \
  --sigma_values "$SIGMAS" \
  --top_k_ratios "$TOP_K_RATIOS" \
  --train_samples "$TRAIN_SAMPLES" \
  --test_samples "$TEST_SAMPLES" \
  --num_engines "$NUM_ENGINES" \
  --perturb_target "$PERTURB_TARGET" \
  --last_n_blocks "$LAST_N_BLOCKS" \
  --global_seed 42 \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "randopt-ssl-${SCOPE_TAG}-N${POPULATION}" \
  --experiment_dir "$EXPERIMENT_DIR"
