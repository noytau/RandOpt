#!/bin/bash
# E2: gradient-reachability control (plateau vs needle) on SPair-71k, DINOv2, no head.
# Gradient ascent on a differentiable PCK surrogate under the SAME scope/eval/A-B
# protocol as the thicket sweep. See scripts/grad_reachability.py docstring.
set -eo pipefail
cd /storage/noy/RandOpt
# Concurrency-safe code sync: wait out any other job's git lock; skip entirely
# with SKIP_SYNC=1 when the PVC is already up to date.
if [ "${SKIP_SYNC:-0}" != "1" ]; then
  for _i in $(seq 1 30); do [ -f .git/index.lock ] && sleep 2 || break; done
  git fetch origin feature/vision-randopt -q && git reset --hard origin/feature/vision-randopt -q
fi
pip install transformers torchvision Pillow --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
mkdir -p results

# Tunables (override via `runai submit -e KEY=VAL`)
SCOPE="${SCOPE:-last1}"
LOSSES="${LOSSES:-softargmax,infonce}"
LRS="${LRS:-3e-6,3e-5,3e-4}"
STEPS="${STEPS:-300}"
EVAL_EVERY="${EVAL_EVERY:-25}"
NTRAIN="${NTRAIN:-800}"
NA="${NA:-250}"
NB="${NB:-250}"
PAIRS="${PAIRS:-8}"
TAU="${TAU:-0.05}"
BATCH="${BATCH:-64}"
SEED="${SEED:-42}"
RUNNAME="${RUNNAME:-grad-reach-${SCOPE}}"

# -u = unbuffered stdout so logs stream live through tee
python3 -u scripts/grad_reachability.py \
    --dataset spair71k \
    --model_name facebook/dinov2-base \
    --scope "$SCOPE" \
    --losses "$LOSSES" \
    --lrs "$LRS" \
    --steps "$STEPS" \
    --eval_every "$EVAL_EVERY" \
    --n_train "$NTRAIN" \
    --nA "$NA" --nB "$NB" \
    --pairs_per_step "$PAIRS" \
    --tau "$TAU" \
    --inference_batch_size "$BATCH" \
    --global_seed "$SEED" \
    --experiment_dir "results/${RUNNAME}" \
    --wandb_project randopt \
    --wandb_run_name "$RUNNAME" \
    2>&1 | tee "results/${RUNNAME}.log"
