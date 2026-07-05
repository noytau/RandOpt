#!/bin/bash
# E1: individual-gain / thicket-existence study on SPair-71k (DINOv2, no head).
# Scans scope x sigma, reports best held-out individual gain over base.
# Parametrized via env vars so multiple jobs can fan out across GPUs.
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
SCOPES="${SCOPES:-all,last2,last1}"
NPOP="${NPOP:-60}"
SIGMAS="${SIGMAS:-0.00003,0.0001,0.0003,0.001,0.003}"
NA="${NA:-200}"
NB="${NB:-200}"
BATCH="${BATCH:-64}"
SEED="${SEED:-42}"
RUNNAME="${RUNNAME:-thicket-spair71k-v1}"

# -u = unbuffered stdout so logs stream live through tee
python3 -u scripts/randopt_corr_thicket.py \
    --dataset spair71k \
    --model_name facebook/dinov2-base \
    --nA "$NA" --nB "$NB" \
    --population_size "$NPOP" \
    --sigma_values "$SIGMAS" \
    --scopes "$SCOPES" \
    --inference_batch_size "$BATCH" \
    --global_seed "$SEED" \
    --experiment_dir "results/corr_thicket_${RUNNAME}" \
    --wandb_project randopt \
    --wandb_run_name "$RUNNAME" \
    2>&1 | tee "results/${RUNNAME}.log"
