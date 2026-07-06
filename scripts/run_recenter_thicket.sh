#!/bin/bash
# E4: thicket-emergence curve — run the E1 thicket protocol at several points
# along the gradient adaptation trajectory (0/50/150/300 steps). Tests whether
# task-adaptation moves DINOv2 from a needle regime into a thicket.
# See scripts/recenter_thicket.py docstring.
set -eo pipefail
cd /storage/noy/RandOpt
if [ "${SKIP_SYNC:-0}" != "1" ]; then
  for _i in $(seq 1 30); do [ -f .git/index.lock ] && sleep 2 || break; done
  git fetch origin feature/vision-randopt -q && git reset --hard origin/feature/vision-randopt -q
fi
pip install transformers torchvision Pillow --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
mkdir -p results

LEVELS="${LEVELS:-0,50,150,300}"
SIGMAS="${SIGMAS:-0.0003,0.001,0.003,0.01}"
NPOP="${NPOP:-300}"
LR="${LR:-3e-5}"
NTRAIN="${NTRAIN:-800}"
NA="${NA:-200}"
NB="${NB:-200}"
BATCH="${BATCH:-64}"
SEED="${SEED:-42}"
RUNNAME="${RUNNAME:-recenter-thicket}"

python3 -u scripts/recenter_thicket.py \
    --model_name facebook/dinov2-base \
    --scope last1 \
    --levels "$LEVELS" \
    --sigmas "$SIGMAS" \
    --npop "$NPOP" \
    --lr "$LR" \
    --n_train "$NTRAIN" \
    --nA "$NA" --nB "$NB" \
    --inference_batch_size "$BATCH" \
    --global_seed "$SEED" \
    --experiment_dir "results/${RUNNAME}" \
    --wandb_project randopt \
    --wandb_run_name "$RUNNAME" \
    2>&1 | tee "results/${RUNNAME}.log"
