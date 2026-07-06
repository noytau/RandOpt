#!/bin/bash
# E3: multi-task thicket profile of the DINOv2 SSL neighborhood.
# Scores each random perturbation on a panel of tasks (PCK + kNN/mAP on
# CUB-200 and FGVC-Aircraft) under the sweep's A/B honesty protocol.
# See scripts/ssl_thicket_profile.py docstring for hypotheses/decision rules.
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
CELLS="${CELLS:-last1:0.0003,last1:0.001,last1:0.003,last1:0.01,last2:0.0003,last2:0.001,all:0.0001,all:0.0003,all:0.001}"
TASKS="${TASKS:-pck,cub,air}"
NPOP="${NPOP:-300}"
SP_NA="${SP_NA:-200}"
SP_NB="${SP_NB:-200}"
GPC="${GPC:-3}"
NQ="${NQ:-500}"
BATCH="${BATCH:-64}"
SEED="${SEED:-42}"
RUNNAME="${RUNNAME:-ssl-profile-v1}"

# -u = unbuffered stdout so logs stream live through tee
python3 -u scripts/ssl_thicket_profile.py \
    --model_name facebook/dinov2-base \
    --cells "$CELLS" \
    --tasks "$TASKS" \
    --population_size "$NPOP" \
    --sp_nA "$SP_NA" --sp_nB "$SP_NB" \
    --gallery_per_class "$GPC" \
    --n_queries "$NQ" \
    --inference_batch_size "$BATCH" \
    --global_seed "$SEED" \
    --experiment_dir "results/${RUNNAME}" \
    --wandb_project randopt \
    --wandb_run_name "$RUNNAME" \
    2>&1 | tee "results/${RUNNAME}.log"
