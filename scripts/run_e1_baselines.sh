#!/bin/bash
# E1 (experiments/E1_imagenet_c.md): ImageNet-C benchmark rungs on DINOv2-giant.
# One entry point for all tasks, selected via TASK=knn|probe|ft|randopt.
# Parametrized via env vars so jobs stay a short `runai submit -e KEY=VAL`.
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
TASK="${TASK:-knn}"
MODEL="${MODEL:-facebook/dinov2-giant}"
CORRUPTION="${CORRUPTION:-gaussian_noise}"
SEVERITY="${SEVERITY:-3}"
BATCH="${BATCH:-32}"
RUNNAME="${RUNNAME:-e1-${TASK}-${CORRUPTION}-s${SEVERITY}}"
EXTRA_ARGS="${EXTRA_ARGS:-}"   # free-form passthrough, e.g. "--sweep_k 1,3,5,10,20,50"

case "$TASK" in
  knn)     SCRIPT=scripts/knn_imagenet_c.py;     BATCH_FLAG=--batch_size ;;
  probe)   SCRIPT=scripts/probe_imagenet_c.py;   BATCH_FLAG=--batch_size ;;
  ft)      SCRIPT=scripts/finetune_imagenet_c.py; BATCH_FLAG=--batch_size ;;
  randopt) SCRIPT=scripts/randopt_imagenet_c.py; BATCH_FLAG=--inference_batch_size ;;
  *) echo "unknown TASK '$TASK' (knn|probe|ft|randopt)"; exit 1 ;;
esac

# -u = unbuffered stdout so logs stream live through tee
python3 -u "$SCRIPT" \
    --model_name "$MODEL" \
    --corruption "$CORRUPTION" \
    --severity "$SEVERITY" \
    "$BATCH_FLAG" "$BATCH" \
    --experiment_dir "results/${RUNNAME}" \
    --wandb_project randopt \
    --wandb_run_name "$RUNNAME" \
    $EXTRA_ARGS \
    2>&1 | tee "results/${RUNNAME}.log"
