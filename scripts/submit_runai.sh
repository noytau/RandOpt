#!/bin/bash
# Submit a RandOpt experiment to the RunAI cluster.
#
# Prerequisites:
#   1. Code on Lustre PVC at /storage/noy/RandOpt/ (push to GitHub, then update_cluster_code.sh)
#   2. Image noyhassid/randopt-vllm:v1 built and pushed:
#        cd ~/PycharmProjects/RandOpt
#        docker build -f docker/Dockerfile_vllm -t noyhassid/randopt-vllm:v1 .
#        docker push noyhassid/randopt-vllm:v1
#   3. W&B key stored on Lustre: echo "KEY" > /storage/noy/.wandb_api_key
#   4. Data downloaded to /storage/noy/RandOpt/data/
#
# Usage:
#   bash scripts/submit_runai.sh --dataset countdown --model Qwen/Qwen2.5-3B-Instruct --gpus 4
#   bash scripts/submit_runai.sh --dataset gsm8k --model Qwen/Qwen2.5-7B-Instruct --gpus 8 --tp 2
#
# Code:    /storage/noy/RandOpt/     (Lustre PVC)
# Results: /storage/noy/RandOpt/results/  (persists)
# Image:   noyhassid/randopt-vllm:v1
# Project: raja

set -e

DATASET="countdown"
MODEL="Qwen/Qwen2.5-3B-Instruct"
GPUS=4
TP=1
POPULATION_SIZE=5000
TRAIN_SAMPLES=200
TOP_K_RATIOS="0.04,0.01,0.05,0.1"
SIGMA_VALUES="0.0005,0.001,0.002"
MAX_TOKENS=1024
SEED=42
WANDB_PROJECT="randopt"
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)     DATASET="$2";         shift 2;;
        --model)       MODEL="$2";            shift 2;;
        --gpus)        GPUS="$2";             shift 2;;
        --tp)          TP="$2";               shift 2;;
        --population)  POPULATION_SIZE="$2";  shift 2;;
        --sigma)       SIGMA_VALUES="$2";     shift 2;;
        --no_wandb)    WANDB_PROJECT="";      shift;;
        *)             EXTRA_ARGS="$EXTRA_ARGS $1"; shift;;
    esac
done

NUM_ENGINES=$((GPUS / TP))
MODEL_SLUG=$(echo "$MODEL" | tr '/' '-' | tr '.' '_')
JOB_NAME="randopt-${DATASET}-${MODEL_SLUG}"
WORKDIR="/storage/noy/RandOpt"

echo "Submitting: $JOB_NAME"
echo "  Dataset : $DATASET | Model: $MODEL"
echo "  GPUs    : $GPUS (TP=$TP, engines=$NUM_ENGINES)"
echo "  N       : $POPULATION_SIZE | Sigmas: $SIGMA_VALUES"
echo "  WandB   : ${WANDB_PROJECT:-disabled}"

runai training submit "$JOB_NAME" \
    --project raja \
    --image noyhassid/randopt-vllm:latest \
    -g "$GPUS" \
    --node-type NVIDIA-H100-80GB-HBM3 \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir "$WORKDIR" \
    --command -- bash -c "
        pip install wandb --quiet
        export HF_HOME=/storage/noy/.cache/huggingface
        export HF_HUB_OFFLINE=1
        export TRANSFORMERS_OFFLINE=1
        [ -f /storage/noy/.wandb_api_key ] && export WANDB_API_KEY=\$(cat /storage/noy/.wandb_api_key)
        export VLLM_NO_USAGE_STATS=1
        export VLLM_DISABLE_COMPILE_SAMPLER=1
        export RAY_DEDUP_LOGS=1
        export VLLM_LOGGING_LEVEL=WARNING

        python3 randopt.py \
            --dataset $DATASET \
            --model_name '$MODEL' \
            --num_engines $NUM_ENGINES \
            --tp $TP \
            --population_size $POPULATION_SIZE \
            --train_samples $TRAIN_SAMPLES \
            --sigma_values '$SIGMA_VALUES' \
            --top_k_ratios '$TOP_K_RATIOS' \
            --max_tokens $MAX_TOKENS \
            --global_seed $SEED \
            --experiment_dir results/${DATASET}_${MODEL_SLUG} \
            --wandb_project '$WANDB_PROJECT' \
            --wandb_run_name '$JOB_NAME' \
            $EXTRA_ARGS
    "

echo ""
echo "Submitted: $JOB_NAME"
echo "Monitor:   runai training logs $JOB_NAME -f"
echo "Status:    runai training standard describe $JOB_NAME"
