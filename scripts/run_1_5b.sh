#!/bin/bash
# Run RandOpt 1.5B countdown experiment on cluster
set -e
pip install wandb --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
export VLLM_NO_USAGE_STATS=1
export RAY_DEDUP_LOGS=1
export VLLM_LOGGING_LEVEL=WARNING

FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
echo "Free GPU memory: ${FREE_MEM} MiB"
if [ "${FREE_MEM}" -lt 15000 ]; then
    echo "ERROR: Not enough free GPU memory (zombie memory node). Exiting."
    exit 1
fi

mkdir -p /storage/noy/RandOpt/results/countdown_1-5b
cd /storage/noy/RandOpt
python3 randopt.py \
    --dataset countdown \
    --model_name Qwen/Qwen2.5-1.5B-Instruct \
    --num_engines 1 \
    --tp 1 \
    --population_size 500 \
    --train_samples 200 \
    --sigma_values 0.0001,0.0005,0.001,0.002 \
    --top_k_ratios 0.01,0.05,0.1 \
    --max_tokens 1024 \
    --experiment_dir results/countdown_1-5b \
    --wandb_project randopt \
    --wandb_run_name randopt-1.5b-countdown-n500 \
    2>&1 | tee /storage/noy/RandOpt/results/countdown_1-5b/run.log
