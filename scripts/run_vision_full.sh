#!/bin/bash
# Full RandOpt vision experiment: DINOv2-base + CIFAR-10, N=500
# Optionally trains a linear probe first for warm init.
set -e
pip install wandb transformers torchvision datasets --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)

cd /storage/noy/RandOpt
mkdir -p results

PROBE_PATH=/storage/noy/RandOpt/data/cifar10/linear_probe_dinov2base.pt
NUM_ENGINES=${NUM_ENGINES:-1}
CUDA_DEVS=${CUDA_DEVS:-0}

# Train linear probe if not already done (~5 min on 1 GPU)
if [ ! -f "$PROBE_PATH" ]; then
    echo "=== Training linear probe (warm init) ==="
    python3 vision/train_linear_probe.py \
        --model_name facebook/dinov2-base \
        --output_path "$PROBE_PATH" \
        --epochs 10 \
        2>&1
fi

python3 randopt_vision.py \
    --dataset cifar10 \
    --model_name facebook/dinov2-base \
    --num_engines "$NUM_ENGINES" \
    --cuda_devices "$CUDA_DEVS" \
    --population_size 500 \
    --train_samples 1000 \
    --test_samples 2000 \
    --sigma_values 0.0001,0.001,0.01,0.1 \
    --top_k_ratios 0.01,0.05,0.1 \
    --linear_init_path "$PROBE_PATH" \
    --wandb_project randopt \
    --wandb_run_name vision-dinov2base-cifar10-n500 \
    2>&1 | tee results/vision_full.log
