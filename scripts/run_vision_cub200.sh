#!/bin/bash
# RandOpt vision — CUB-200-2011 (200 bird species), DINOv2-base full perturbation.
# DINOv2-base linear probe baseline ~82% — good gap; DINOv2 has known bird part features.
set -e
pip install wandb transformers torchvision datasets --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)

cd /storage/noy/RandOpt
mkdir -p results

# Train linear probe for CUB-200 if not cached
PROBE_PATH=/storage/noy/RandOpt/data/cub200/linear_probe_dinov2base.pt
if [ ! -f "$PROBE_PATH" ]; then
    echo "=== Training CUB-200 linear probe ==="
    python3 vision/train_linear_probe.py \
        --model_name facebook/dinov2-base \
        --dataset cub200 \
        --num_classes 200 \
        --output_path "$PROBE_PATH" \
        --epochs 20 \
        2>&1
fi

python3 randopt_vision.py \
    --dataset cub200 \
    --model_name facebook/dinov2-base \
    --num_engines 1 \
    --cuda_devices 0 \
    --num_classes 200 \
    --population_size 500 \
    --train_samples 1000 \
    --test_samples 2000 \
    --sigma_values 0.00001,0.0001,0.001,0.01 \
    --top_k_ratios 0.01,0.05,0.1 \
    --linear_init_path "$PROBE_PATH" \
    --perturb_target all \
    --wandb_project randopt \
    --wandb_run_name vision-cub200-n500 \
    2>&1 | tee results/vision_cub200.log
