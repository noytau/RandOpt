#!/bin/bash
# RandOpt semantic correspondence — DINOv2-base on SPair-71k, N=5000.
set -eo pipefail
pip install transformers torchvision Pillow --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)

cd /storage/noy/RandOpt
mkdir -p results

python3 randopt_correspondence.py \
    --dataset spair71k \
    --model_name facebook/dinov2-base \
    --num_engines 1 \
    --cuda_devices 0 \
    --population_size 5000 \
    --train_samples 500 \
    --test_samples 1000 \
    --sigma_values 0.00001,0.0001,0.001,0.01 \
    --top_k_ratios 0.01,0.05,0.1 \
    --wandb_project randopt \
    --wandb_run_name vision-corr-spair71k-n5000 \
    2>&1 | tee results/vision_correspondence_n5000.log
