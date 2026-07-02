#!/bin/bash
# RandOpt semantic correspondence — DINOv2-base on SPair-71k.
# No linear head. Reward = PCK@0.1 from raw patch embedding cosine similarity.
# DINOv2-base baseline ~64% PCK@0.1 on SPair-71k test set.
set -eo pipefail
cd /storage/noy/RandOpt
git pull origin feature/vision-randopt -q
pip install wandb transformers torchvision datasets --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)

cd /storage/noy/RandOpt
mkdir -p results

python3 randopt_correspondence.py \
    --dataset spair71k \
    --model_name facebook/dinov2-base \
    --num_engines 1 \
    --cuda_devices 0 \
    --population_size 500 \
    --train_samples 500 \
    --test_samples 1000 \
    --sigma_values 0.00001,0.0001,0.001,0.01 \
    --top_k_ratios 0.01,0.05,0.1 \
    --wandb_project randopt \
    --wandb_run_name vision-corr-spair71k-n500 \
    2>&1 | tee results/vision_correspondence.log
