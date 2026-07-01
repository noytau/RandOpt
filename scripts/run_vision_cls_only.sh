#!/bin/bash
# RandOpt vision — classifier-only perturbation experiment.
# Backbone (DINOv2-base, 86M params) is FROZEN at pretrained weights.
# Only the linear head (768→10, ~7.7K params) is perturbed.
# Linear probe warm-init: data/cifar10/linear_probe_dinov2base.pt (98.22% acc)
# Sigma sweep: 0.01, 0.1, 0.5, 1.0, 5.0  (much larger than backbone run)
set -eo pipefail
pip install wandb transformers torchvision datasets --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)

cd /storage/noy/RandOpt
mkdir -p results

python3 randopt_vision.py \
    --dataset cifar10 \
    --model_name facebook/dinov2-base \
    --num_engines 1 \
    --cuda_devices 0 \
    --population_size 500 \
    --train_samples 1000 \
    --test_samples 2000 \
    --sigma_values 0.01,0.1,0.5,1.0,5.0 \
    --top_k_ratios 0.01,0.05,0.1 \
    --linear_init_path /storage/noy/RandOpt/data/cifar10/linear_probe_dinov2base.pt \
    --perturb_target classifier \
    --wandb_project randopt \
    --wandb_run_name vision-cls-only-n500 \
    2>&1 | tee results/vision_cls_only.log
