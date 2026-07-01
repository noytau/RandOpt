#!/bin/bash
# Run RandOpt vision smoke test on cluster
set -eo pipefail
pip install wandb transformers torchvision --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)

cd /storage/noy/RandOpt
mkdir -p results

python3 randopt_vision.py \
    --dataset cifar10 \
    --model_name facebook/dinov2-small \
    --num_engines 1 \
    --cuda_devices 0 \
    --population_size 20 \
    --train_samples 100 \
    --test_samples 200 \
    --sigma_values 0.001,0.01,0.1 \
    --top_k_ratios 0.2,0.5 \
    --wandb_project randopt \
    --wandb_run_name vision-smoke-test \
    2>&1 | tee /storage/noy/RandOpt/results/vision_smoke.log
