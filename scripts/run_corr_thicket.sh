#!/bin/bash
# E1: individual-gain / thicket-existence study on SPair-71k (DINOv2, no head).
# Scans scope x sigma, reports best held-out individual gain over base.
set -eo pipefail
cd /storage/noy/RandOpt
git pull origin feature/vision-randopt -q
pip install transformers torchvision Pillow --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)
mkdir -p results

python3 scripts/randopt_corr_thicket.py \
    --dataset spair71k \
    --model_name facebook/dinov2-base \
    --nA 200 --nB 200 \
    --population_size 60 \
    --sigma_values 0.00003,0.0001,0.0003,0.001,0.003 \
    --scopes all,last2,last1 \
    --wandb_project randopt \
    --wandb_run_name thicket-spair71k-v1 \
    2>&1 | tee results/corr_thicket.log
