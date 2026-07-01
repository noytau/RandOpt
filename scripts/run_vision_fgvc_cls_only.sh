#!/bin/bash
# RandOpt vision — FGVC-Aircraft, classifier-only perturbation.
# Backbone frozen at pretrained DINOv2-base weights.
# Only linear head (768→100, ~76.8K params) is perturbed.
# Comparison: vs run_vision_fgvc.sh (full backbone) to isolate backbone contribution.
set -e
pip install wandb transformers torchvision datasets --quiet

export HF_HOME=/storage/noy/.cache/huggingface
export WANDB_API_KEY=$(cat /storage/noy/.wandb_api_key)

cd /storage/noy/RandOpt
mkdir -p results

PROBE_PATH=/storage/noy/RandOpt/data/fgvc_aircraft/linear_probe_dinov2base.pt
if [ ! -f "$PROBE_PATH" ]; then
    echo "=== Training FGVC-Aircraft linear probe ==="
    python3 vision/train_linear_probe.py \
        --model_name facebook/dinov2-base \
        --dataset fgvc_aircraft \
        --num_classes 100 \
        --output_path "$PROBE_PATH" \
        --epochs 20 \
        2>&1
fi

python3 randopt_vision.py \
    --dataset fgvc_aircraft \
    --model_name facebook/dinov2-base \
    --num_engines 1 \
    --cuda_devices 0 \
    --num_classes 100 \
    --population_size 500 \
    --train_samples 1000 \
    --test_samples 2000 \
    --sigma_values 0.1,0.5,1.0,5.0,10.0 \
    --top_k_ratios 0.01,0.05,0.1 \
    --linear_init_path "$PROBE_PATH" \
    --perturb_target classifier \
    --wandb_project randopt \
    --wandb_run_name vision-fgvc-cls-only-n500 \
    2>&1 | tee results/vision_fgvc_cls_only.log
