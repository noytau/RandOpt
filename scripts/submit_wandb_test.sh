#!/bin/bash
# Minimal smoke test to verify W&B logging end-to-end.
# Runtime: ~10-15 min on 1x H100.
# After completion, check: https://wandb.ai → project "randopt" → run "randopt-wandb-test"
#
# Usage:
#   bash scripts/submit_wandb_test.sh

set -e

JOB_NAME="randopt-wandb-test"
WORKDIR="/storage/noy/RandOpt"

runai training delete "$JOB_NAME" -p raja 2>/dev/null && sleep 2 || true

echo "Submitting W&B smoke test: $JOB_NAME"

runai training submit "$JOB_NAME" \
    --project raja \
    --image noyhassid/randopt-vllm:latest \
    -g 1 \
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
            --dataset countdown \
            --train_data_path data/countdown/countdown.json \
            --test_data_path  data/countdown/countdown.json \
            --model_name Qwen/Qwen2.5-0.5B-Instruct \
            --num_engines 1 \
            --tp 1 \
            --cuda_devices 0 \
            --population_size 20 \
            --train_samples 30 \
            --test_samples 50 \
            --sigma_values '0.0005,0.001,0.002' \
            --top_k_ratios '0.1,0.2,0.5' \
            --max_tokens 512 \
            --global_seed 42 \
            --experiment_dir results/wandb_test \
            --wandb_project randopt \
            --wandb_run_name $JOB_NAME
    "

echo ""
echo "Submitted: $JOB_NAME"
echo "Monitor:   runai training logs $JOB_NAME -f"
echo "W&B:       https://wandb.ai → project 'randopt' → run '$JOB_NAME'"
