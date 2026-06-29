#!/bin/bash
# Pull latest RandOpt code on Lustre PVC via a CPU RunAI job.
#
# Usage (run from Mac):
#   bash scripts/update_cluster_code.sh
#
# One-time cluster setup (run once from inside any container with /storage mounted):
#   git -C /storage/noy/RandOpt remote set-url origin \
#     https://$(cat /storage/noy/.github_token)@github.com/noytau/RandOpt.git

set -e

JOB_NAME="randopt-update"

runai training delete "$JOB_NAME" -p raja 2>/dev/null && sleep 2 || true

echo "Submitting git pull job: $JOB_NAME"

runai training submit "$JOB_NAME" \
    --project raja \
    --image noyhassid/randopt-vllm:latest \
    -g 0 \
    --existing-pvc claimname=storage,path=/storage \
    --working-dir /storage/noy/RandOpt \
    --command -- bash -c "
        echo 'Pulling latest code into /storage/noy/RandOpt ...'
        git -C /storage/noy/RandOpt pull --ff-only
        echo 'Done. Current HEAD:'
        git -C /storage/noy/RandOpt log --oneline -3
    "

echo ""
echo "Watch: runai training logs $JOB_NAME -f"
