#!/bin/bash
# PVC-side DINOv3 ingest (0-GPU RunAI job): clone the repo + download gated
# weights via Meta's signed CDN token (env DINOV3_TOKEN_QS = the query string
# of a valid signed URL; its Resource covers the whole host so one token
# authorizes every file). Asserts sizes so job status == outcome.
set -euo pipefail

: "${DINOV3_TOKEN_QS:?need DINOV3_TOKEN_QS env (Policy=...&Signature=...&Key-Pair-Id=...)}"

REPO=/storage/noy/dinov3
MODELS=/storage/noy/models/dinov3
BASE=https://dinov3.llamameta.net

free_gb=$(df -BG /storage | awk 'NR==2 {gsub("G","",$4); print $4}')
[ "$free_gb" -ge 40 ] || { echo "only ${free_gb}GB free on /storage, need 40"; exit 1; }

if [ ! -d "$REPO/.git" ]; then
    git clone https://github.com/facebookresearch/dinov3.git "$REPO"
fi
echo "dinov3 repo @ $(git -C "$REPO" rev-parse --short HEAD)"

mkdir -p "$MODELS"
declare -A want=(
    ["dinov3_vit7b16/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"]=26864407683
    ["dinov3_vit7b16/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth"]=32774141
    ["dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"]=86531063
)
for path in "${!want[@]}"; do
    out="$MODELS/$(basename "$path")"
    expected="${want[$path]}"
    if [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$expected" ]; then
        echo "already present: $(basename "$out")"; continue
    fi
    echo "downloading $(basename "$out") ($expected bytes)..."
    wget -q -c -O "$out" "$BASE/$path?$DINOV3_TOKEN_QS"
    actual=$(stat -c%s "$out")
    [ "$actual" = "$expected" ] || { echo "SIZE MISMATCH $out: $actual != $expected"; exit 1; }
done

echo "DINOV3 INGEST OK: repo + $(ls "$MODELS" | wc -l) weight files"
ls -l "$MODELS"
