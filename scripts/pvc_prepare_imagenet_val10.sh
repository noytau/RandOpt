#!/bin/bash
# PVC-side prep of the clean-VAL scoring set (0-GPU RunAI job): fetch the
# W&B artifact (10/class clean val images, IC-train counterparts, uploaded
# from Geoffry), extract, and build the manifest with the leakage-checking
# generator. Asserts throughout: job Completed == dataset verified.
set -euo pipefail

REPO=/storage/noy/RandOpt
VAL_ROOT=/storage/noy/datasets/imagenet/val
TMP=/storage/noy/tmp_val10

pip install wandb --quiet
export WANDB_API_KEY="$(cat /storage/noy/.wandb_api_key)"

mkdir -p "$TMP" "$VAL_ROOT"
python3 - <<PY
import wandb
api = wandb.Api()
art = api.artifact("randopt/imagenet-val10:latest")
art.download(root="$TMP")
print("downloaded", art.name)
PY

tar -C "$VAL_ROOT" -xzf "$TMP"/imagenet_val10.tar.gz
rm -rf "$TMP"

cd "$REPO"
# generator asserts: 10/class found, none in IC test split (no leakage)
python3 scripts/make_val_scoring_manifest.py \
    --ic_manifest data/imagenet_c/data.json \
    --val_root "$VAL_ROOT" \
    --per_class 10 \
    --out data/imagenet_val10/data.json

python3 - <<PY
import json, os
m = json.load(open("data/imagenet_val10/data.json"))
assert len(m) == 10000, len(m)
missing = [e["image"] for e in m if not os.path.exists(e["image"])]
assert not missing, missing[:3]
print("VERIFIED: 10000 clean-val scoring images on PVC")
PY

echo "PVC imagenet val10 scoring set ready"
