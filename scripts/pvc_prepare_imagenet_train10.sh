#!/bin/bash
# PVC-side dataset prep (run as a 0-GPU RunAI job from /storage/noy/RandOpt):
# fetch the clean-ImageNet 10/class subset (W&B artifact uploaded from
# Geoffry — the EXACT files of its data/imagenet/data.json) into
# /storage/noy/datasets/imagenet/train and regenerate the manifest with PVC
# paths. Exits nonzero if any verification fails, so job status == outcome
# (needed while the cluster-api cert outage blocks `runai logs`).
set -euo pipefail

REPO=/storage/noy/RandOpt
DATA_ROOT=/storage/noy/datasets/imagenet/train
TMP=/storage/noy/tmp_train10
ARTIFACT="imagenet-train10:latest"

pip install wandb --quiet
export WANDB_API_KEY="$(cat /storage/noy/.wandb_api_key)"

mkdir -p "$TMP" "$DATA_ROOT"
python3 - <<PY
import wandb
api = wandb.Api()
art = api.artifact(f"randopt/$ARTIFACT")
art.download(root="$TMP")
print("downloaded", art.name)
PY

tar -C "$DATA_ROOT" -xzf "$TMP"/imagenet_train10.tar.gz
rm -rf "$TMP"

cd "$REPO"
python3 scripts/make_imagenet_c_manifest.py \
    --data_dir "$DATA_ROOT" \
    --out data/imagenet/data.json \
    --train_per_class 10 --val_per_class 0 --test_per_class 0

python3 - <<PY
import json, os, sys
clean = json.load(open("data/imagenet/data.json"))
assert len(clean) == 10000, f"expected 10000 entries, got {len(clean)}"
wnids = {e["wnid"] for e in clean}
assert len(wnids) == 1000, f"expected 1000 classes, got {len(wnids)}"
assert all(e["split"] == "train" for e in clean)
missing = [e["image"] for e in clean if not os.path.exists(e["image"])]
assert not missing, f"{len(missing)} missing files, e.g. {missing[:3]}"
ic = json.load(open("data/imagenet_c/data.json"))
m1 = {e["wnid"]: e["label"] for e in clean}
m2 = {e["wnid"]: e["label"] for e in ic}
assert m1 == m2, "label<->wnid map differs from the ImageNet-C manifest"
print("VERIFIED: 10000 entries / 1000 classes / files exist / label map matches IC")
PY

echo "PVC imagenet train10 dataset ready"
