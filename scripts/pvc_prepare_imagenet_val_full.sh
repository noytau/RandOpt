#!/bin/bash
# PVC prep (0-GPU job): full ImageNet val split (50/class = 50k) from the
# W&B artifact + the two manifests the balanced-sampling experiments use:
#   data/imagenet_val/data.json    - all 50k val images, split=train
#   data/imagenet_c/data_full.json - all 50k IC images,  split=test
# Asserts everywhere: job Completed == verified.
set -euo pipefail

REPO=/storage/noy/RandOpt
IMAGENET_ROOT=/storage/noy/datasets/imagenet
IC_ROOT=/storage/noy/RandOpt/data/imagenet_c/gaussian_noise/3
TMP=/storage/noy/tmp_val_full

pip install wandb --quiet
export WANDB_API_KEY="$(cat /storage/noy/.wandb_api_key)"

free_gb=$(df -BG /storage | awk 'NR==2 {gsub("G","",$4); print $4}')
[ "$free_gb" -ge 20 ] || { echo "only ${free_gb}GB free"; exit 1; }

mkdir -p "$TMP" "$IMAGENET_ROOT"
python3 - <<PY
import wandb
api = wandb.Api()
art = api.artifact("randopt/imagenet-val-full:latest")
art.download(root="$TMP")
print("downloaded", art.name)
PY

tar -C "$IMAGENET_ROOT" -xzf "$TMP"/imagenet_val_full.tar.gz   # -> val/
rm -rf "$TMP"

cd "$REPO"
python3 scripts/make_imagenet_c_manifest.py \
    --data_dir "$IMAGENET_ROOT/val" \
    --out data/imagenet_val/data.json \
    --train_per_class 50 --val_per_class 0 --test_per_class 0
python3 scripts/make_imagenet_c_manifest.py \
    --data_dir "$IC_ROOT" \
    --out data/imagenet_c/data_full.json \
    --train_per_class 0 --val_per_class 0 --test_per_class 50

python3 - <<PY
import json, os, random
for path, split, n in [("data/imagenet_val/data.json", "train", 50000),
                       ("data/imagenet_c/data_full.json", "test", 50000)]:
    m = json.load(open(path))
    assert len(m) == n, (path, len(m))
    assert all(e["split"] == split for e in m)
    assert len({e["wnid"] for e in m}) == 1000
    for e in random.Random(0).sample(m, 200):
        assert os.path.exists(e["image"]), e["image"]
    print(f"VERIFIED {path}: {n} entries, 1000 classes, files exist")
PY
echo "PVC full-val + full-IC manifests ready"
