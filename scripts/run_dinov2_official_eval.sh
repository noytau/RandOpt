#!/bin/bash
# Official DINOv2 evaluation (facebookresearch/dinov2) of ViT-g/14+reg4 on
# ImageNet-1k, using Meta's own eval code — dinov2/eval/knn.py and
# dinov2/eval/linear.py. These are the exact modules the README's
# dinov2/run/eval/* commands execute; run/eval/* is only a SLURM submitit
# launcher, and our cluster is RunAI, so we invoke the eval modules directly.
#
# Env vars:
#   EVAL           knn | linear                      (default knn)
#   IMAGENET_ROOT  dir with train/ val/ labels.txt   (required)
#   IMAGENET_EXTRA metadata dir (entries-*.npy)      (default $IMAGENET_ROOT/extra)
#   VAL_ROOT       override val root — point at an ImageNet-C "view" dir to
#                  evaluate corrupted val against the clean-train gallery/head
#   VAL_EXTRA      metadata dir for VAL_ROOT         (default $IMAGENET_EXTRA)
#   WEIGHTS        backbone .pth                     (auto-downloaded if absent)
#   OUTPUT_DIR     results/logs dir
#   BATCH          eval batch size                   (default 256)
#   NPROC          GPUs; >1 runs via torchrun        (default 1; linear wants 4-8)
set -euo pipefail

EVAL=${EVAL:-knn}
DINOV2_DIR=${DINOV2_DIR:-/storage/noy/dinov2}
DATASETS_ROOT=${DATASETS_ROOT:-/storage/noy/datasets}
IMAGENET_ROOT=${IMAGENET_ROOT:-$DATASETS_ROOT/imagenet}
[ -d "$IMAGENET_ROOT/train" ] || { echo "ImageNet not found at $IMAGENET_ROOT (need train/ val/ labels.txt) — no shared storage between servers, check THIS server's datasets root"; exit 1; }
IMAGENET_EXTRA=${IMAGENET_EXTRA:-$IMAGENET_ROOT/extra}
VAL_ROOT=${VAL_ROOT:-$IMAGENET_ROOT}
VAL_EXTRA=${VAL_EXTRA:-$IMAGENET_EXTRA}
WEIGHTS=${WEIGHTS:-/storage/noy/checkpoints/dinov2_vitg14_reg4_pretrain.pth}
OUTPUT_DIR=${OUTPUT_DIR:-/storage/noy/RandOpt/results/dinov2-official-$EVAL}
BATCH=${BATCH:-256}
NPROC=${NPROC:-1}

# one-time pieces: official repo (pinned commit), python deps, backbone weights
DINOV2_COMMIT=${DINOV2_COMMIT:-7764ea0f912e53c92e82eb78a2a1631e92725fc8}
[ -d "$DINOV2_DIR" ] || git clone -q https://github.com/facebookresearch/dinov2 "$DINOV2_DIR"
git -C "$DINOV2_DIR" checkout -q "$DINOV2_COMMIT" \
  || { git -C "$DINOV2_DIR" fetch -q origin; git -C "$DINOV2_DIR" checkout -q "$DINOV2_COMMIT"; }
python -c "import torchmetrics, omegaconf" 2>/dev/null || pip install -q torchmetrics omegaconf
mkdir -p "$(dirname "$WEIGHTS")" "$OUTPUT_DIR"
[ -f "$WEIGHTS" ] || wget -q -c -O "$WEIGHTS" \
  https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth

cd "$DINOV2_DIR"
export PYTHONPATH="$DINOV2_DIR"

# one-time split metadata (entries-*.npy) the official ImageNet class reads
python -u - <<EOF
import os
from dinov2.data.datasets import ImageNet
for split, root, extra in [(ImageNet.Split.TRAIN, "$IMAGENET_ROOT", "$IMAGENET_EXTRA"),
                           (ImageNet.Split.VAL, "$VAL_ROOT", "$VAL_EXTRA")]:
    if not os.path.exists(os.path.join(extra, f"entries-{split.value.upper()}.npy")):
        print(f"dumping {split.value} metadata -> {extra}")
        ImageNet(split=split, root=root, extra=extra).dump_extra()
EOF

TRAIN_DS="ImageNet:split=TRAIN:root=$IMAGENET_ROOT:extra=$IMAGENET_EXTRA"
VAL_DS="ImageNet:split=VAL:root=$VAL_ROOT:extra=$VAL_EXTRA"
if [ "$NPROC" -gt 1 ]; then
  RUN="python -u -m torch.distributed.run --nproc_per_node=$NPROC"
else
  RUN="python -u"
fi

if [ "$EVAL" = knn ]; then
  $RUN dinov2/eval/knn.py \
    --config-file dinov2/configs/eval/vitg14_reg4_pretrain.yaml \
    --pretrained-weights "$WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --train-dataset "$TRAIN_DS" \
    --val-dataset "$VAL_DS" \
    --batch-size "$BATCH" \
    --gather-on-cpu
else
  $RUN dinov2/eval/linear.py \
    --config-file dinov2/configs/eval/vitg14_reg4_pretrain.yaml \
    --pretrained-weights "$WEIGHTS" \
    --output-dir "$OUTPUT_DIR" \
    --train-dataset "$TRAIN_DS" \
    --val-dataset "$VAL_DS" \
    --batch-size "$BATCH"
fi
