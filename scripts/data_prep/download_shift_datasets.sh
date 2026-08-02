#!/usr/bin/env bash
# Download + checksum-verify + extract the 4 distribution-shift datasets into
# $DATASETS_ROOT/raw/. Idempotent: `wget -c` resumes, and each dataset is
# skipped if its raw/ subdirectory already looks populated (override with
# --force). Intended to run backgrounded (`nohup ... &`) — ES alone is ~26 GB.
#
# Usage:
#   bash scripts/data_prep/download_shift_datasets.sh [--dataset imagenet_a|imagenet_r|imagenet_sketch|imagenet_es|all] [--force]
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-/mnt5/noy/datasets}"
RAW="$DATASETS_ROOT/raw"
LOGS="$DATASETS_ROOT/logs"
mkdir -p "$RAW" "$LOGS"

DATASET="all"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) DATASET="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

TS="$(date +%Y%m%d_%H%M%S)"
LOG="$LOGS/download_shift_datasets_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== download_shift_datasets.sh started $(date -Iseconds) -> $RAW ==="

_want() { [[ "$DATASET" == "all" || "$DATASET" == "$1" ]]; }
_populated() { [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -print -quit 2>/dev/null)" ]]; }

# ---------------------------------------------------------------------------
# ImageNet-R: Berkeley tar, md5 verified against hendrycks/imagenet-r
# ---------------------------------------------------------------------------
if _want imagenet_r; then
  dst="$RAW/imagenet_r"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar
    echo "a61312130a589d0ca1a8fca1f2bd3337  imagenet-r.tar" | md5sum -c -
    tar -xf imagenet-r.tar -C "$dst" --strip-components=1
    rm -f imagenet-r.tar
    echo "imagenet_r: extracted $(find "$dst" -maxdepth 1 -mindepth 1 -type d | wc -l) class dirs"
  else
    echo "imagenet_r: $dst already populated, skipping (use --force to redo)"
  fi
fi

# ---------------------------------------------------------------------------
# ImageNet-A: Berkeley tar, md5 verified against hendrycks/natural-adv-examples
# ---------------------------------------------------------------------------
if _want imagenet_a; then
  dst="$RAW/imagenet_a"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar
    echo "c3e55429088dc681f30d81f4726b6595  imagenet-a.tar" | md5sum -c -
    tar -xf imagenet-a.tar -C "$dst" --strip-components=1
    rm -f imagenet-a.tar
    echo "imagenet_a: extracted $(find "$dst" -maxdepth 1 -mindepth 1 -type d | wc -l) class dirs"
  else
    echo "imagenet_a: $dst already populated, skipping (use --force to redo)"
  fi
fi

# ---------------------------------------------------------------------------
# ImageNet-Sketch: hosted directly on HF (songweig/imagenet_sketch) — direct
# wget, no gdown/Google Drive (verified reachable from Geoffry 2026-08-01).
# ---------------------------------------------------------------------------
if _want imagenet_sketch; then
  dst="$RAW/imagenet_sketch"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c -O ImageNet-Sketch.zip \
      https://huggingface.co/datasets/songweig/imagenet_sketch/resolve/main/data/ImageNet-Sketch.zip
    unzip -q ImageNet-Sketch.zip -d "$dst.tmp"
    # zip extracts to a nested "sketch/" dir per upstream layout; flatten it.
    inner="$(find "$dst.tmp" -mindepth 1 -maxdepth 1 -type d | head -1)"
    mv "$inner" "$dst"
    rmdir "$dst.tmp" 2>/dev/null || true
    rm -f ImageNet-Sketch.zip
    echo "imagenet_sketch: extracted $(find "$dst" -maxdepth 1 -mindepth 1 -type d | wc -l) class dirs"
  else
    echo "imagenet_sketch: $dst already populated, skipping (use --force to redo)"
  fi
fi

# ---------------------------------------------------------------------------
# ImageNet-ES: hosted directly on HF (Edw2n/ImageNet-ES) — direct wget, no
# ES-Studio setup (verified reachable from Geoffry 2026-08-01, 26.27 GB).
# Standard ES (not ES-Diverse) per 2026-08-01 decision — see plan/TASKS.md.
# ---------------------------------------------------------------------------
if _want imagenet_es; then
  dst="$RAW/imagenet_es"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c -O ImageNet-ES.zip \
      https://huggingface.co/datasets/Edw2n/ImageNet-ES/resolve/main/ImageNet-ES.zip
    unzip -q ImageNet-ES.zip -d "$dst.tmp"
    inner="$(find "$dst.tmp" -mindepth 1 -maxdepth 1 -type d | head -1)"
    mv "$inner"/* "$dst"/ 2>/dev/null || mv "$inner" "$dst"
    rm -rf "$dst.tmp"
    rm -f ImageNet-ES.zip
    echo "imagenet_es: top-level tree after extraction:"
    find "$dst" -maxdepth 4 | sort
  else
    echo "imagenet_es: $dst already populated, skipping (use --force to redo)"
  fi
fi

echo "=== download_shift_datasets.sh finished $(date -Iseconds) ==="
