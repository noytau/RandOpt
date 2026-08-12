#!/usr/bin/env bash
# Download + extract 5 new (non-ImageNet) classification datasets into
# $DATASETS_ROOT/raw/, each chosen because it ships an OFFICIAL train/val/test
# (or train/test) split of its own — unlike imagenet_a/r/sketch/es, which are
# test-only draws scored against clean ImageNet-train. Idempotent: skipped if
# already populated (override with --force). Intended to run backgrounded
# (nohup ... &) — DomainNet-sketch alone is ~2.5 GB, aircraft ~2.75 GB.
#
# Datasets:
#   exdark            - low-light/dark images, 12 classes, official train(3000)
#                        /val(1800)/test(2563) split via imageclasslist.txt.
#                        Source image archive is Google Drive -> needs gdown
#                        (pip install gdown into the active python env first).
#   fgvc_aircraft      - 100 aircraft variants, official train(3334)/val(3333)
#                        /test(3333) split, direct download.
#   stanford_dogs      - 120 dog breeds, official train(12000)/test(8580)
#                        split (no official val — carved later by the manifest
#                        builder), direct download.
#   domainnet_clipart  - DomainNet "clipart" domain (cartoon/graphic style),
#                        345 classes, official train/test split, direct dl.
#   domainnet_sketch   - DomainNet "sketch" domain (hand-drawn), same 345
#                        classes, official train/test split, direct dl.
#
# Usage:
#   bash scripts/data_prep/download_new_datasets.sh [--dataset exdark|fgvc_aircraft|stanford_dogs|domainnet_clipart|domainnet_sketch|all] [--force]
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
LOG="$LOGS/download_new_datasets_${TS}.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== download_new_datasets.sh started $(date -Iseconds) -> $RAW ==="

_want() { [[ "$DATASET" == "all" || "$DATASET" == "$1" ]]; }
_populated() { [[ -d "$1" ]] && [[ -n "$(find "$1" -mindepth 1 -print -quit 2>/dev/null)" ]]; }

# ---------------------------------------------------------------------------
# ExDark: low-light/dark images. Image archive lives on Google Drive (no
# direct wget host) -> use gdown. Groundtruth split file (imageclasslist.txt)
# is a direct GitHub raw file.
# ---------------------------------------------------------------------------
if _want exdark; then
  dst="$RAW/exdark"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    command -v gdown >/dev/null || { echo "gdown not found — run: pip install gdown" >&2; exit 1; }
    gdown "https://drive.google.com/uc?id=1BHmPgu8EsHoFDDkMGLVoXIlCth2dW6Yx" -O ExDark.zip
    unzip -q ExDark.zip -d "$dst.tmp"
    # zip extracts to a nested "ExDark/<Class>/..." dir; flatten it.
    inner="$(find "$dst.tmp" -mindepth 1 -maxdepth 1 -type d | head -1)"
    mv "$inner"/* "$dst"/ 2>/dev/null || mv "$inner" "$dst"
    rm -rf "$dst.tmp" ExDark.zip
    wget -c -O "$dst/imageclasslist.txt" \
      https://raw.githubusercontent.com/cs-chan/Exclusively-Dark-Image-Dataset/master/Groundtruth/imageclasslist.txt
    echo "exdark: extracted $(find "$dst" -maxdepth 1 -mindepth 1 -type d | wc -l) class dirs"
  else
    echo "exdark: $dst already populated, skipping (use --force to redo)"
  fi
fi

# ---------------------------------------------------------------------------
# FGVC-Aircraft: official Oxford VGG tar.gz, official train/val/test lists
# included inside data/images_variant_{train,val,test}.txt.
# ---------------------------------------------------------------------------
if _want fgvc_aircraft; then
  dst="$RAW/fgvc_aircraft"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz
    tar -xzf fgvc-aircraft-2013b.tar.gz -C "$dst" --strip-components=1
    rm -f fgvc-aircraft-2013b.tar.gz
    echo "fgvc_aircraft: $(find "$dst/data/images" -maxdepth 1 -type f | wc -l) images"
  else
    echo "fgvc_aircraft: $dst already populated, skipping (use --force to redo)"
  fi
fi

# ---------------------------------------------------------------------------
# Stanford Dogs: official Stanford Vision tars. lists.tar has train_list.mat/
# test_list.mat (official split, 1-indexed labels); no official val (carved
# by the manifest builder from train).
# ---------------------------------------------------------------------------
if _want stanford_dogs; then
  dst="$RAW/stanford_dogs"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar
    wget -c http://vision.stanford.edu/aditya86/ImageNetDogs/lists.tar
    tar -xf images.tar -C "$dst"
    tar -xf lists.tar -C "$dst"
    rm -f images.tar lists.tar
    echo "stanford_dogs: $(find "$dst/Images" -maxdepth 1 -mindepth 1 -type d | wc -l) class dirs"
  else
    echo "stanford_dogs: $dst already populated, skipping (use --force to redo)"
  fi
fi

# ---------------------------------------------------------------------------
# DomainNet clipart / sketch: cleaned-version zips + official train/test txt
# lists, both hosted directly on csr.bu.edu (no Google Drive/gdown needed).
# Same 345-class label space in both domains.
# ---------------------------------------------------------------------------
if _want domainnet_clipart; then
  dst="$RAW/domainnet_clipart"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c -O clipart.zip https://csr.bu.edu/ftp/visda/2019/multi-source/groundtruth/clipart.zip
    unzip -q clipart.zip -d "$dst.tmp"
    inner="$(find "$dst.tmp" -mindepth 1 -maxdepth 1 -type d | head -1)"
    mv "$inner"/* "$dst"/ 2>/dev/null || mv "$inner" "$dst"
    rm -rf "$dst.tmp" clipart.zip
    wget -c -O "$dst/clipart_train.txt" https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_train.txt
    wget -c -O "$dst/clipart_test.txt" https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/clipart_test.txt
    echo "domainnet_clipart: $(find "$dst" -maxdepth 1 -mindepth 1 -type d | wc -l) class dirs"
  else
    echo "domainnet_clipart: $dst already populated, skipping (use --force to redo)"
  fi
fi

if _want domainnet_sketch; then
  dst="$RAW/domainnet_sketch"
  if [[ "$FORCE" == 1 ]] || ! _populated "$dst"; then
    mkdir -p "$dst"
    cd "$RAW"
    wget -c -O sketch.zip https://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip
    unzip -q sketch.zip -d "$dst.tmp"
    inner="$(find "$dst.tmp" -mindepth 1 -maxdepth 1 -type d | head -1)"
    mv "$inner"/* "$dst"/ 2>/dev/null || mv "$inner" "$dst"
    rm -rf "$dst.tmp" sketch.zip
    wget -c -O "$dst/sketch_train.txt" https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_train.txt
    wget -c -O "$dst/sketch_test.txt" https://csr.bu.edu/ftp/visda/2019/multi-source/domainnet/txt/sketch_test.txt
    echo "domainnet_sketch: $(find "$dst" -maxdepth 1 -mindepth 1 -type d | wc -l) class dirs"
  else
    echo "domainnet_sketch: $dst already populated, skipping (use --force to redo)"
  fi
fi

echo "=== download_new_datasets.sh finished $(date -Iseconds) ==="
