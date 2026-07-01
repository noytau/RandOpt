"""SPair-71k dataset handler for DINOv2 semantic correspondence.

SPair-71k: ~70k image pairs across 18 semantic categories with keypoint annotations.
Evaluation: PCK@0.1 — keypoint predicted within 10% of target bbox max(w,h).

No linear head needed. Reward = PCK@0.1 from cosine similarity of DINOv2 patch features.
DINOv2-base baseline PCK@0.1 ~64% on test, headroom to ~82% (SOTA with learned heads).

Data is downloaded from POSTECH's official server and cached on PVC.

Reference: Min et al., "SPair-71k: A Large-scale Benchmark for Semantic Correspondence", 2019
Official: http://cvlab.postech.ac.kr/research/SPair-71k/
"""
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision import transforms

from .base import DatasetHandler

_to_tensor = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])

SPAIR_URL = "http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz"

# DINOv2-base patch grid: 224/14 = 16 patches per side
PATCH_SIZE = 14
INPUT_SIZE = 224   # after engine _preprocess
LOAD_SIZE  = 448   # images are resized to this at load time, halved by engine
GRID = INPUT_SIZE // PATCH_SIZE  # 16


def _ensure_downloaded(data_dir: str) -> Path:
    """Download and extract SPair-71k to data_dir if not already present."""
    root = Path(data_dir) / "SPair-71k"
    if (root / "ImageData").exists():
        return root
    os.makedirs(data_dir, exist_ok=True)
    tgz = Path(data_dir) / "SPair-71k.tar.gz"
    print(f"Downloading SPair-71k from {SPAIR_URL} ...")
    subprocess.run(["wget", "-c", "-q", "--show-progress", "-O", str(tgz), SPAIR_URL], check=True)
    print("Extracting SPair-71k ...")
    subprocess.run(["tar", "-xzf", str(tgz), "-C", data_dir], check=True)
    return root


def _load_split(root: Path, split: str) -> List[Dict]:
    """Parse SPair-71k annotation JSONs into a list of pair dicts."""
    from PIL import Image

    split_name = {"train": "trn", "val": "val", "test": "test"}.get(split, split)
    pair_dir = root / "PairAnnotation" / split_name

    items = []
    for json_path in sorted(pair_dir.rglob("*.json")):
        with open(json_path) as f:
            ann = json.load(f)

        src_path = root / "ImageData" / ann["category"] / ann["src_imname"]
        tgt_path = root / "ImageData" / ann["category"] / ann["trg_imname"]

        src_img = Image.open(src_path).convert("RGB")
        tgt_img = Image.open(tgt_path).convert("RGB")
        src_w, src_h = src_img.size
        tgt_w, tgt_h = tgt_img.size

        # Bounding box threshold in patch units (standard SPair PCK@0.1)
        tgt_bbox = ann["trg_bndbox"]   # [x1, y1, x2, y2]
        bbox_w = (tgt_bbox[2] - tgt_bbox[0]) / tgt_w
        bbox_h = (tgt_bbox[3] - tgt_bbox[1]) / tgt_h
        # After resize to LOAD_SIZE → engine rescales to INPUT_SIZE (factor 2 smaller)
        bbox_thresh = 0.1 * max(bbox_w, bbox_h) * GRID  # in patch units

        # Keypoints: [[x,y], ...], only use visible ones
        kps_A = ann.get("kps_A", [])  # source keypoints
        kps_B = ann.get("kps_B", [])  # target keypoints
        valid = [i for i in range(min(len(kps_A), len(kps_B)))
                 if kps_A[i] is not None and kps_B[i] is not None]

        kpts_src = [_xy_to_patch(kps_A[i][0], kps_A[i][1], src_w, src_h) for i in valid]
        kpts_tgt = [_xy_to_patch(kps_B[i][0], kps_B[i][1], tgt_w, tgt_h) for i in valid]

        if not kpts_src:
            continue

        items.append({
            "image_tensor":     _to_tensor(src_img),
            "image_tensor_tgt": _to_tensor(tgt_img),
            "kpts_src":   kpts_src,
            "kpts_tgt":   kpts_tgt,
            "bbox_thresh": bbox_thresh,
            "category":   ann["category"],
            "ground_truth": "pck",
            "messages": [],
        })
    return items


def _xy_to_patch(x: float, y: float, img_w: int, img_h: int) -> Tuple[int, int]:
    """Map keypoint (x, y) in original image to (row, col) in 16×16 patch grid.

    Images are loaded at LOAD_SIZE (448) → engine resizes to INPUT_SIZE (224, factor 2).
    """
    scale = (INPUT_SIZE / 2) / max(img_w, 1)  # 224 / 448 * (448 / img_w) = 224 / img_w... wait
    # Actually: image loaded at LOAD_SIZE=448 then engine resizes to 224 (factor 0.5)
    # So effective scale from original image to 224: 224 / img_w (or img_h)
    px = min(int(x * (INPUT_SIZE / img_w) / PATCH_SIZE), GRID - 1)
    py = min(int(y * (INPUT_SIZE / img_h) / PATCH_SIZE), GRID - 1)
    return py, px  # (row, col)


def compute_pck(
    feats_src: torch.Tensor,
    feats_tgt: Optional[torch.Tensor],
    kpts_src: List[Tuple[int, int]],
    kpts_tgt: List[Tuple[int, int]],
    bbox_thresh: float = 1.6,
    precomputed_sim: bool = False,
) -> float:
    """Compute PCK@0.1 for one image pair.

    Args:
        feats_src: (P, D) patch features OR (P, P) similarity matrix
        feats_tgt: (P, D) patch features; None if precomputed_sim=True
        kpts_src/tgt: (row, col) in GRID x GRID patch space
        bbox_thresh: threshold in patch units = 0.1 * max(bbox_w, bbox_h) * GRID
        precomputed_sim: if True, feats_src is already a (P, P) sim matrix
    """
    if not kpts_src:
        return 0.0

    if precomputed_sim:
        src_idx = [r * GRID + c for r, c in kpts_src]
        sim = feats_src[src_idx]          # (K, P)
    else:
        src_vecs = torch.stack([feats_src[r * GRID + c] for r, c in kpts_src])  # (K, D)
        sim = src_vecs @ feats_tgt.T      # (K, P)

    pred_flat = sim.argmax(dim=-1)        # (K,)

    correct = 0
    for i, (tgt_r, tgt_c) in enumerate(kpts_tgt):
        pred_r = int(pred_flat[i]) // GRID
        pred_c = int(pred_flat[i]) % GRID
        dist = ((pred_r - tgt_r) ** 2 + (pred_c - tgt_c) ** 2) ** 0.5
        if dist <= bbox_thresh:
            correct += 1
    return correct / len(kpts_src)


class SPair71kHandler(DatasetHandler):
    name = "spair71k"
    default_train_path = "data/spair71k"
    default_test_path  = "data/spair71k"
    default_max_tokens = 0

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        root = _ensure_downloaded(path)
        items = _load_split(root, split)
        if start_index:
            items = items[start_index:]
        if max_samples is not None:
            items = items[:max_samples]
        return items

    def compute_reward(self, response: str, ground_truth: str) -> float:
        return float(response)

    def extract_answer(self, response: str) -> str:
        return response

    def is_answer_correct(self, response: str, ground_truth: str) -> bool:
        return float(response) >= 0.5
