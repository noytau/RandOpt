"""SPair-71k dataset handler for DINOv2 semantic correspondence.

SPair-71k: ~70k image pairs across 18 semantic categories with keypoint annotations.
Evaluation metric: PCK@0.1 (Percentage of Correct Keypoints within 10% of bbox size).

No linear head needed — reward is computed from cosine similarity between patch features.
DINOv2-base baseline PCK@0.1 ~64%, with headroom to ~82% (SOTA with learned heads).

Reference: Min et al., "SPair-71k: A Large-scale Benchmark for Semantic Correspondence"
HuggingFace: datasets.load_dataset("jxu124/spair-71k")
"""
import json
import os
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

# DINOv2-base patch grid for 224×224 input: 16×16 = 256 patches (14px stride)
PATCH_SIZE = 14
INPUT_SIZE = 224
GRID = INPUT_SIZE // PATCH_SIZE  # 16


def _ensure_downloaded(data_dir: str) -> Path:
    """Download SPair-71k to data_dir via HuggingFace datasets."""
    from datasets import load_dataset
    cache = Path(data_dir) / "hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return load_dataset("jxu124/spair-71k", cache_dir=str(cache))


def _kpt_to_patch(kx: float, ky: float, img_w: int, img_h: int) -> Tuple[int, int]:
    """Map a keypoint in original image coords to (row, col) in the 16×16 patch grid."""
    # Image was resized to 448 → then engine resizes to 224 (factor of 2)
    scale_x = (INPUT_SIZE / 2) / img_w
    scale_y = (INPUT_SIZE / 2) / img_h
    px = min(int(kx * scale_x / PATCH_SIZE), GRID - 1)
    py = min(int(ky * scale_y / PATCH_SIZE), GRID - 1)
    return py, px  # row, col


def compute_pck(
    feats_src: torch.Tensor,         # (P, D) features OR (P, P) sim matrix
    feats_tgt: Optional[torch.Tensor],  # (P, D) features; None if precomputed_sim=True
    kpts_src: List[Tuple[int, int]],  # (row, col) in patch grid
    kpts_tgt: List[Tuple[int, int]],
    threshold: float = 0.1,
    bbox_size: Optional[float] = None,
    precomputed_sim: bool = False,    # if True, feats_src is a (P, P) sim matrix
) -> float:
    """Compute PCK@threshold for one image pair.

    For each source keypoint, finds the most similar target patch.
    Accepts either raw features or a pre-computed (P, P) similarity matrix
    (used during ensemble averaging).
    """
    if not kpts_src:
        return 0.0

    if bbox_size is None:
        bbox_size = threshold * GRID  # 0.1 * 16 = 1.6 patches

    if precomputed_sim:
        sim_matrix = feats_src  # (P, P) — rows=src, cols=tgt
        src_indices = [r * GRID + c for r, c in kpts_src]
        sim = sim_matrix[src_indices]  # (K, P)
    else:
        src_vecs = torch.stack([feats_src[r * GRID + c] for r, c in kpts_src])  # (K, D)
        sim = src_vecs @ feats_tgt.T  # (K, P)

    pred_flat = sim.argmax(dim=-1)  # (K,)

    correct = 0
    for i, (tgt_r, tgt_c) in enumerate(kpts_tgt):
        pred_r = int(pred_flat[i]) // GRID
        pred_c = int(pred_flat[i]) % GRID
        dist = ((pred_r - tgt_r) ** 2 + (pred_c - tgt_c) ** 2) ** 0.5
        if dist <= bbox_size:
            correct += 1
    return correct / len(kpts_src)


class SPair71kHandler(DatasetHandler):
    name = "spair71k"
    default_train_path = "data/spair71k"
    default_test_path = "data/spair71k"
    default_max_tokens = 0

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        from PIL import Image as PILImage

        ds = _ensure_downloaded(path)
        hf_split = "trn" if split == "train" else "test"
        subset = ds[hf_split]

        items = []
        for idx in range(start_index, len(subset)):
            if max_samples is not None and len(items) >= max_samples:
                break
            row = subset[idx]

            # Load and resize both images
            img_src = _to_tensor(row["src_img"].convert("RGB"))
            img_tgt = _to_tensor(row["trg_img"].convert("RGB"))

            src_w, src_h = row["src_img"].size
            tgt_w, tgt_h = row["trg_img"].size

            # Convert keypoints to patch-grid coords
            kpts_src_raw = row["src_kps"]   # list of [x, y] or similar
            kpts_tgt_raw = row["trg_kps"]

            # SPair-71k stores keypoints as list of [x, y] pairs
            kpts_src = [_kpt_to_patch(k[0], k[1], src_w, src_h) for k in kpts_src_raw]
            kpts_tgt = [_kpt_to_patch(k[0], k[1], tgt_w, tgt_h) for k in kpts_tgt_raw]

            items.append({
                "image_tensor": img_src,       # used for source
                "image_tensor_tgt": img_tgt,   # used for target
                "kpts_src": kpts_src,
                "kpts_tgt": kpts_tgt,
                "category": row.get("category", ""),
                "ground_truth": "pck",          # placeholder, reward computed externally
                "messages": [],
            })
        return items

    def compute_reward(self, response: str, ground_truth: str) -> float:
        # PCK is computed externally via compute_pck(); this is a passthrough
        return float(response)

    def extract_answer(self, response: str) -> str:
        return response

    def is_answer_correct(self, response: str, ground_truth: str) -> bool:
        return float(response) >= 0.5
