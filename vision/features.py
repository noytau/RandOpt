"""Shared frozen-backbone feature cache (E1, experiments/E1_imagenet_c.md).

kNN and the linear probe never modify the backbone, so each image's embedding
is deterministic: compute it once, cache it to disk, and every later k-sweep /
probe run is pure matrix math on the cached vectors (no GPU forwards).

Two feature types are stored from the SAME forward pass, because the two
consumers conventionally use different tokens:
  cls     last_hidden_state[:, 0, :] raw  — kNN path; L2-normalized at use.
          Exact parity with VisionEngine._cls_features_gpu (vision/engine.py).
  pooler  pooler_output                   — probe path; parity with
          vision/train_linear_probe.py and VisionEngine.forward.

Preprocessing (bicubic resize to 224 + ImageNet mean/std) mirrors
VisionEngine._preprocess so cached features match what the RandOpt engine
computes for the unperturbed model.

Cache layout: <cache_dir>/<model_short>/<corruption>-s<severity>-<split>.pt
holding {"cls": (N,D) fp16, "pooler": (N,D) fp16, "labels": LongTensor(N)}.

FT and RandOpt must NOT use this cache inside their core loops — they move the
backbone weights, so features change every step/perturbation.
"""
import os
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

_NORM_MEAN = [0.485, 0.456, 0.406]  # = VisionEngine._NORM_MEAN
_NORM_STD = [0.229, 0.224, 0.225]   # = VisionEngine._NORM_STD


def cache_path(cache_dir: str, model_name: str, corruption: str,
               severity: int, split: str) -> Path:
    model_short = model_name.split("/")[-1]
    return Path(cache_dir) / model_short / f"{corruption}-s{severity}-{split}.pt"


def extract_split_features(
    model_name: str,
    corruption: str,
    severity: int,
    split: str,
    data_root: str = "data/imagenet_c",
    cache_dir: str = "results/features",
    batch_size: int = 64,
    device: str = None,
    handler=None,
) -> Dict[str, torch.Tensor]:
    """Return {"cls","pooler","labels"} for one split, computing+caching on miss.

    `handler` lets callers pass a pre-configured ImageNetCHandler (custom
    per-class split sizes); default uses the registry handler with E1 defaults.
    """
    path = cache_path(cache_dir, model_name, corruption, severity, split)
    if path.exists():
        return torch.load(path, map_location="cpu")

    from transformers import Dinov2Model
    from data_handlers import get_dataset_handler

    if handler is None:
        handler = get_dataset_handler("imagenet_c")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    items = handler.load_data(os.path.join(data_root, corruption, str(severity)),
                              split=split)
    print(f"[features] {corruption}/s{severity}/{split}: {len(items)} images, "
          f"extracting with {model_name} on {dev} ...")

    backbone = Dinov2Model.from_pretrained(model_name).to(dev).eval()
    mean = torch.tensor(_NORM_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(_NORM_STD, device=dev).view(1, 3, 1, 1)

    cls_out, pool_out = [], []
    for i in range(0, len(items), batch_size):
        batch = torch.stack([d["image_tensor"]
                             for d in items[i:i + batch_size]]).to(dev)
        batch = F.interpolate(batch, size=(224, 224), mode="bicubic",
                              align_corners=False)
        batch = (batch - mean) / std
        with torch.no_grad():
            out = backbone(pixel_values=batch)
        cls_out.append(out.last_hidden_state[:, 0, :].half().cpu())
        pool_out.append(out.pooler_output.half().cpu())

    data = {
        "cls": torch.cat(cls_out),
        "pooler": torch.cat(pool_out),
        "labels": torch.tensor([d["class_id"] for d in items], dtype=torch.long),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"[features] cached -> {path}")
    return data
