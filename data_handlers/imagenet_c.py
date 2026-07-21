"""ImageNet-C dataset handler (image classification via manifest JSON).

The manifest (data/imagenet_c/data.json, written by
scripts/make_imagenet_c_manifest.py) links to image paths and labels:
    {"image": "/abs/path/….JPEG", "label": 17, "wnid": "n01440764",
     "split": "train"}
Images are decoded lazily at consumption time (load_image / load_image_batch),
never embedded in the items.
"""
import json
from typing import Dict, List, Optional

from torchvision import transforms

from utils.reward_score import imagenet_c as imagenet_c_reward
from .base import DatasetHandler

_to_tensor = transforms.ToTensor()


def load_image(item: Dict, transform=None):
    """Decode one manifest item -> tensor (default ToTensor; pass an eval
    transform to resize/normalize in the same step)."""
    from PIL import Image
    img = Image.open(item["image_path"]).convert("RGB")
    return (transform or _to_tensor)(img)


def load_image_batch(items: List[Dict], transform=None):
    """Decode manifest items -> stacked (N,C,H,W) tensor. Images must share a
    size after `transform` (ImageNet-C ships pre-sized 224x224)."""
    import torch
    return torch.stack([load_image(d, transform) for d in items])


class ImageNetCHandler(DatasetHandler):
    name = "imagenet_c"
    default_train_path = "data/imagenet_c/data.json"
    default_test_path = "data/imagenet_c/data.json"
    default_max_tokens = 16

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        with open(path) as f:
            raw = json.load(f)
        out = []
        for item in raw:
            if item["split"] != split:
                continue
            out.append({
                "image_path": item["image"],
                "ground_truth": str(item["label"]),
                "class_id": item["label"],
                "messages": [],
            })
        if start_index:
            out = out[start_index:]
        if max_samples:
            out = out[:max_samples]
        return out

    def compute_reward(self, response: str, ground_truth: str) -> float:
        return imagenet_c_reward.compute_score(response, ground_truth)

    def extract_answer(self, response: str) -> str:
        return imagenet_c_reward.extract_answer(response)
