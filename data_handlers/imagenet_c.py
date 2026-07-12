"""ImageNet-C dataset handler for vision RandOpt (E6).

ImageNet-C (Hendrycks & Dietterich 2019): the ImageNet-1k val set (50k images)
with 15 corruption types x 5 severities applied, distributed as 224x224 JPEGs
(DINOv2's native input size — no resizing needed). Source: the canonical
Zenodo tarballs (record 2235448), one tar per corruption family; we download
only the family tar a requested corruption lives in and extract only the
requested corruption/severity subtree (~1GB of a 7-21GB tar).

Directory layout after extraction:  <data_dir>/<corruption>/<severity>/<wnid>/*.JPEG

Labels: class ids are assigned by sorted wnid order (0..999). For kNN-based
E6 run 1 only internal consistency matters; this ordering also happens to
match torchvision's ImageFolder convention and ImageNet's standard class
index, which the future pretrained-head baseline (TASKS.md) will need.

Splits (class-stratified, fixed seed, disjoint):
  "adaptation"  adapt_per_class images/class (default 2  -> M=2,000)
  "test"        test_per_class images/class  (default 5  -> 5,000)
The E6 spec's optional A/B honesty splits are carved by the runner out of
extra adaptation-phase data, not here (OFF in run 1).
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from torchvision import transforms

from .base import DatasetHandler

_to_tensor = transforms.ToTensor()

ZENODO_URL = "https://zenodo.org/records/2235448/files/{tar}?download=1"

# corruption -> Zenodo tar file that contains it
CORRUPTION_TO_TAR = {
    "gaussian_noise": "noise.tar", "shot_noise": "noise.tar",
    "impulse_noise": "noise.tar",
    "defocus_blur": "blur.tar", "glass_blur": "blur.tar",
    "motion_blur": "blur.tar", "zoom_blur": "blur.tar",
    "snow": "weather.tar", "frost": "weather.tar", "fog": "weather.tar",
    "brightness": "weather.tar",
    "contrast": "digital.tar", "elastic_transform": "digital.tar",
    "pixelate": "digital.tar", "jpeg_compression": "digital.tar",
}


def _ensure_downloaded(data_dir: str, corruption: str, severity: int) -> str:
    """Download+extract one corruption/severity subtree if not already present.

    Returns the path <data_dir>/<corruption>/<severity>/ containing wnid dirs.
    The family tar is kept after extraction (other corruptions/severities in
    it may be needed later); delete manually if PVC space matters.
    """
    if corruption not in CORRUPTION_TO_TAR:
        raise ValueError(f"Unknown corruption '{corruption}'. "
                         f"Known: {sorted(CORRUPTION_TO_TAR)}")
    sev_dir = Path(data_dir) / corruption / str(severity)
    if sev_dir.is_dir() and any(sev_dir.iterdir()):
        return str(sev_dir)

    import subprocess
    os.makedirs(data_dir, exist_ok=True)
    tar_name = CORRUPTION_TO_TAR[corruption]
    tar_path = Path(data_dir) / tar_name
    if not tar_path.exists():
        url = ZENODO_URL.format(tar=tar_name)
        print(f"Downloading ImageNet-C {tar_name} from Zenodo (may be large)...")
        subprocess.run(["wget", "-c", "-q", "-O", str(tar_path), url], check=True)
    print(f"Extracting {corruption}/{severity} from {tar_name} ...")
    subprocess.run(["tar", "-xf", str(tar_path), "-C", data_dir,
                    f"{corruption}/{severity}"], check=True)
    if not (sev_dir.is_dir() and any(sev_dir.iterdir())):
        raise RuntimeError(f"Extraction produced no files at {sev_dir} — "
                           f"check the tar's internal layout with 'tar -tf'.")
    return str(sev_dir)


class ImageNetCHandler(DatasetHandler):
    """splits: 'adaptation' (RandOpt scoring + baseline training) / 'test'."""
    name = "imagenet_c"
    default_train_path = "data/imagenet_c/gaussian_noise/3"
    default_test_path = "data/imagenet_c/gaussian_noise/3"
    default_max_tokens = 0

    adapt_per_class = 2   # M = 2,000 over 1000 classes (E6 spec default)
    test_per_class = 5    # test = 5,000
    split_seed = 42

    def load_data(
        self,
        path: str,
        split: str = "adaptation",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        """`path` = <data_dir>/<corruption>/<severity>; downloads if missing."""
        from PIL import Image

        p = Path(path)
        corruption, severity = p.parts[-2], int(p.parts[-1])
        sev_dir = Path(_ensure_downloaded(str(p.parents[1]), corruption, severity))

        if split not in ("adaptation", "test"):
            raise ValueError(f"split must be 'adaptation' or 'test', got '{split}'")

        rng = np.random.default_rng(self.split_seed)
        items = []
        for label, wnid_dir in enumerate(sorted(sev_dir.iterdir())):
            files = sorted(f for f in wnid_dir.iterdir() if f.suffix.lower()
                           in (".jpeg", ".jpg", ".png"))
            order = rng.permutation(len(files))
            if split == "adaptation":
                picks = order[:self.adapt_per_class]
            else:
                picks = order[self.adapt_per_class:
                              self.adapt_per_class + self.test_per_class]
            for i in picks:
                img = Image.open(files[i]).convert("RGB")
                items.append({
                    "image_tensor": _to_tensor(img),
                    "ground_truth": str(label),
                    "class_id": label,
                    "messages": [],
                })
        if start_index:
            items = items[start_index:]
        if max_samples is not None:
            items = items[:max_samples]
        return items

    def compute_reward(self, response: str, ground_truth: str) -> float:
        return 1.0 if response == ground_truth else 0.0

    def extract_answer(self, response: str) -> str:
        return response

    def is_answer_correct(self, response: str, ground_truth: str) -> bool:
        return response == ground_truth

    def postprocess_outputs(self, predictions: List[int], task_datas: List[Dict]) -> float:
        if not task_datas:
            return 0.0
        correct = sum(
            1 for pred, data in zip(predictions, task_datas)
            if pred == data["class_id"]
        )
        return correct / len(task_datas)
