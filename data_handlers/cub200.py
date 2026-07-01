"""CUB-200-2011 dataset handler for vision RandOpt.

200 bird species. DINOv2-base linear probe achieves ~82% — good gap for RandOpt.
DINOv2 was shown to learn bird part features unsupervisedly, making this a strong
test of whether weight perturbations can unlock finer species discrimination.

Data is downloaded from Caltech (data.caltech.edu) and cached on PVC.
Run script calls _ensure_downloaded() before the main experiment starts.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional

from torchvision import transforms

from .base import DatasetHandler

_to_tensor = transforms.ToTensor()

CUB_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"
CUB_NUM_CLASSES = 200


def _ensure_downloaded(data_dir: str) -> str:
    """Download and extract CUB-200-2011 to data_dir if not already present.

    Returns path to the CUB_200_2011/ root directory.
    """
    root = Path(data_dir) / "CUB_200_2011"
    if (root / "images.txt").exists():
        return str(root)

    import subprocess
    os.makedirs(data_dir, exist_ok=True)
    tgz = Path(data_dir) / "CUB_200_2011.tgz"
    if not tgz.exists():
        print(f"Downloading CUB-200-2011 from {CUB_URL} ...")
        subprocess.run(["wget", "-q", "-O", str(tgz), CUB_URL], check=True)
    print("Extracting CUB-200-2011 ...")
    subprocess.run(["tar", "-xzf", str(tgz), "-C", data_dir], check=True)
    return str(root)


def _load_from_disk(cub_root: str, split: str) -> List[Dict]:
    """Parse CUB-200-2011 directory structure into a list of dicts."""
    from PIL import Image

    root = Path(cub_root)
    # image_id → filename
    id_to_file = {}
    with open(root / "images.txt") as f:
        for line in f:
            img_id, fname = line.strip().split(" ", 1)
            id_to_file[img_id] = fname
    # image_id → class_id (1-indexed → 0-indexed)
    id_to_label = {}
    with open(root / "image_class_labels.txt") as f:
        for line in f:
            img_id, cls = line.strip().split()
            id_to_label[img_id] = int(cls) - 1
    # image_id → train(1)/test(0)
    id_to_split = {}
    with open(root / "train_test_split.txt") as f:
        for line in f:
            img_id, is_train = line.strip().split()
            id_to_split[img_id] = int(is_train)

    want_train = (split == "train")
    items = []
    for img_id, fname in sorted(id_to_file.items(), key=lambda x: int(x[0])):
        if bool(id_to_split[img_id]) != want_train:
            continue
        img_path = root / "images" / fname
        img = Image.open(img_path).convert("RGB")
        items.append({
            "image_tensor": _to_tensor(img),
            "ground_truth": str(id_to_label[img_id]),
            "class_id": id_to_label[img_id],
            "messages": [],
        })
    return items


class CUB200Handler(DatasetHandler):
    name = "cub200"
    default_train_path = "data/cub200"
    default_test_path = "data/cub200"
    default_max_tokens = 0

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        cub_root = _ensure_downloaded(path)
        items = _load_from_disk(cub_root, split)
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
