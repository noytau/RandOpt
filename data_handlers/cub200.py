"""CUB-200-2011 dataset handler for vision RandOpt.

200 bird species. DINOv2-base linear probe achieves ~82% — good gap for RandOpt.
DINOv2 was shown to learn bird part features unsupervisedly, making this a strong
test of whether weight perturbations can unlock finer species discrimination.
"""
from typing import Dict, List, Optional

import torch
from torchvision import transforms

from .base import DatasetHandler

_to_tensor = transforms.ToTensor()

CUB200_NUM_CLASSES = 200


def _load_via_hf(split: str):
    from datasets import load_dataset
    hf_split = "train" if split == "train" else "test"
    return load_dataset("nateraw/caltech-ucsd-birds-200-2011",
                        split=hf_split, trust_remote_code=False)


class CUB200Handler(DatasetHandler):
    name = "cub200"
    default_train_path = "data/cub200"
    default_test_path = "data/cub200"
    default_max_tokens = 0

    CLASSES: List[str] = []

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        try:
            hf_ds = _load_via_hf(split)
        except Exception as e:
            raise RuntimeError(
                f"CUB-200 HF load failed ({e}). "
                "Ensure datasets package is installed and HF_HOME is set."
            ) from e
        if not CUB200Handler.CLASSES:
            all_labels = sorted(set(hf_ds["label"]))
            CUB200Handler.CLASSES = [str(i) for i in all_labels]
        items = []
        for idx in range(start_index, len(hf_ds)):
            if max_samples is not None and len(items) >= max_samples:
                break
            row = hf_ds[idx]
            img = _to_tensor(row["image"].convert("RGB"))
            label = row["label"]
            items.append({
                "image_tensor": img,
                "ground_truth": str(label),
                "class_id": label,
                "messages": [],
            })
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
