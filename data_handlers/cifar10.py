"""CIFAR-10 dataset handler for vision RandOpt."""
from typing import Dict, List, Optional

import torch
from torchvision import transforms

from .base import DatasetHandler

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_to_tensor = transforms.ToTensor()


def _load_via_hf(split: str) -> list:
    """Load CIFAR-10 via HuggingFace datasets (avoids slow Toronto download)."""
    from datasets import load_dataset
    ds = load_dataset("uoft-cs/cifar10", split="train" if split == "train" else "test",
                      trust_remote_code=True)
    return ds


def _load_via_torchvision(path: str, split: str) -> object:
    from torchvision import datasets as tvd
    return tvd.CIFAR10(root=path, train=(split == "train"), download=True, transform=_to_tensor)


class CIFAR10Handler(DatasetHandler):
    name = "cifar10"
    default_train_path = "data/cifar10"
    default_test_path = "data/cifar10"
    default_max_tokens = 0

    CLASSES = CIFAR10_CLASSES

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        """Load CIFAR-10. Tries HuggingFace datasets first (faster CDN), falls back to torchvision."""
        try:
            hf_ds = _load_via_hf(split)
            items = []
            for idx in range(start_index, len(hf_ds)):
                if max_samples is not None and len(items) >= max_samples:
                    break
                row = hf_ds[idx]
                # HF CIFAR-10: row["img"] is a PIL image, row["label"] is int
                img = _to_tensor(row["img"])
                label = row["label"]
                items.append({
                    "image_tensor": img,
                    "ground_truth": CIFAR10_CLASSES[label],
                    "class_id": label,
                    "messages": [],
                })
            return items
        except Exception as e:
            print(f"HF datasets load failed ({e}), falling back to torchvision...")

        dataset = _load_via_torchvision(path, split)
        items = []
        for idx in range(start_index, len(dataset)):
            if max_samples is not None and len(items) >= max_samples:
                break
            img, label = dataset[idx]
            items.append({
                "image_tensor": img,
                "ground_truth": CIFAR10_CLASSES[label],
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
        """predictions: list of predicted class IDs (int) from VisionEngine.forward()."""
        if not task_datas:
            return 0.0
        correct = sum(
            1 for pred, data in zip(predictions, task_datas)
            if pred == data["class_id"]
        )
        return correct / len(task_datas)
