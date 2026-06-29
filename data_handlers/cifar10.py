"""CIFAR-10 dataset handler for vision RandOpt."""
from typing import Dict, List, Optional

import torch
from torchvision import datasets, transforms

from .base import DatasetHandler

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

_to_tensor = transforms.ToTensor()


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
        """Load CIFAR-10; returns dicts with float32 (3,32,32) image tensors in [0,1]."""
        dataset = datasets.CIFAR10(
            root=path, train=(split == "train"), download=True, transform=_to_tensor
        )
        items = []
        for idx in range(start_index, len(dataset)):
            if max_samples is not None and len(items) >= max_samples:
                break
            img, label = dataset[idx]
            items.append({
                "image_tensor": img,              # float32 (3, 32, 32) in [0, 1]
                "ground_truth": CIFAR10_CLASSES[label],
                "class_id": label,
                "messages": [],                   # not used for vision
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
