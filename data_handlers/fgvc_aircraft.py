"""FGVC-Aircraft dataset handler for vision RandOpt.

100 aircraft variant classes (e.g. Boeing 737-800 vs 737-900).
DINOv2-base linear probe achieves ~61% — meaningful gap for RandOpt to exploit.
"""
from typing import Dict, List, Optional

import torch
from torchvision import transforms

from .base import DatasetHandler

# Resize to fixed size at load time — FGVC images have variable native dimensions
_to_tensor = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
])


def _load_via_hf(split: str):
    from datasets import load_dataset
    dataset_name = ("Multimodal-Fatima/FGVC_Aircraft_train_dataset" if split == "train"
                    else "Multimodal-Fatima/FGVC_Aircraft_test_dataset")
    return load_dataset(dataset_name, split="train", trust_remote_code=False)


def _load_via_torchvision(path: str, split: str):
    from torchvision.datasets import FGVCAircraft
    tv_split = "trainval" if split == "train" else "test"
    return FGVCAircraft(root=path, split=tv_split,
                        annotation_level="variant", download=True, transform=_to_tensor)


class FGVCAircraftHandler(DatasetHandler):
    name = "fgvc_aircraft"
    default_train_path = "data/fgvc_aircraft"
    default_test_path = "data/fgvc_aircraft"
    default_max_tokens = 0

    # 100 aircraft variant classes
    CLASSES: List[str] = []

    def load_data(
        self,
        path: str,
        split: str = "train",
        max_samples: Optional[int] = None,
        start_index: int = 0,
    ) -> List[Dict]:
        # Try HuggingFace first
        try:
            hf_ds = _load_via_hf(split)
            # Build class list from dataset on first load
            if not FGVCAircraftHandler.CLASSES:
                all_labels = set(hf_ds["label"])
                FGVCAircraftHandler.CLASSES = [str(i) for i in sorted(all_labels)]
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
        except Exception as e:
            print(f"HF load failed ({e}), falling back to torchvision...")

        tv_ds = _load_via_torchvision(path, split)
        if not FGVCAircraftHandler.CLASSES:
            FGVCAircraftHandler.CLASSES = tv_ds.classes
        items = []
        for idx in range(start_index, len(tv_ds)):
            if max_samples is not None and len(items) >= max_samples:
                break
            img, label = tv_ds[idx]
            items.append({
                "image_tensor": img,
                "ground_truth": FGVCAircraftHandler.CLASSES[label],
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
