"""VisionEngine: Ray actor wrapping DINOv2 backbone + linear head with RandOpt perturbation."""
import gc
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import ray


@ray.remote(num_gpus=1)
class VisionEngine:
    """Ray actor that holds a DINOv2 backbone + linear classifier on one GPU.

    Supports in-place weight perturbation/restoration using the same deterministic
    seed-based Gaussian noise scheme as WorkerExtension in the LLM path.
    """

    # ImageNet mean/std used by DINOv2 pretraining
    _NORM_MEAN = [0.485, 0.456, 0.406]
    _NORM_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        model_name: str,
        num_classes: int,
        linear_init_path: Optional[str] = None,
        inference_batch_size: int = 64,
        perturb_target: str = "all",  # "all" | "classifier"
    ):
        from transformers import Dinov2Model

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_classes = num_classes
        self.inference_batch_size = inference_batch_size

        self.backbone = Dinov2Model.from_pretrained(model_name).to(self.device)
        self.backbone.eval()

        embed_dim = self.backbone.config.hidden_size
        self.classifier = nn.Linear(embed_dim, num_classes).to(self.device)

        if linear_init_path:
            state = torch.load(linear_init_path, map_location=self.device)
            self.classifier.load_state_dict(state)

        # Precompute normalization tensors
        self._norm_mean = torch.tensor(self._NORM_MEAN, device=self.device).view(1, 3, 1, 1)
        self._norm_std  = torch.tensor(self._NORM_STD,  device=self.device).view(1, 3, 1, 1)

        self.perturb_target = perturb_target  # "all" | "classifier"
        self._base_weights: Optional[dict] = None
        self.store_base_weights()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """images: (N,3,H,W) float32 [0,1] → resized+normalized for DINOv2."""
        x = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False)
        return (x - self._norm_mean) / self._norm_std

    def forward(self, images: torch.Tensor) -> List[int]:
        """Run inference on a batch of images.

        Args:
            images: (N, 3, H, W) float32 in [0, 1]
        Returns:
            List of N predicted class IDs (int).
        """
        all_preds: List[int] = []
        for i in range(0, len(images), self.inference_batch_size):
            batch = images[i : i + self.inference_batch_size].to(self.device)
            batch = self._preprocess(batch)
            with torch.no_grad():
                embeddings = self.backbone(pixel_values=batch).pooler_output
                logits = self.classifier(embeddings)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
        return all_preds

    # ------------------------------------------------------------------
    # Perturbation — mirrors worker_extn.py:perturb_self_weights()
    # Each parameter gets noise generated from a fresh Generator seeded
    # with `seed`, making perturb/restore exactly invertible.
    # ------------------------------------------------------------------

    def _all_params(self):
        """Yield (name, param) for backbone then classifier in consistent order."""
        yield from self.backbone.named_parameters()
        for name, p in self.classifier.named_parameters():
            yield f"cls.{name}", p

    def _perturb_params(self):
        """Params to perturb: all weights, or classifier-only depending on perturb_target."""
        if self.perturb_target == "classifier":
            for name, p in self.classifier.named_parameters():
                yield f"cls.{name}", p
        else:
            yield from self._all_params()

    def perturb_weights(self, seed: int, sigma: float) -> None:
        for _name, p in self._perturb_params():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(seed)
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(sigma * noise)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    def restore_weights(self, seed: int, sigma: float) -> None:
        for _name, p in self._perturb_params():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(seed)
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device, generator=gen)
            p.data.add_(-sigma * noise)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Base weight snapshot (for ensemble path that avoids accumulation drift)
    # ------------------------------------------------------------------

    def store_base_weights(self) -> None:
        self._base_weights = {name: p.data.clone() for name, p in self._all_params()}

    def reset_to_base(self) -> None:
        for name, p in self._all_params():
            p.data.copy_(self._base_weights[name])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def get_embed_dim(self) -> int:
        return self.backbone.config.hidden_size

    def cleanup(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
