"""VisionEngine: Ray actor wrapping DINOv2 backbone + linear head with RandOpt perturbation."""
import gc
import os
import sys
from typing import List, Optional

# Ensure repo root is importable inside Ray worker processes (for data_handlers)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        perturb_target: str = "all",  # "all" | "classifier" | "last_n_blocks"
        last_n_blocks: int = 0,       # used when perturb_target == "last_n_blocks"
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

        self.perturb_target = perturb_target  # "all" | "classifier" | "last_n_blocks"
        self.last_n_blocks = last_n_blocks
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
        """Params to perturb, depending on perturb_target:
        "all"           -> backbone + classifier
        "classifier"    -> linear head only
        "last_n_blocks" -> only the last N transformer blocks of the backbone
        """
        if self.perturb_target == "classifier":
            for name, p in self.classifier.named_parameters():
                yield f"cls.{name}", p
        elif self.perturb_target == "last_n_blocks":
            num_layers = self.backbone.config.num_hidden_layers
            keep = set(range(num_layers - self.last_n_blocks, num_layers))
            for name, p in self.backbone.named_parameters():
                # DINOv2 block params are named "encoder.layer.{i}.*"
                parts = name.split(".")
                if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layer" \
                        and int(parts[2]) in keep:
                    yield name, p
        else:
            yield from self._all_params()

    def set_perturb_scope(self, perturb_target: str, last_n_blocks: int = 0) -> None:
        """Switch the perturbation scope on a live engine (avoids reloading the model)."""
        self.perturb_target = perturb_target
        self.last_n_blocks = last_n_blocks

    def count_perturb_params(self) -> int:
        """Number of scalar parameters currently in scope for perturbation."""
        return sum(p.numel() for _n, p in self._perturb_params())

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

    def get_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch token embeddings for semantic correspondence.

        Args:
            images: (N, 3, H, W) float32 in [0, 1]
        Returns:
            (N, num_patches, embed_dim) float32 — L2-normalized patch features.
            num_patches = (224/14)^2 = 256 for DINOv2-base.
        """
        all_feats = []
        for i in range(0, len(images), self.inference_batch_size):
            batch = images[i : i + self.inference_batch_size].to(self.device)
            batch = self._preprocess(batch)
            with torch.no_grad():
                out = self.backbone(pixel_values=batch, output_hidden_states=False)
                # last_hidden_state: (B, 1+num_patches, D) — first token is CLS
                patch_tokens = out.last_hidden_state[:, 1:, :]  # (B, P, D)
                patch_tokens = F.normalize(patch_tokens, dim=-1)
            all_feats.append(patch_tokens.cpu())
        return torch.cat(all_feats, dim=0)

    def _patch_features_gpu(self, images: torch.Tensor) -> torch.Tensor:
        """Like get_patch_features but keeps the (N, P, D) tensor on the GPU."""
        all_feats = []
        for i in range(0, len(images), self.inference_batch_size):
            batch = images[i : i + self.inference_batch_size].to(self.device)
            batch = self._preprocess(batch)
            with torch.no_grad():
                out = self.backbone(pixel_values=batch, output_hidden_states=False)
                patch_tokens = out.last_hidden_state[:, 1:, :]
                patch_tokens = F.normalize(patch_tokens, dim=-1)
            all_feats.append(patch_tokens)
        return torch.cat(all_feats, dim=0)

    def eval_pck(self, src_imgs, tgt_imgs, kpts_src, kpts_tgt, bbox_thresh) -> float:
        """Compute mean PCK@0.1 entirely on the GPU and return only the scalar.

        Features never leave the GPU — only the final float crosses the Ray
        boundary, avoiding ~600MB of feature-tensor transfer per call.

        Args:
            src_imgs, tgt_imgs: (N,3,H,W) float32 in [0,1]
            kpts_src, kpts_tgt: length-N lists of [(row,col), ...] keypoints
            bbox_thresh: length-N list of per-pair PCK thresholds (patch units)
        """
        from data_handlers.spair71k import compute_pck
        # Bulk-copy features to CPU ONCE (a single ~25ms PCIe transfer inside the
        # actor), then run PCK on CPU. This avoids BOTH the Ray serialization of
        # the 157MB tensors AND the per-keypoint GPU->CPU sync that int() would
        # trigger if compute_pck ran on GPU tensors.
        sf = self._patch_features_gpu(src_imgs).cpu()
        tf = self._patch_features_gpu(tgt_imgs).cpu()
        scores = [compute_pck(sf[i], tf[i], kpts_src[i], kpts_tgt[i],
                              bbox_thresh=bbox_thresh[i])
                  for i in range(len(kpts_src))]
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _cls_features_gpu(self, images: torch.Tensor) -> torch.Tensor:
        """(N,3,H,W) [0,1] -> (N, D) L2-normalized CLS tokens, kept on GPU."""
        all_feats = []
        for i in range(0, len(images), self.inference_batch_size):
            batch = images[i : i + self.inference_batch_size].to(self.device)
            batch = self._preprocess(batch)
            with torch.no_grad():
                out = self.backbone(pixel_values=batch)
                cls = F.normalize(out.last_hidden_state[:, 0, :], dim=-1)
            all_feats.append(cls)
        return torch.cat(all_feats, dim=0)

    def eval_global(self, gallery_imgs, gallery_labels, query_sets,
                    k: int = 20, tau: float = 0.07):
        """kNN top-1 + retrieval mAP for each query set against one gallery.

        The gallery is forwarded ONCE and reused across query sets (the A/B
        splits), matching the shared-forward design of the thicket profile.

        Args:
            gallery_imgs: (G,3,H,W) float32 in [0,1]
            gallery_labels: length-G list of int class ids
            query_sets: list of (imgs, labels) tuples
            k: kNN neighbors (DINO-style weighted vote)
            tau: vote temperature
        Returns:
            list of {"knn_top1": float, "map": float}, one per query set.
        """
        g = self._cls_features_gpu(gallery_imgs)                       # (G, D)
        gl = torch.tensor(gallery_labels, device=self.device)
        num_classes = int(gl.max()) + 1
        results = []
        for imgs, labels in query_sets:
            q = self._cls_features_gpu(imgs)                           # (Q, D)
            ql = torch.tensor(labels, device=self.device)
            sim = q @ g.T                                              # (Q, G)
            # DINO-style weighted kNN vote
            topv, topi = sim.topk(min(k, sim.shape[1]), dim=1)
            w = (topv / tau).exp()
            votes = torch.zeros(len(q), num_classes, device=self.device)
            votes.scatter_add_(1, gl[topi], w)
            top1 = (votes.argmax(dim=1) == ql).float().mean().item()
            # retrieval mAP (binary relevance = same class)
            order = sim.argsort(dim=1, descending=True)
            rel = (gl[order] == ql[:, None]).float()                   # (Q, G)
            ranks = torch.arange(1, rel.shape[1] + 1,
                                 device=self.device, dtype=torch.float)
            prec = rel.cumsum(dim=1) / ranks
            ap = (prec * rel).sum(dim=1) / rel.sum(dim=1).clamp(min=1)
            results.append({"knn_top1": top1, "map": float(ap.mean())})
        return results

    def get_embed_dim(self) -> int:
        return self.backbone.config.hidden_size

    def cleanup(self) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
