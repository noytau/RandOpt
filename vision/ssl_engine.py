"""SSLEngine: model-access layer for SSL vision models in RandOpt.

The SSL analogue of core/engine.py + utils/worker_extn.py collapsed into one
class: vLLM owns the LLM inside its worker (forcing the worker_extension_cls
graft + collective_rpc tunnel), but a torch vision model needs no middleman —
a Ray actor OWNS the model, so weight access is direct attribute access and
every WorkerExtension method becomes a plain actor method.

Model source: the PINNED official dinov2 clone (never vendored, never edited —
CLAUDE.md rule), resolved via $DINOV2_DIR, + Meta's released ImageNet-1k
linear head ([CLS ; avg-pooled patches] -> 1000). This makes the RandOpt
center byte-identical to the model behind our verified baselines
(87.02% clean / 82.09% ImageNet-C top-1).

Perturbation scheme: copied VERBATIM from WorkerExtension.perturb_self_weights
— per parameter, a FRESH torch.Generator seeded with `seed`, noise of the
param's shape, p += sign*sigma*noise; restore regenerates the identical noise
and subtracts. One (seed, sigma) semantics across the LLM and SSL paths.
NB restore is NEAR-exact, not bit-exact — float rounding leaves ~1 ulp drift
(measured max 2.4e-7 abs on ViT-g); the drift-free path is
store_base_weights/reset_to_base_weights, whose snapshot lives in CPU RAM
(a GPU-resident fp32 copy of ViT-g would not fit an 11GB card next to the
model + activations).

SSLEngineImpl is a plain class (unit-testable without a Ray cluster);
SSLEngine wraps it as a 1-GPU Ray actor.
"""
import gc
import os
import sys
from typing import Dict, List, Optional

import ray
import torch
import torch.nn as nn

_HEAD_URL = ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/"
             "dinov2_vitg14_reg4_linear_head.pth")


def dinov2_repo_dir() -> str:
    """Pinned official clone: $DINOV2_DIR, else per-server convention."""
    for d in (os.environ.get("DINOV2_DIR"),
              "/mnt5/noy/dinov2",
              "/storage/noy/dinov2"):
        if d and os.path.isdir(d):
            return d
    raise FileNotFoundError(
        "dinov2 clone not found — git clone facebookresearch/dinov2 and/or "
        "set DINOV2_DIR")


class SSLEngineImpl:
    """DINOv2 backbone + released linear head with RandOpt weight perturbation.

    perturb_target: "all" | "head" | "last_n_blocks"
    input_mode:     "presized224" (ImageNet-C protocol: normalize only)
                  | "official_resize" (Resize 256 -> CenterCrop 224, clean val)
    """

    def __init__(
        self,
        backbone_name: str = "dinov2_vitg14_reg",
        head_url: str = _HEAD_URL,
        inference_batch_size: int = 16,
        perturb_target: str = "all",
        last_n_blocks: int = 0,
        input_mode: str = "presized224",
    ):
        repo = dinov2_repo_dir()
        sys.path.insert(0, repo)
        from dinov2.data.transforms import (make_classification_eval_transform,
                                            make_normalize_transform)
        from torchvision import transforms

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inference_batch_size = inference_batch_size

        self.backbone = torch.hub.load(repo, backbone_name, source="local")
        self.backbone = self.backbone.to(self.device).eval()
        embed_dim = self.backbone.embed_dim

        self.head = nn.Linear(2 * embed_dim, 1000)
        self.head.load_state_dict(
            torch.hub.load_state_dict_from_url(head_url, map_location="cpu"))
        self.head = self.head.to(self.device).eval()

        # crop is NOT needed for ImageNet-C (Hendrycks ships presized 224x224:
        # normalize only) but IS needed for raw clean-ImageNet JPEGs (variable
        # size -> official Resize 256 -> CenterCrop 224). Both transforms are
        # built so one engine can score and test on different datasets;
        # input_mode picks the default, predict() can override per call.
        self.transforms = {
            "presized224": transforms.Compose(
                [transforms.ToTensor(), make_normalize_transform()]),
            "official_resize": make_classification_eval_transform(),
        }
        if input_mode not in self.transforms:
            raise ValueError(f"unknown input_mode '{input_mode}'")
        self.default_input_mode = input_mode

        self.perturb_target = perturb_target
        self.last_n_blocks = last_n_blocks
        self._base_weights: Optional[dict] = None

    # ------------------------------------------------------------------
    # Inference (replaces vLLM.generate): manifest items -> label strings,
    # directly consumable by handler.compute_reward / extract_answer.
    # ------------------------------------------------------------------

    def predict(self, items: List[Dict], input_mode: str = None) -> List[str]:
        """Manifest items ({"image_path": ...}) -> predicted labels as strings.

        input_mode overrides the engine default for this call (e.g. score on
        clean ImageNet with "official_resize", test on IC with "presized224").
        """
        from data_handlers.imagenet_c import load_image_batch
        transform = self.transforms[input_mode or self.default_input_mode]
        preds: List[str] = []
        for i in range(0, len(items), self.inference_batch_size):
            batch = load_image_batch(items[i:i + self.inference_batch_size],
                                     transform).to(self.device)
            with torch.no_grad():
                f = self.backbone.forward_features(batch)
                feat = torch.cat([f["x_norm_clstoken"],
                                  f["x_norm_patchtokens"].mean(dim=1)], dim=1)
                labels = self.head(feat.float()).argmax(dim=1).cpu().tolist()
            preds.extend(str(l) for l in labels)
        return preds

    # ------------------------------------------------------------------
    # Scope selection (replaces WorkerExtension._should_perturb)
    # ------------------------------------------------------------------

    def _all_params(self):
        yield from self.backbone.named_parameters()
        for name, p in self.head.named_parameters():
            yield f"head.{name}", p

    def _perturb_params(self):
        if self.perturb_target == "head":
            for name, p in self.head.named_parameters():
                yield f"head.{name}", p
        elif self.perturb_target == "last_n_blocks":
            # official dinov2 hub models (block_chunks=0): "blocks.{i}.*"
            n_blocks = len(self.backbone.blocks)
            keep = set(range(n_blocks - self.last_n_blocks, n_blocks))
            for name, p in self.backbone.named_parameters():
                parts = name.split(".")
                if parts[0] == "blocks" and int(parts[1]) in keep:
                    yield name, p
        elif self.perturb_target == "all":
            yield from self._all_params()
        else:
            raise ValueError(f"unknown perturb_target '{self.perturb_target}'")

    def set_perturb_scope(self, perturb_target: str, last_n_blocks: int = 0):
        self.perturb_target = perturb_target
        self.last_n_blocks = last_n_blocks

    def count_perturb_params(self) -> int:
        return sum(p.numel() for _n, p in self._perturb_params())

    # ------------------------------------------------------------------
    # Perturbation (bodies verbatim from WorkerExtension.perturb_self_weights
    # / restore_self_weights — one (seed, sigma) semantics project-wide)
    # ------------------------------------------------------------------

    def perturb_weights(self, seed: int, sigma: float, negate: bool = False):
        sign = -1.0 if negate else 1.0
        for _name, p in self._perturb_params():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device,
                                generator=gen)
            p.data.add_(sign * float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def restore_weights(self, seed: int, sigma: float, negate: bool = False):
        """Undo perturb_weights. Must use the same seed/sigma/negate."""
        sign = -1.0 if negate else 1.0
        for _name, p in self._perturb_params():
            gen = torch.Generator(device=p.device)
            gen.manual_seed(int(seed))
            noise = torch.randn(p.shape, dtype=p.dtype, device=p.device,
                                generator=gen)
            p.data.add_(-sign * float(sigma) * noise)
            del noise
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    # ------------------------------------------------------------------
    # Base snapshot (drift-free ensemble path, as in WorkerExtension)
    # ------------------------------------------------------------------

    def store_base_weights(self):
        # snapshot on CPU: a second fp32 ViT-g on an 11GB GPU would OOM
        self._base_weights = {n: p.data.detach().cpu().clone()
                              for n, p in self._all_params()}
        return True

    def reset_to_base_weights(self):
        for n, p in self._all_params():
            p.data.copy_(self._base_weights[n].to(p.device, non_blocking=True))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def cleanup_gpu_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True


SSLEngine = ray.remote(num_gpus=1)(SSLEngineImpl)


def launch_ssl_engines(num_engines: int, **engine_kwargs):
    """Launch N SSLEngine actors (1 GPU each), mirroring launch_engines'
    GPU-count check and store_base_weights readiness barrier."""
    available = int(ray.cluster_resources().get("GPU", 0))
    if available < num_engines:
        print(f"WARNING: {num_engines} engines requested, {available} GPUs "
              f"available — reducing to {available}.")
        num_engines = available
    if num_engines == 0:
        raise RuntimeError("no GPUs available in the Ray cluster")
    engines = [SSLEngine.remote(**engine_kwargs) for _ in range(num_engines)]
    ray.get([e.store_base_weights.remote() for e in engines])  # readiness
    print(f"{num_engines} SSL engines ready")
    return engines
