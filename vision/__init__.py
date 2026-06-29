"""Vision module for RandOpt: DINOv2-based engines with weight perturbation."""
from typing import List, Optional

import ray

from .engine import VisionEngine


def launch_vision_engines(
    num_engines: int,
    model_name: str,
    num_classes: int,
    linear_init_path: Optional[str] = None,
    inference_batch_size: int = 64,
) -> List:
    """Launch `num_engines` VisionEngine Ray actors, each on one GPU.

    Args:
        num_engines: Number of parallel engines (= number of GPUs to use).
        model_name: HuggingFace model ID, e.g. "facebook/dinov2-base".
        num_classes: Number of output classes for the linear head.
        linear_init_path: Optional path to pretrained linear head .pt file.
        inference_batch_size: Images per GPU forward pass (tune to GPU VRAM).

    Returns:
        List of VisionEngine Ray actor handles.
    """
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    engines = [
        VisionEngine.remote(
            model_name=model_name,
            num_classes=num_classes,
            linear_init_path=linear_init_path,
            inference_batch_size=inference_batch_size,
        )
        for _ in range(num_engines)
    ]
    # Block until all engines are ready
    ray.get([e.get_embed_dim.remote() for e in engines])
    return engines


def cleanup_vision_engines(engines: List) -> None:
    ray.get([e.cleanup.remote() for e in engines])
    for e in engines:
        ray.kill(e)
