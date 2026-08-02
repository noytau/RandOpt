"""Dataset handlers registry."""
from .base import DatasetHandler
from .countdown import CountdownHandler
from .gqa import GQAHandler
from .gsm8k import GSM8KHandler
from .imagenet import ImageNetHandler
from .imagenet_a import ImageNetAHandler
from .imagenet_c import ImageNetCHandler
from .imagenet_es import ImageNetESHandler
from .imagenet_r import ImageNetRHandler
from .imagenet_sketch import ImageNetSketchHandler
from .math500 import MATH500Handler
from .mbpp import MBPPHandler
from .olympiadbench import OlympiadBenchHandler
from .rocstories import ROCStoriesHandler
from .uspto50k import USPTO50KHandler

# -----------------------------------------------------------------------------
# Registry
# -----------------------------------------------------------------------------

DATASET_HANDLERS = {
    "gsm8k": GSM8KHandler,
    "math500": MATH500Handler,
    "countdown": CountdownHandler,
    "olympiadbench": OlympiadBenchHandler,
    "mbpp": MBPPHandler,
    "uspto50k": USPTO50KHandler,
    "rocstories": ROCStoriesHandler,
    "gqa": GQAHandler,
    "imagenet": ImageNetHandler,
    "imagenet_c": ImageNetCHandler,
    "imagenet_a": ImageNetAHandler,
    "imagenet_r": ImageNetRHandler,
    "imagenet_sketch": ImageNetSketchHandler,
    "imagenet_es": ImageNetESHandler,
}

# -----------------------------------------------------------------------------
# Factory functions
# -----------------------------------------------------------------------------


def get_dataset_handler(name: str) -> DatasetHandler:
    """Get dataset handler by name."""
    if name not in DATASET_HANDLERS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_HANDLERS.keys())}")
    return DATASET_HANDLERS[name]()


def list_datasets() -> list:
    """List available dataset names."""
    return list(DATASET_HANDLERS.keys())

