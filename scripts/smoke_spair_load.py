"""Smoke test: load 10 SPair-71k train pairs and verify shapes."""
import sys
sys.path.insert(0, "/storage/noy/RandOpt")

from data_handlers.spair71k import SPair71kHandler

handler = SPair71kHandler()
items = handler.load_data("data/spair71k", split="train", max_samples=10)
print(f"Loaded {len(items)} pairs")
for i, d in enumerate(items[:3]):
    print(f"  [{i}] cat={d['category']} "
          f"src={d['image_tensor'].shape} tgt={d['image_tensor_tgt'].shape} "
          f"kpts_src={len(d['kpts_src'])} bbox_thresh={d['bbox_thresh']:.3f}")
print("OK")
