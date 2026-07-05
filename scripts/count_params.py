"""Step 0: exact DINOv2 parameter inventory (CPU only).

Prints total param count, per-component breakdown, and the exact scalar count
that each perturbation scope ("all", "last_n_blocks" for n=1,2) would touch.
Validates the ~86M figure and the encoder.layer.{i} name filter used for
last-N-block perturbation in vision/engine.py.
"""
import argparse
from collections import OrderedDict

from transformers import Dinov2Model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="facebook/dinov2-base")
    args = ap.parse_args()

    model = Dinov2Model.from_pretrained(args.model_name)
    num_layers = model.config.num_hidden_layers

    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_name}")
    print(f"num_hidden_layers = {num_layers} | hidden_size = {model.config.hidden_size}")
    print(f"TOTAL params: {total:,} ({total/1e6:.2f}M)\n")

    # Component breakdown
    groups = OrderedDict()
    for name, p in model.named_parameters():
        parts = name.split(".")
        if parts[0] == "encoder" and parts[1] == "layer":
            key = f"encoder.layer.{parts[2]}"
        else:
            key = parts[0]
        groups[key] = groups.get(key, 0) + p.numel()

    print("Per-component parameter counts:")
    for k, v in groups.items():
        print(f"  {k:24s} {v:>12,}  ({v/1e6:.3f}M)")

    # Scope counts
    def scope_count(n):
        keep = set(range(num_layers - n, num_layers))
        c = 0
        for name, p in model.named_parameters():
            parts = name.split(".")
            if len(parts) >= 3 and parts[0] == "encoder" and parts[1] == "layer" \
                    and int(parts[2]) in keep:
                c += p.numel()
        return c

    print("\nPerturbation scope scalar counts:")
    print(f"  all              : {total:>12,}  ({total/1e6:.2f}M)")
    for n in (1, 2):
        c = scope_count(n)
        print(f"  last_{n}_blocks    : {c:>12,}  ({c/1e6:.2f}M)  "
              f"= {100*c/total:.1f}% of all  (~{total/max(c,1):.1f}x smaller search space)")


if __name__ == "__main__":
    main()
