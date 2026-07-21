"""ONE entry point for Meta's OFFICIAL dinov2 evaluations of the released
ViT-g/14+reg4 backbone on our datasets — merges run_official_knn.py and
run_official_linear.py (superseded).

Modes (--eval, default all):
  knn      official kNN — dinov2/eval/knn.py::eval_knn_with_model.
           Gallery = --train_dataset (labeled features ARE the classifier,
           so a train split is required); queries = each --val_dataset.
  linear   the RELEASED Meta-trained head(s) through the official linear-eval
           machinery (dinov2/eval/linear.py's LinearClassifier /
           test_on_datasets). No training happens; --train_dataset is unused —
           the heads were trained by Meta on full ImageNet-1k train.
  all      both.

Efficiency: the ViT-g backbone is built ONCE per invocation; every
--val_dataset (e.g. clean val, then the ImageNet-C view) is evaluated
back-to-back; and in linear mode BOTH released heads ride one feature pass
(the feature model emits the last-4-block tokens; each head slices what it
needs — exactly how official linear.py evaluates its classifier grid).

Geoffry runtime patches (in-memory only, the pinned dinov2 clone is never
modified; NONE of these are needed on RunAI):
  1. ImageNet.__len__ without the hardcoded full-split length assert, so the
     50/class train subset is accepted as a gallery.
  2. dinov2.distributed.enable -> 1-process GLOO group: NCCL's NVML topology
     probe is fatal on Geoffry's dead GPU 3 (nvmlDeviceGetHandleByIndex(3)).
  3. torch.inference_mode -> torch.no_grad: GLOO collectives / torchmetrics
     sync cannot write into inference tensors.

Checkpoints (auto-downloaded to the torch.hub cache):
  released heads dinov2_vitg14_reg4_linear_head.pth  (3072 -> 1000)
                 dinov2_vitg14_reg4_linear4_head.pth (7680 -> 1000)
Backbone .pth is passed explicitly (--pretrained_weights).

Example (Geoffry):
  source ~/randopt_env.sh && export CUDA_VISIBLE_DEVICES=1
  python -u scripts/run_official_eval.py --eval all \
    --pretrained_weights ~/.cache/torch/hub/checkpoints/dinov2_vitg14_reg4_pretrain.pth \
    --train_dataset "ImageNet:split=TRAIN:root=$DATASETS_ROOT/imagenet:extra=$DATASETS_ROOT/imagenet/extra" \
    --val_dataset "ImageNet:split=VAL:root=$DATASETS_ROOT/imagenet:extra=$DATASETS_ROOT/imagenet/extra" \
                  "ImageNet:split=VAL:root=$DATASETS_ROOT/imagenet_c_view/gaussian_noise_s3:extra=$DATASETS_ROOT/imagenet_c_view/gaussian_noise_s3/extra" \
    --output_dir results/official-eval
"""
import argparse
import os
import re
import socket
import sys
from functools import partial

HEAD_URL = ("https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/"
            "dinov2_vitg14_reg4_linear{suffix}_head.pth")
# head name -> (use_n_blocks, head input dim, url suffix)
HEAD_CFG = {"linear": (1, 2 * 1536, ""), "linear4": (4, 5 * 1536, "4")}


# ---------------------------------------------------------------------------
# Geoffry runtime patches (see module docstring)
# ---------------------------------------------------------------------------

def _patch_imagenet_len():
    from dinov2.data.datasets import image_net
    image_net.ImageNet.__len__ = lambda self: len(self._get_entries())


def _patch_distributed_gloo():
    import torch.distributed as dist
    import dinov2.distributed as D

    def _enable(**kwargs):
        if dist.is_available() and dist.is_initialized():
            return
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        with socket.socket() as s:                      # OS-assigned free port
            s.bind(("", 0))
            os.environ["MASTER_PORT"] = str(s.getsockname()[1])
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
        D._LOCAL_RANK, D._LOCAL_WORLD_SIZE = 0, 1

    D.enable = _enable


def _dataset_tag(dataset_str):
    """Short per-dataset output tag from the root's basename
    (…root=/x/imagenet:… -> 'imagenet'; …gaussian_noise_s3 -> that)."""
    m = re.search(r"root=([^:]+)", dataset_str)
    return os.path.basename(m.group(1).rstrip("/")) if m else "dataset"


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_knn(model, autocast_dtype, args):
    """Official kNN per val dataset (gallery re-extracted per call — that's
    eval_knn_with_model's own flow)."""
    from dinov2.eval.knn import eval_knn_with_model
    from dinov2.eval.metrics import AccuracyAveraging

    out = {}
    for ds in args.val_dataset:
        tag = _dataset_tag(ds)
        outdir = os.path.join(args.output_dir, f"knn-{tag}")
        os.makedirs(outdir, exist_ok=True)
        print(f"\n=== kNN: gallery={_dataset_tag(args.train_dataset)} "
              f"queries={tag} ===")
        results = eval_knn_with_model(
            model=model,
            output_dir=outdir,
            train_dataset_str=args.train_dataset,
            val_dataset_str=ds,
            nb_knn=tuple(int(k) for k in args.nb_knn.split(",")),
            temperature=args.temperature,
            autocast_dtype=autocast_dtype,
            accuracy_averaging=AccuracyAveraging.MEAN_ACCURACY,
            transform=None,               # -> official eval transform
            gather_on_cpu=True,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_per_class_list=[-1],
            n_tries=1,
        )
        out[tag] = {k: (float(v) if hasattr(v, "item") else v)
                    for k, v in results.items()}
    return out


def run_linear(model, autocast_dtype, args):
    """Released head(s) through the official linear-eval machinery: ONE
    feature pass per val dataset serves every requested head."""
    import torch
    from dinov2.eval.linear import (AllClassifiers, LinearClassifier,
                                    test_on_datasets)
    from dinov2.eval.metrics import MetricType
    from dinov2.eval.utils import ModelWithIntermediateLayers

    heads = args.heads.split(",")
    n_last_blocks = max(HEAD_CFG[h][0] for h in heads)
    autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True,
                           dtype=autocast_dtype)
    feature_model = ModelWithIntermediateLayers(model, n_last_blocks,
                                                autocast_ctx)
    classifiers = {}
    for h in heads:
        use_n_blocks, out_dim, suffix = HEAD_CFG[h]
        lc = LinearClassifier(out_dim, use_n_blocks=use_n_blocks,
                              use_avgpool=True, num_classes=1000)
        state = torch.hub.load_state_dict_from_url(
            HEAD_URL.format(suffix=suffix), map_location="cpu")
        lc.linear.load_state_dict(state, strict=True)
        classifiers[f"released_{h}_head"] = lc.cuda().eval()
        print(f"released {h} head loaded: {tuple(lc.linear.weight.shape)}")

    outdir = os.path.join(args.output_dir, "linear-heads")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n=== released heads ({args.heads}) on: "
          f"{[_dataset_tag(d) for d in args.val_dataset]} ===")
    return test_on_datasets(
        feature_model=feature_model,
        linear_classifiers=AllClassifiers(classifiers),
        test_dataset_strs=list(args.val_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        test_metric_types=[MetricType.MEAN_ACCURACY] * len(args.val_dataset),
        metrics_file_path=os.path.join(outdir, "results_eval_linear.json"),
        training_num_classes=1000,
        iteration=0,
        best_classifier_on_val=None,
        test_class_mappings=[None] * len(args.val_dataset),
    )


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval", choices=["knn", "linear", "all"], default="all")
    p.add_argument("--dinov2_dir",
                   default=os.environ.get("DINOV2_DIR", "/mnt5/noy/dinov2"))
    p.add_argument("--config_file", default=None,
                   help="default: <dinov2>/dinov2/configs/eval/vitg14_reg4_pretrain.yaml")
    p.add_argument("--pretrained_weights", required=True,
                   help="released dinov2_vitg14_reg4_pretrain.pth (backbone)")
    p.add_argument("--train_dataset", default=None,
                   help="kNN gallery dataset string (REQUIRED for knn; unused "
                        "by linear — the released heads are already trained)")
    p.add_argument("--val_dataset", nargs="+", required=True,
                   help="one or more query/eval dataset strings, run in order")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--batch_size", type=int, default=16,
                   help="16 fits ViT-g on an 11GB 2080 Ti")
    p.add_argument("--num_workers", type=int, default=2)
    # knn options (official defaults)
    p.add_argument("--nb_knn", default="10,20,100,200")
    p.add_argument("--temperature", type=float, default=0.07)
    # linear options
    p.add_argument("--heads", default="linear,linear4",
                   help="comma list of released heads to evaluate in one pass")
    return p.parse_args()


def main(args):
    if args.eval in ("knn", "all") and not args.train_dataset:
        sys.exit("--train_dataset is required for --eval knn/all: the labeled "
                 "gallery IS the kNN classifier (use --eval linear for the "
                 "gallery-free released-head evaluation)")
    for h in args.heads.split(","):
        if h not in HEAD_CFG:
            sys.exit(f"unknown head '{h}' (choices: {sorted(HEAD_CFG)})")

    sys.path.insert(0, args.dinov2_dir)  # import dinov2 from the pinned clone
    import torch
    torch.inference_mode = torch.no_grad  # must precede dinov2 eval imports
    _patch_imagenet_len()
    _patch_distributed_gloo()

    from dinov2.eval.setup import get_args_parser as get_setup_args_parser
    from dinov2.eval.setup import setup_and_build_model

    config = args.config_file or os.path.join(
        args.dinov2_dir, "dinov2/configs/eval/vitg14_reg4_pretrain.yaml")
    os.makedirs(args.output_dir, exist_ok=True)
    setup_args = get_setup_args_parser().parse_args([
        "--config-file", config,
        "--pretrained-weights", args.pretrained_weights,
        "--output-dir", args.output_dir,
    ])
    model, autocast_dtype = setup_and_build_model(setup_args)  # built ONCE

    results = {}
    if args.eval in ("knn", "all"):
        results["knn"] = run_knn(model, autocast_dtype, args)
    if args.eval in ("linear", "all"):
        results["linear"] = run_linear(model, autocast_dtype, args)
    print("\nFINAL RESULTS:", results)
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))
