"""Export HF ILSVRC/imagenet-1k TRAIN into the official dinov2 ImageNet layout,
optionally capped to N images/class so it fits a small disk (Geoffry POC).

Downloads ONLY train parquet shards (hf_hub_download, never load_dataset — that
grabs every split, ~155GB, and fills the disk: CLAUDE.md trap). Each shard is
processed then DELETED, so peak disk ≈ one shard + the kept subset, not the full
147GB. Writes raw JPEG bytes (no re-encode) to:

    <root>/train/<wnid>/<wnid>_<idx>.JPEG

which is exactly what dinov2's ImageNet class expects for TRAIN
(get_image_relpath basename = f"{class_id}_{actual_index}"; parse_image_relpath
reads actual_index = int(basename.split("_")[-1])).

--per-class N  : stop collecting a class once it has N (Stage A, N=50 -> ~6GB).
--per-class 0  : keep everything (Stage B full train, ~146GB on the RunAI PVC).

HF stores train filenames as "<wnid>_<idx>.JPEG"; we keep the name verbatim so
the wnid (label) and idx round-trip through dinov2's parser. Needs a HF token
(license accepted) in ~/.cache/huggingface/token or $HF_TOKEN.
"""
import argparse
import os

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download

REPO = "ILSVRC/imagenet-1k"
NUM_CLASSES = 1000


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=os.path.join(
        os.environ.get("DATASETS_ROOT", "data"), "imagenet"),
        help="dataset root; images go to <root>/train/<wnid>/")
    p.add_argument("--per_class", type=int, default=50,
                   help="max images/class (0 = keep all = full train)")
    p.add_argument("--max_shards", type=int, default=0,
                   help="stop after this many shards (0 = no limit)")
    return p.parse_args()


def list_train_shards():
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files
                  if f.startswith("data/train-") and f.endswith(".parquet"))


def main(args):
    train_dir = os.path.join(args.root, "train")
    os.makedirs(train_dir, exist_ok=True)
    cap = args.per_class or None
    shards = list_train_shards()
    print(f"{len(shards)} train shards; cap={cap or 'ALL'} per class -> {train_dir}")

    counts = {}          # wnid -> images written
    total = 0
    for si, shard in enumerate(shards):
        if args.max_shards and si >= args.max_shards:
            print(f"stop: reached --max_shards {args.max_shards}")
            break
        # every class already full? (only reachable when capped)
        if cap and len(counts) == NUM_CLASSES and all(
                c >= cap for c in counts.values()):
            print("stop: every class reached the cap")
            break

        local = hf_hub_download(REPO, shard, repo_type="dataset")
        table = pq.read_table(local, columns=["image"]).column("image").to_pylist()
        for img in table:
            path, data = img["path"], img["bytes"]
            assert path and data, f"shard {shard} missing filename/bytes"
            # HF appends "_<wnid>" to every name (e.g. n01440764_10183_n01440764
            # .JPEG); strip it to the "<wnid>_<idx>.JPEG" dinov2 parses.
            parts = os.path.basename(path).rsplit(".", 1)[0].split("_")
            wnid = parts[-1]
            fname = "_".join(parts[:-1]) + ".JPEG"
            if cap and counts.get(wnid, 0) >= cap:
                continue
            d = os.path.join(train_dir, wnid)
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, fname), "wb") as f:
                f.write(data)
            counts[wnid] = counts.get(wnid, 0) + 1
            total += 1
        os.remove(local)                                # bound peak disk
        full = sum(1 for c in counts.values() if not cap or c >= cap)
        print(f"shard {si+1}/{len(shards)}: {total} imgs, "
              f"{len(counts)}/{NUM_CLASSES} classes seen, "
              f"{full} at target", flush=True)

    short = [w for w, c in counts.items() if cap and c < cap]
    print(f"DONE: {total} images, {len(counts)} classes.")
    if len(counts) < NUM_CLASSES:
        print(f"WARNING: {NUM_CLASSES - len(counts)} classes missing entirely.")
    if short:
        print(f"WARNING: {len(short)} classes below cap (e.g. {short[:5]}).")


if __name__ == "__main__":
    main(parse_args())
