"""Build manifests for 5 new non-ImageNet datasets, each with a genuine
train/val/test split of its own (unlike imagenet_a/r/sketch/es, which are
test-only draws scored against clean ImageNet-train — see make_shift_manifests.py).

Each dataset has its OWN label space (not ImageNet-1k), so there is no shared
canonical class index to resolve against. Instead every dataset gets its own
explicit, alphabetically-sorted class list saved to
_meta/<dataset>_class_index.json — written once and reused, never silently
re-derived from a directory listing at consumption time (that silent-resort
trap is what make_shift_manifests.py's docstring warns about for imagenet_r).

Splits used:
    exdark            - official (imageclasslist.txt col 5: train/val/test),
                         PLUS a "train_holdout" split carved out of official
                         train (stratified, seeded) for RandOpt scoring —
                         kept disjoint from the (smaller) remaining "train"
                         split that scripts/fit_head.py fits the linear head
                         on, so scoring never re-measures images the head has
                         already seen.
    fgvc_aircraft      - official (data/images_variant_{train,val,test}.txt)
    stanford_dogs      - official train/test (lists/{train,test}_list.mat);
                         val carved from train (stratified, seeded)
    domainnet_clipart  - official train/test ({domain}_{train,test}.txt);
                         val carved from train (stratified, seeded)
    domainnet_sketch   - same as domainnet_clipart

Manifest schema (JSON array):
    {"image": "/abs/path/...", "label": 3, "class_name": "Cat",
     "split": "train", "dataset": "exdark", "sha256": "..."}

Usage:
    python scripts/data_prep/make_new_dataset_manifests.py --dataset exdark
    python scripts/data_prep/make_new_dataset_manifests.py --dataset all --seed 0
"""
import argparse
import hashlib
import json
import os
import random
import sys

IMG_EXTS = (".jpg", ".jpeg", ".png")
ALL_DATASETS = ["exdark", "fgvc_aircraft", "stanford_dogs", "domainnet_clipart", "domainnet_sketch"]


def _root():
    return os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")


def _paths():
    root = _root()
    return {
        "raw": os.path.join(root, "raw"),
        "manifests": os.path.join(root, "manifests"),
        "meta": os.path.join(root, "manifests", "_meta"),
    }


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def save_class_index(meta_dir, dataset, class_names):
    os.makedirs(meta_dir, exist_ok=True)
    path = os.path.join(meta_dir, f"{dataset}_class_index.json")
    with open(path, "w") as f:
        json.dump({"dataset": dataset, "classes": class_names,
                    "name_to_idx": {n: i for i, n in enumerate(class_names)}}, f, indent=2)
    return path


def carve_split_from_train(entries, frac=0.1, seed=0, new_split="val"):
    """Stratified per-class carve of `frac` of the split=="train" subset of
    `entries` into a new split (default "val"), mutating those entries in
    place. Only entries already labeled "train" are grouped/shuffled/carved —
    entries with any other split (e.g. "test") are never inspected, so
    already-held-out splits can never be contaminated. Also used to carve a
    RandOpt-scoring holdout (new_split="train_holdout") out of a dataset's
    train split, disjoint from whatever the rest of "train" is used for
    (e.g. head-fitting) — see build_exdark."""
    by_class = {}
    for e in entries:
        if e["split"] == "train":
            by_class.setdefault(e["label"], []).append(e)
    rng = random.Random(seed)
    for label, items in by_class.items():
        items = items[:]
        rng.shuffle(items)
        n_carve = max(1, int(round(len(items) * frac)))
        for e in items[:n_carve]:
            e["split"] = new_split
    return entries


def _write_manifest(manifests_dir, dataset, entries, hash_images):
    if hash_images:
        for e in entries:
            e["sha256"] = sha256_file(e["image"])
    os.makedirs(manifests_dir, exist_ok=True)
    path = os.path.join(manifests_dir, f"{dataset}.json")
    with open(path, "w") as f:
        json.dump(entries, f)
    counts = {}
    for e in entries:
        counts[e["split"]] = counts.get(e["split"], 0) + 1
    print(f"{dataset}: wrote {len(entries)} entries -> {path} ({counts})")


# -----------------------------------------------------------------------------
# exdark
# -----------------------------------------------------------------------------

_EXDARK_CLASSES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair",
                   "Cup", "Dog", "Motorbike", "People", "Table"]
_EXDARK_SPLIT = {"1": "train", "2": "val", "3": "test"}


def build_exdark(paths, hash_images, randopt_holdout_frac, seed):
    root = os.path.join(paths["raw"], "exdark")
    list_path = os.path.join(root, "imageclasslist.txt")
    if not os.path.exists(list_path):
        raise FileNotFoundError(f"{list_path} missing — run download_new_datasets.sh first")
    entries = []
    with open(list_path) as f:
        next(f)  # header
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            name, class_id, _light, _inout, split_id = parts[:5]
            label = int(class_id) - 1
            class_name = _EXDARK_CLASSES[label]
            img_path = os.path.join(root, class_name, name)
            if not os.path.exists(img_path):
                # ExDark's own README notes class folder names match exactly;
                # skip silently missing files rather than fail the whole build.
                continue
            entries.append({"image": img_path, "label": label,
                             "class_name": class_name,
                             "split": _EXDARK_SPLIT[split_id], "dataset": "exdark"})
    # RandOpt scoring needs data disjoint from whatever the linear head is
    # fit on (scripts/fit_head.py loads split="train"), or "scoring on train"
    # just re-measures how well the head already memorized those exact
    # images. Carve a class-balanced holdout out of the official train split
    # for that purpose; official val/test are untouched (carve_split_from_train
    # only ever looks at split=="train" entries).
    entries = carve_split_from_train(entries, frac=randopt_holdout_frac, seed=seed,
                                     new_split="train_holdout")
    save_class_index(paths["meta"], "exdark", _EXDARK_CLASSES)
    _write_manifest(paths["manifests"], "exdark", entries, hash_images)


# -----------------------------------------------------------------------------
# fgvc_aircraft
# -----------------------------------------------------------------------------

def build_fgvc_aircraft(paths, hash_images):
    root = os.path.join(paths["raw"], "fgvc_aircraft", "data")
    images_dir = os.path.join(root, "images")
    variants_path = os.path.join(root, "variants.txt")
    with open(variants_path) as f:
        class_names = sorted(line.strip() for line in f if line.strip())
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    entries = []
    for split_file, split in [("images_variant_train.txt", "train"),
                               ("images_variant_val.txt", "val"),
                               ("images_variant_test.txt", "test")]:
        with open(os.path.join(root, split_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_id, variant = line.split(" ", 1)
                img_path = os.path.join(images_dir, f"{image_id}.jpg")
                entries.append({"image": img_path, "label": name_to_idx[variant],
                                 "class_name": variant, "split": split,
                                 "dataset": "fgvc_aircraft"})
    save_class_index(paths["meta"], "fgvc_aircraft", class_names)
    _write_manifest(paths["manifests"], "fgvc_aircraft", entries, hash_images)


# -----------------------------------------------------------------------------
# stanford_dogs
# -----------------------------------------------------------------------------

def _load_dogs_mat_list(mat_path):
    from scipy.io import loadmat
    mat = loadmat(mat_path, squeeze_me=True)
    file_list = mat["file_list"]
    labels = mat["labels"]
    out = []
    for rel_path, label in zip(file_list, labels):
        rel_path = str(rel_path)
        out.append((rel_path, int(label) - 1))  # 1-indexed -> 0-indexed
    return out


def build_stanford_dogs(paths, hash_images, val_frac, seed):
    root = os.path.join(paths["raw"], "stanford_dogs")
    images_dir = os.path.join(root, "Images")
    class_dirs = sorted(d for d in os.listdir(images_dir)
                         if os.path.isdir(os.path.join(images_dir, d)))
    class_names = [d.split("-", 1)[1] for d in class_dirs]

    entries = []
    for mat_file, split in [("train_list.mat", "train"), ("test_list.mat", "test")]:
        for rel_path, label in _load_dogs_mat_list(os.path.join(root, mat_file)):
            img_path = os.path.join(images_dir, rel_path)
            entries.append({"image": img_path, "label": label,
                             "class_name": class_names[label], "split": split,
                             "dataset": "stanford_dogs"})
    entries = carve_split_from_train(entries, frac=val_frac, seed=seed, new_split="val")
    save_class_index(paths["meta"], "stanford_dogs", class_names)
    _write_manifest(paths["manifests"], "stanford_dogs", entries, hash_images)


# -----------------------------------------------------------------------------
# domainnet (clipart / sketch)
# -----------------------------------------------------------------------------

def build_domainnet(paths, domain, hash_images, val_frac, seed):
    dataset = f"domainnet_{domain}"
    root = os.path.join(paths["raw"], dataset)
    class_names = sorted(d for d in os.listdir(root)
                          if os.path.isdir(os.path.join(root, d)))
    name_to_idx = {n: i for i, n in enumerate(class_names)}

    entries = []
    for split_file, split in [(f"{domain}_train.txt", "train"),
                               (f"{domain}_test.txt", "test")]:
        with open(os.path.join(root, split_file)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rel_path, label_str = line.rsplit(" ", 1)
                # official lists prefix every path with the domain name itself
                # (e.g. "clipart/aircraft_carrier/xxx.jpg"), but our extracted
                # tree is already rooted at the domain dir — strip it.
                domain_prefix, sub_path = rel_path.split("/", 1)
                assert domain_prefix == domain, (domain_prefix, domain)
                class_name = sub_path.split("/")[0]
                img_path = os.path.join(root, sub_path)
                entries.append({"image": img_path, "label": name_to_idx[class_name],
                                 "class_name": class_name, "split": split,
                                 "dataset": dataset})
    entries = carve_split_from_train(entries, frac=val_frac, seed=seed, new_split="val")
    save_class_index(paths["meta"], dataset, class_names)
    _write_manifest(paths["manifests"], dataset, entries, hash_images)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all",
                     choices=ALL_DATASETS + ["all"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.1,
                     help="fraction of official train carved into val, for "
                          "datasets with no official val split")
    ap.add_argument("--randopt_holdout_frac", type=float, default=0.2,
                     help="fraction of exdark's official train carved into a "
                          "'train_holdout' split for RandOpt scoring, kept "
                          "disjoint from whatever fit_head.py trains the "
                          "head on (the rest of 'train')")
    ap.add_argument("--no_hash", action="store_true",
                     help="skip sha256 checksums (faster, no integrity record)")
    args = ap.parse_args()

    paths = _paths()
    hash_images = not args.no_hash
    targets = ALL_DATASETS if args.dataset == "all" else [args.dataset]

    for name in targets:
        if name == "exdark":
            build_exdark(paths, hash_images, args.randopt_holdout_frac, args.seed)
        elif name == "fgvc_aircraft":
            build_fgvc_aircraft(paths, hash_images)
        elif name == "stanford_dogs":
            build_stanford_dogs(paths, hash_images, args.val_frac, args.seed)
        elif name == "domainnet_clipart":
            build_domainnet(paths, "clipart", hash_images, args.val_frac, args.seed)
        elif name == "domainnet_sketch":
            build_domainnet(paths, "sketch", hash_images, args.val_frac, args.seed)


if __name__ == "__main__":
    sys.exit(main())
