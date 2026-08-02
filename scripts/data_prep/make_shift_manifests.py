"""Build manifests for the 4 distribution-shift datasets (ImageNet-A/R/Sketch/ES).

Every label is resolved through the canonical class index
(_meta/imagenet_class_index.json, built by build_class_index.py) — never by
re-sorting the wnids present in a dataset's own directory tree, which would
silently give a 200-class subset local indices 0..199 instead of its true
ImageNet-1k indices (see build_class_index.py's docstring; confirmed to be a
real trap, not hypothetical, via the published ImageNet-R split's `targets`
field, which IS 0..199-local and is discarded here for exactly that reason).

Manifest schema (JSON array, matching data_handlers/imagenet_c.py's existing
consumer contract, plus extra provenance fields added on top):
    {"image": "/abs/path/...", "label": 17, "wnid": "n01440764",
     "split": "train", "dataset": "imagenet_r", "source": "imagenet_r",
     "condition": null, "sha256": "..."}

Usage:
    python scripts/data_prep/make_shift_manifests.py --dataset imagenet_r
    python scripts/data_prep/make_shift_manifests.py --dataset all --seed 0
    python scripts/data_prep/make_shift_manifests.py --dataset imagenet_es --inspect
"""
import argparse
import ast
import hashlib
import json
import os
import random
import re
import sys
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime

IMG_EXTS = (".jpg", ".jpeg", ".png")

_IMAGENET_R_EVAL_URL = "https://raw.githubusercontent.com/hendrycks/imagenet-r/master/eval.py"
_CODA_TRAIN_URL = ("https://raw.githubusercontent.com/GT-RIPL/CODA-Prompt/"
                    "main/dataloaders/splits/imagenet-r_train.yaml")
_CODA_TEST_URL = ("https://raw.githubusercontent.com/GT-RIPL/CODA-Prompt/"
                   "main/dataloaders/splits/imagenet-r_test.yaml")
_NAE_EVAL_URL = ("https://raw.githubusercontent.com/hendrycks/"
                  "natural-adv-examples/master/eval.py")


# -----------------------------------------------------------------------------
# Small shared helpers
# -----------------------------------------------------------------------------

def _root():
    return os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")


def _paths():
    root = _root()
    return {
        "raw": os.path.join(root, "raw"),
        "manifests": os.path.join(root, "manifests"),
        "meta": os.path.join(root, "manifests", "_meta"),
        "logs": os.path.join(root, "logs"),
    }


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def list_class_dirs(root: str):
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def list_images(class_dir: str):
    return sorted(f for f in os.listdir(class_dir)
                  if f.lower().endswith(IMG_EXTS))


def load_class_index(meta_dir: str) -> dict:
    path = os.path.join(meta_dir, "imagenet_class_index.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} missing — run build_class_index.py first")
    with open(path) as f:
        return json.load(f)["wnid_to_idx"]


def _fetch_and_cache_text(cache_path: str, url: str) -> str:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return f.read()
    with urllib.request.urlopen(url, timeout=30) as resp:
        text = resp.read().decode()
    with open(cache_path, "w") as f:
        f.write(text)
    return text


def _find_wnid_root(base_dir: str, min_classes: int = 100, max_depth: int = 4):
    """BFS for the directory whose immediate children are mostly `n\\d{8}`
    dirs — robust to whatever top-level nesting an upstream zip/tar used,
    rather than assuming a fixed layout (spec's "verify, don't assume")."""
    wnid_re = re.compile(r"^n\d{8}$")
    frontier = [(base_dir, 0)]
    while frontier:
        d, depth = frontier.pop(0)
        try:
            children = [c for c in os.listdir(d) if os.path.isdir(os.path.join(d, c))]
        except (FileNotFoundError, PermissionError):
            continue
        if children and sum(1 for c in children if wnid_re.match(c)) >= min_classes:
            return d
        if depth < max_depth:
            frontier.extend((os.path.join(d, c), depth + 1) for c in children)
    raise RuntimeError(f"no wnid-named directory tree found under {base_dir}")


def _split_yaml(text: str):
    """Minimal parser for CODA-Prompt's `data:` / `targets:` split YAML —
    a flat two-list format, so a full YAML dependency isn't needed."""
    lines = text.splitlines()
    idx_data = lines.index("data:")
    idx_targets = lines.index("targets:")
    data = [l[2:].strip() for l in lines[idx_data + 1:idx_targets] if l.startswith("- ")]
    targets = [int(l[2:].strip()) for l in lines[idx_targets + 1:] if l.startswith("- ")]
    return data, targets


def _reconcile_duplicate_content_across_splits(entries, dataset_name):
    """Final safety net, applied to every dataset regardless of how its split
    was assigned: group ALL entries (any class) by content hash and force any
    hash-group spanning more than one split into a single split.

    Per-class hash-grouping (_split_by_target_counts_no_dup_leak) only catches
    duplicates WITHIN one class directory; it can't see duplicates ACROSS
    classes, and imagenet_r's split comes from a published file-path list
    with no hash-awareness at all. Found for real: imagenet_r's L2P/DualPrompt
    split leaked 28 duplicate-content images across train/val/test, and
    imagenet_sketch leaked 319 CROSS-CLASS duplicates per-class grouping
    couldn't catch. Reassignment target = the split holding the plurality of
    that hash group (tie broken by min image path, for determinism).
    """
    by_hash = defaultdict(list)
    for e in entries:
        by_hash[e["sha256"]].append(e)
    n_groups_fixed = n_entries_moved = 0
    for group in by_hash.values():
        splits_in_group = {e["split"] for e in group}
        if len(splits_in_group) <= 1:
            continue
        counts = Counter(e["split"] for e in group)
        max_count = max(counts.values())
        candidates = sorted(s for s, c in counts.items() if c == max_count)
        target_split = candidates[0]
        n_groups_fixed += 1
        for e in group:
            if e["split"] != target_split:
                n_entries_moved += 1
                e["split"] = target_split
    if n_groups_fixed:
        print(f"{dataset_name}: reconciled {n_groups_fixed} duplicate-content "
              f"group(s) spanning splits, moved {n_entries_moved} entries to "
              f"keep every duplicate-content image in one split")
    return entries


def _write_entries(entries, out_path, dataset_name):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    entries = sorted(entries, key=lambda e: e["image"])
    for e in entries:
        # builders that already dedup-grouped by content hash
        # (_split_by_target_counts_no_dup_leak) pass it through instead of
        # paying for a second full read of every file
        if e.get("sha256") is None:
            e["sha256"] = sha256_file(e["image"])
    entries = _reconcile_duplicate_content_across_splits(entries, dataset_name)
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=1)
    return entries


def _write_meta(meta_path, entries, subset_indices, wnids, seed, source_info):
    per_split = {}
    for e in entries:
        per_split[e["split"]] = per_split.get(e["split"], 0) + 1
    per_class = {}
    for e in entries:
        per_class[e["wnid"]] = per_class.get(e["wnid"], 0) + 1
    meta = {
        "total": len(entries),
        "per_split": per_split,
        "per_class": per_class,
        "subset_indices": subset_indices,
        "effective_class_count": len(wnids),
        "seed": seed,
        "source": source_info,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=1)


# -----------------------------------------------------------------------------
# ImageNet-R
# -----------------------------------------------------------------------------

def get_imagenet_r_wnids(meta_dir: str):
    cache = os.path.join(meta_dir, "imagenet_r_wnids.txt")
    if os.path.exists(cache):
        return sorted(l.strip() for l in open(cache) if l.strip())
    text = _fetch_and_cache_text(os.path.join(meta_dir, "_imagenet_r_eval.py.cache"),
                                  _IMAGENET_R_EVAL_URL)
    m = re.search(r"imagenet_r_wnids\s*=\s*\{([^}]*)\}", text)
    if not m:
        raise RuntimeError("could not find imagenet_r_wnids in upstream eval.py")
    wnids = sorted(set(re.findall(r"n\d{8}", m.group(1))))
    with open(cache, "w") as f:
        f.write("\n".join(wnids) + "\n")
    return wnids


def build_imagenet_r(paths, wnid_to_idx, seed):
    raw_dir = os.path.join(paths["raw"], "imagenet_r")
    meta_dir = paths["meta"]
    ref_wnids = get_imagenet_r_wnids(meta_dir)
    dir_wnids = list_class_dirs(raw_dir)
    if set(dir_wnids) != set(ref_wnids):
        raise RuntimeError(
            "imagenet_r wnid mismatch between extracted dirs and "
            f"hendrycks/imagenet-r's imagenet_r_wnids: "
            f"dirs-only={sorted(set(dir_wnids) - set(ref_wnids))}, "
            f"ref-only={sorted(set(ref_wnids) - set(dir_wnids))}")

    train_yaml = _fetch_and_cache_text(
        os.path.join(meta_dir, "_imagenet_r_train_split.yaml.cache"), _CODA_TRAIN_URL)
    test_yaml = _fetch_and_cache_text(
        os.path.join(meta_dir, "_imagenet_r_test_split.yaml.cache"), _CODA_TEST_URL)
    train_full_raw, _discarded_local_targets = _split_yaml(train_yaml)
    test_raw, _ = _split_yaml(test_yaml)
    prefix = "data/imagenet-r/"
    train_full = sorted(p[len(prefix):] for p in train_full_raw)
    test = sorted(p[len(prefix):] for p in test_raw)

    on_disk = {f"{w}/{f}" for w in dir_wnids
               for f in list_images(os.path.join(raw_dir, w))}
    overlap = set(train_full) & set(test)
    if overlap:
        raise RuntimeError(f"imagenet_r train/test split overlap: {len(overlap)} files")
    missing = (set(train_full) | set(test)) - on_disk
    if missing:
        raise RuntimeError(
            f"{len(missing)} published-split files not found on disk, "
            f"e.g. {sorted(missing)[:5]}")
    extra = on_disk - (set(train_full) | set(test))
    if extra:
        print(f"WARNING imagenet_r: {len(extra)} on-disk files not covered by the "
              f"published split; excluded, e.g. {sorted(extra)[:5]}")

    rng = random.Random(seed)
    train_full_shuffled = sorted(train_full)  # sort before shuffle: determinism (spec S6)
    rng.shuffle(train_full_shuffled)
    n_val = len(train_full_shuffled) // 6  # 5:1 train:val -> val = 1/6 of train_full
    val = sorted(train_full_shuffled[:n_val])
    train = sorted(train_full_shuffled[n_val:])

    entries = []
    for rel, split in ([(p, "train") for p in train] +
                        [(p, "val") for p in val] +
                        [(p, "test") for p in sorted(test)]):
        wnid = rel.split("/")[0]
        entries.append({
            "image": os.path.join(raw_dir, rel),
            "wnid": wnid,
            "label": wnid_to_idx[wnid],
            "split": split,
            "dataset": "imagenet_r",
            "source": "imagenet_r",
            "condition": None,
        })
    subset_indices = sorted(wnid_to_idx[w] for w in ref_wnids)
    with open(os.path.join(meta_dir, "imagenet_r_wnids.txt"), "w") as f:
        f.write("\n".join(ref_wnids) + "\n")
    return entries, subset_indices, ref_wnids


def _split_by_target_counts_no_dup_leak(class_dir, files, targets, rng):
    """Assign `files` (within one class dir) to splits per `targets` (dict
    split_name -> desired count, summing to len(files)), grouping files that
    share a content hash so exact-duplicate images (same photo, two
    filenames) never land in different splits.

    Not hypothetical: caught 2 duplicate-content ImageNet-A images that a
    filename-only stratified split put in different splits (train vs val) —
    exactly the leak spec S7#5's sha256-disjointness check exists to catch.
    Greedy: shuffle groups, assign each to whichever split is furthest below
    its target count; small ratio slop is acceptable, leakage is not.

    Phase 0 (coverage guarantee): some classes are almost entirely duplicate
    content — one ImageNet-Sketch class has 50 files but only 4 distinct
    images repeated ~12x each. With groups that large relative to the
    val/test targets, plain greedy-by-ratio can leave a split with a
    positive target completely empty (all 4 groups got out-greedied by
    train's bigger target) — a hard spec-S7#6 violation, not a rounding
    nit. So every split with target > 0 claims the smallest remaining group
    FIRST, before ratio-based greedy fills the rest — trading ratio
    precision for the one thing that can't be traded away: every split that
    should have images does.

    Returns (assignment, hashes): hashes is {fname: sha256}, computed here
    anyway to build the dedup groups — the caller threads it into the
    manifest entry so _write_entries doesn't hash the same file twice.
    """
    groups = {}
    hashes = {}
    for fname in files:
        h = sha256_file(os.path.join(class_dir, fname))
        hashes[fname] = h
        groups.setdefault(h, []).append(fname)
    group_list = list(groups.values())
    # shuffle only matters for tie-breaking: the sort below is stable, so
    # equal-length groups keep this random relative order, longer-vs-shorter
    # ordering is unaffected
    rng.shuffle(group_list)
    order = list(targets)
    counts = {name: 0 for name in order}
    assignment = {}

    unassigned = sorted(group_list, key=len)
    for name in order:
        if targets[name] > 0 and counts[name] == 0 and unassigned:
            group = unassigned.pop(0)
            for fname in group:
                assignment[fname] = name
            counts[name] += len(group)

    for group in unassigned:
        name = min(order, key=lambda s: counts[s] - targets[s])
        for fname in group:
            assignment[fname] = name
        counts[name] += len(group)
    return assignment, hashes


def _stratified_class_entries(class_dir, wnid, wnid_to_idx, rng, dataset_name,
                               test_ratio, val_ratio, min_class_size,
                               small_class_all_test):
    """Ratio-stratified train/val/test split for one class's files, shared by
    build_imagenet_a and build_imagenet_sketch (both: list files -> pick
    per-split target counts from a ratio, with a small-class fallback since
    there isn't enough data to stratify a handful of images -> assign via
    the dedup-aware splitter -> build manifest entries). Only the ratios and
    small-class behavior differ between the two datasets.

    small_class_all_test=True: classes under min_class_size go entirely to
    test (imagenet_sketch's spec-mandated rule, S3.3). False: keep n-1
    train / 1 test / 0 val instead of discarding the class from train
    entirely (imagenet_a has no published small-class rule, so this
    preserves some train signal rather than defaulting to spec's sketch-
    specific choice).

    Returns (entries, is_small, n) — is_small/n let ImageNet-Sketch log its
    dropped-small-class report; imagenet_a doesn't use them.
    """
    files = list_images(class_dir)
    n = len(files)
    is_small = n < min_class_size
    if is_small and small_class_all_test:
        targets = {"train": 0, "val": 0, "test": n}
    elif is_small:
        targets = {"train": n - 1, "val": 0, "test": 1}
    else:
        n_te = max(1, round(n * test_ratio))
        n_va = max(1, round(n * val_ratio))
        targets = {"train": n - n_te - n_va, "val": n_va, "test": n_te}
    assignment, hashes = _split_by_target_counts_no_dup_leak(class_dir, files, targets, rng)
    entries = [{
        "image": os.path.join(class_dir, fname),
        "wnid": wnid,
        "label": wnid_to_idx[wnid],
        "split": assignment[fname],
        "sha256": hashes[fname],
        "dataset": dataset_name,
        "source": dataset_name,
        "condition": None,
    } for fname in files]
    return entries, is_small, n


# -----------------------------------------------------------------------------
# ImageNet-A
# -----------------------------------------------------------------------------

def get_imagenet_a_indices_in_1k(meta_dir: str):
    cache = os.path.join(meta_dir, "imagenet_a_indices_in_1k.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    text = _fetch_and_cache_text(os.path.join(meta_dir, "_natural_adv_eval.py.cache"),
                                  _NAE_EVAL_URL)
    m = re.search(r"thousand_k_to_200\s*=\s*(\{[^}]*\})", text, re.DOTALL)
    if not m:
        raise RuntimeError("could not find thousand_k_to_200 in upstream eval.py")
    mapping = ast.literal_eval(m.group(1))
    indices_in_1k = sorted(k for k, v in mapping.items() if v != -1)
    with open(cache, "w") as f:
        json.dump(indices_in_1k, f)
    return indices_in_1k


def build_imagenet_a(paths, wnid_to_idx, seed):
    raw_dir = os.path.join(paths["raw"], "imagenet_a")
    meta_dir = paths["meta"]
    dir_wnids = list_class_dirs(raw_dir)
    ref_indices = set(get_imagenet_a_indices_in_1k(meta_dir))
    dir_indices = {wnid_to_idx[w] for w in dir_wnids}
    if dir_indices != ref_indices:
        raise RuntimeError(
            "imagenet_a class-index mismatch between extracted dirs and "
            f"natural-adv-examples' indices_in_1k: "
            f"dirs-only={sorted(dir_indices - ref_indices)}, "
            f"ref-only={sorted(ref_indices - dir_indices)}")

    rng = random.Random(seed)
    entries = []
    for w in dir_wnids:
        class_entries, _, _ = _stratified_class_entries(
            os.path.join(raw_dir, w), w, wnid_to_idx, rng, "imagenet_a",
            test_ratio=0.2, val_ratio=0.2, min_class_size=5,
            small_class_all_test=False)
        entries.extend(class_entries)
    subset_indices = sorted(ref_indices)
    with open(os.path.join(meta_dir, "imagenet_a_wnids.txt"), "w") as f:
        f.write("\n".join(sorted(dir_wnids)) + "\n")
    return entries, subset_indices, sorted(dir_wnids)


# -----------------------------------------------------------------------------
# ImageNet-Sketch
# -----------------------------------------------------------------------------

def build_imagenet_sketch(paths, wnid_to_idx, seed):
    base = os.path.join(paths["raw"], "imagenet_sketch")
    root = _find_wnid_root(base)
    dir_wnids = list_class_dirs(root)
    if len(dir_wnids) != 1000:
        raise RuntimeError(
            f"imagenet_sketch: expected 1000 class dirs under {root}, "
            f"found {len(dir_wnids)}")

    rng = random.Random(seed)
    entries = []
    dropped_small_classes = []
    for w in dir_wnids:
        class_entries, is_small, n = _stratified_class_entries(
            os.path.join(root, w), w, wnid_to_idx, rng, "imagenet_sketch",
            test_ratio=0.15, val_ratio=0.15, min_class_size=3,
            small_class_all_test=True)
        if is_small:
            dropped_small_classes.append((w, n))
        entries.extend(class_entries)
    if dropped_small_classes:
        print(f"imagenet_sketch: {len(dropped_small_classes)} classes had <3 "
              f"images (all assigned to test): {dropped_small_classes}")
    return entries, None, dir_wnids  # full 1000-way: no logit mask (subset_indices=None)


# -----------------------------------------------------------------------------
# ImageNet-ES — requires a hand-written config after inspecting the real tree
# -----------------------------------------------------------------------------

def inspect_imagenet_es(paths):
    base = os.path.join(paths["raw"], "imagenet_es")
    if not os.path.isdir(base):
        raise RuntimeError(f"{base} does not exist — download it first")
    print(f"=== imagenet_es on-disk tree (depth<=4) under {base} ===")
    for dirpath, dirnames, filenames in os.walk(base):
        depth = dirpath[len(base):].count(os.sep)
        if depth >= 4:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{os.path.basename(dirpath) or dirpath}/ "
              f"({len(dirnames)} dirs, {len(filenames)} files)")
        dirnames.sort()


def build_imagenet_es(paths, wnid_to_idx, seed):
    meta_dir = paths["meta"]
    config_path = os.path.join(meta_dir, "imagenet_es_config.json")
    if not os.path.exists(config_path):
        raise RuntimeError(
            f"{config_path} missing. Run with --dataset imagenet_es --inspect "
            "first, then hand-write imagenet_es_config.json describing "
            "{'splits': {'train': [...], 'val': [...], 'test': [...]}, "
            "'reference_dirs': [...], 'dark_condition_predicate': '...'} "
            "against the REAL directory names (spec mandates inspecting "
            "before writing parsing logic — the upstream tree isn't fully "
            "documented).")
    with open(config_path) as f:
        config = json.load(f)
    base = os.path.join(paths["raw"], "imagenet_es")
    wnid_re = re.compile(r"n\d{8}")
    reference_dirs = set(config.get("reference_dirs", []))

    entries = []
    # Tiny-ImageNet's 200 wnids aren't assumed to all be in ILSVRC-2012 (spec
    # S4.3) — collected in the same walk as `entries` (every child dirname
    # seen, not just ones already in wnid_to_idx) instead of a second full
    # tree walk over the same ~26GB directory.
    tin_wnids_on_disk = set()
    for split, rel_dirs in config["splits"].items():
        for rel_dir in rel_dirs:
            split_root = os.path.join(base, rel_dir)
            for dirpath, dirnames, filenames in os.walk(split_root):
                if any(ref in dirpath.split(os.sep) for ref in reference_dirs):
                    dirnames[:] = []
                    continue
                tin_wnids_on_disk.update(d for d in dirnames if wnid_re.match(d))
                m = wnid_re.search(dirpath)
                if not m or m.group(0) not in wnid_to_idx:
                    continue
                wnid = m.group(0)
                condition = os.path.relpath(dirpath, split_root)
                for fname in sorted(f for f in filenames
                                     if f.lower().endswith(IMG_EXTS)):
                    entries.append({
                        "image": os.path.join(dirpath, fname),
                        "wnid": wnid,
                        "label": wnid_to_idx[wnid],
                        "split": split,
                        "dataset": "imagenet_es",
                        "source": config.get("source", "imagenet_es"),
                        "condition": condition,
                    })

    dropped = sorted(w for w in tin_wnids_on_disk if w not in wnid_to_idx)
    if dropped:
        print(f"imagenet_es: {len(dropped)}/{ len(tin_wnids_on_disk)} "
              f"Tiny-ImageNet wnids are NOT in the canonical ILSVRC-1k index "
              f"and were dropped: {dropped}")
    kept = sorted(w for w in tin_wnids_on_disk if w in wnid_to_idx)
    print(f"imagenet_es: Tiny-ImageNet(200) ∩ ILSVRC-1k(1000) = {len(kept)}")

    subset_indices = sorted(wnid_to_idx[w] for w in kept)
    with open(os.path.join(meta_dir, "imagenet_es_wnids.txt"), "w") as f:
        f.write("\n".join(kept) + "\n")
    return entries, subset_indices, kept


# -----------------------------------------------------------------------------
# Common-subset intersection (spec S4.4)
# -----------------------------------------------------------------------------

def build_common_wnids(paths):
    meta_dir = paths["meta"]
    sets = {}
    for name, fname in [("imagenet_r", "imagenet_r_wnids.txt"),
                         ("imagenet_a", "imagenet_a_wnids.txt"),
                         ("imagenet_es", "imagenet_es_wnids.txt")]:
        p = os.path.join(meta_dir, fname)
        if not os.path.exists(p):
            print(f"build_common_wnids: {p} missing, skipping intersection "
                  f"(build {name}'s manifest first)")
            return
        with open(p) as f:
            sets[name] = {l.strip() for l in f if l.strip()}
    common = sets["imagenet_r"] & sets["imagenet_a"] & sets["imagenet_es"]
    with open(os.path.join(meta_dir, "common_wnids.txt"), "w") as f:
        f.write("\n".join(sorted(common)) + "\n")
    print(f"common_wnids: {len(common)} wnids in all three 200-class sets")


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

_BUILDERS = {
    "imagenet_r": build_imagenet_r,
    "imagenet_a": build_imagenet_a,
    "imagenet_sketch": build_imagenet_sketch,
    "imagenet_es": build_imagenet_es,
}
_SOURCE_INFO = {
    "imagenet_r": {"url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
                   "md5": "a61312130a589d0ca1a8fca1f2bd3337"},
    "imagenet_a": {"url": "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
                   "md5": "c3e55429088dc681f30d81f4726b6595"},
    "imagenet_sketch": {"url": "https://huggingface.co/datasets/songweig/imagenet_sketch/"
                                "resolve/main/data/ImageNet-Sketch.zip"},
    "imagenet_es": {"url": "https://huggingface.co/datasets/Edw2n/ImageNet-ES/"
                            "resolve/main/ImageNet-ES.zip", "variant": "standard (not ES-Diverse)"},
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True,
                    choices=list(_BUILDERS) + ["common", "all"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--inspect", action="store_true",
                    help="imagenet_es only: print the on-disk tree and exit")
    return p.parse_args()


def run_one(name, paths, wnid_to_idx, seed, log):
    entries, subset_indices, wnids = _BUILDERS[name](paths, wnid_to_idx, seed)
    out_path = os.path.join(paths["manifests"], f"{name}.json")
    entries = _write_entries(entries, out_path, name)
    meta_path = os.path.join(paths["meta"], f"{name}_meta.json")
    _write_meta(meta_path, entries, subset_indices, wnids, seed, _SOURCE_INFO[name])
    counts = {}
    for e in entries:
        counts[e["split"]] = counts.get(e["split"], 0) + 1
    msg = (f"{name}: wrote {len(entries)} entries ({len(wnids)} classes) "
           f"-> {out_path}; splits={counts}")
    print(msg)
    log.write(msg + "\n")


def main():
    args = parse_args()
    paths = _paths()
    os.makedirs(paths["logs"], exist_ok=True)
    if args.dataset == "imagenet_es" and args.inspect:
        inspect_imagenet_es(paths)
        return
    if args.dataset == "common":
        build_common_wnids(paths)
        return
    wnid_to_idx = load_class_index(paths["meta"])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    targets = list(_BUILDERS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        if name == "imagenet_es" and not os.path.exists(
                os.path.join(paths["meta"], "imagenet_es_config.json")):
            print("imagenet_es: no imagenet_es_config.json yet — skipping "
                  "(run --dataset imagenet_es --inspect, write the config, "
                  "then re-run --dataset imagenet_es)")
            continue
        log_path = os.path.join(paths["logs"], f"prepare_{name}_{ts}.log")
        with open(log_path, "w") as log:
            run_one(name, paths, wnid_to_idx, args.seed, log)

    if args.dataset == "all":
        build_common_wnids(paths)


if __name__ == "__main__":
    main()
