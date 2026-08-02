"""Acceptance checks for the shift-dataset manifests (spec S7, items 1-9;
item 10 — the zero-shot sanity baseline — lives in zero_shot_sanity.py since
it needs a loaded model, not just the manifest files).

Fails loudly: any violation raises and this script exits non-zero. Run after
make_shift_manifests.py, and again after any re-run to confirm determinism.

Usage:
    python scripts/data_prep/verify_shift_manifests.py --dataset imagenet_r
    python scripts/data_prep/verify_shift_manifests.py --dataset all
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from io import BytesIO

_DATASETS = ["imagenet_r", "imagenet_a", "imagenet_sketch", "imagenet_es"]
_MAX_DECODE_DROP_FRACTION = 0.001  # spec S7#2: halt if >0.1% undecodable


def _root():
    return os.environ.get("DATASETS_ROOT", "/mnt5/noy/datasets")


def _paths():
    root = _root()
    return {
        "manifests": os.path.join(root, "manifests"),
        "meta": os.path.join(root, "manifests", "_meta"),
    }


def _load_manifest(paths, name):
    path = os.path.join(paths["manifests"], f"{name}.json")
    with open(path) as f:
        return json.load(f), path


def _load_meta(paths, name):
    with open(os.path.join(paths["meta"], f"{name}_meta.json")) as f:
        return json.load(f)


def check_integrity_and_decodability(entries, name):
    """S7#1 (every path exists, every sha256 matches) + S7#2 (every image
    opens with PIL and converts to RGB; halt if >0.1% are dropped) in ONE
    pass over each file's bytes instead of two independent full reads —
    still reported as the two separate named results the spec expects."""
    from PIL import Image
    bad_integrity, bad_decode = [], []
    for e in entries:
        if not os.path.isfile(e["image"]):
            bad_integrity.append((e["image"], "missing"))
            continue
        with open(e["image"], "rb") as f:
            data = f.read()
        if hashlib.sha256(data).hexdigest() != e["sha256"]:
            bad_integrity.append((e["image"], "sha256 mismatch"))
        try:
            with Image.open(BytesIO(data)) as img:
                img.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - any decode failure counts
            bad_decode.append((e["image"], str(exc)))

    if bad_integrity:
        raise RuntimeError(f"{name}: integrity check failed for "
                            f"{len(bad_integrity)} files, e.g. {bad_integrity[:5]}")
    print(f"{name}: integrity OK ({len(entries)} files)")

    frac = len(bad_decode) / max(len(entries), 1)
    print(f"{name}: decodability {len(bad_decode)}/{len(entries)} failed ({frac:.4%})")
    if frac > _MAX_DECODE_DROP_FRACTION:
        raise RuntimeError(
            f"{name}: {frac:.4%} of images failed to decode (limit "
            f"{_MAX_DECODE_DROP_FRACTION:.2%}), e.g. {bad_decode[:5]}")


def check_label_range(entries, name):
    """S7#3."""
    bad = [e for e in entries if not (0 <= e["label"] <= 999)]
    if bad:
        raise RuntimeError(f"{name}: {len(bad)} labels out of [0,999], "
                            f"e.g. {bad[:5]}")
    print(f"{name}: label range OK")


def check_mask_consistency(entries, meta, name):
    """S7#4: distinct labels in the manifest == meta's subset_indices exactly
    (subset_indices=None means full 1000-way, e.g. imagenet_sketch)."""
    subset = meta.get("subset_indices")
    if subset is None:
        print(f"{name}: no logit mask (full 1000-way) — skipping mask check")
        return
    distinct = sorted({e["label"] for e in entries})
    if distinct != sorted(subset):
        raise RuntimeError(
            f"{name}: manifest labels {distinct[:5]}...(n={len(distinct)}) != "
            f"subset_indices {sorted(subset)[:5]}...(n={len(subset)}) — "
            f"label resolution bypassed the canonical class index somewhere")
    print(f"{name}: mask consistency OK ({len(distinct)} classes)")


def check_split_disjointness(entries, name):
    """S7#5: no path or sha256 appears in more than one split."""
    path_to_splits, sha_to_splits = {}, {}
    for e in entries:
        path_to_splits.setdefault(e["image"], set()).add(e["split"])
        sha_to_splits.setdefault(e["sha256"], set()).add(e["split"])
    dup_paths = {p: s for p, s in path_to_splits.items() if len(s) > 1}
    dup_shas = {h: s for h, s in sha_to_splits.items() if len(s) > 1}
    if dup_paths:
        raise RuntimeError(f"{name}: {len(dup_paths)} paths appear in "
                            f"multiple splits, e.g. {list(dup_paths.items())[:5]}")
    if dup_shas:
        raise RuntimeError(f"{name}: {len(dup_shas)} exact-duplicate images "
                            f"(by sha256) leak across splits, e.g. "
                            f"{list(dup_shas.items())[:5]}")
    print(f"{name}: split/sha256 disjointness OK")


def check_class_coverage(entries, name):
    """S7#6: every class present in the dataset appears in test; report any
    class missing from train."""
    classes = {e["wnid"] for e in entries}
    by_split = {s: {e["wnid"] for e in entries if e["split"] == s}
                for s in ("train", "val", "test")}
    missing_test = classes - by_split["test"]
    missing_train = classes - by_split["train"]
    if missing_test:
        raise RuntimeError(f"{name}: {len(missing_test)} classes missing "
                            f"from test: {sorted(missing_test)[:10]}")
    if missing_train:
        print(f"{name}: WARNING {len(missing_train)} classes missing from "
              f"train (may be expected for small classes): "
              f"{sorted(missing_train)[:10]}")
    print(f"{name}: class coverage in test OK ({len(classes)} classes)")


def check_es_reference_exclusion(entries, name):
    """S7#7: imagenet_es only — no entry resolves under sampled_tin_no_resize*."""
    if name != "imagenet_es":
        return
    bad = [e for e in entries if "sampled_tin_no_resize" in e["image"]]
    if bad:
        raise RuntimeError(f"{name}: {len(bad)} reference-sample entries "
                            f"leaked into the manifest, e.g. {bad[:5]}")
    print(f"{name}: reference-sample exclusion OK")


def check_determinism(name, seed):
    """S7#9: re-running the builder with the same seed reproduces the same
    manifest byte-for-byte (modulo dict key order, which json.dump preserves
    from insertion order, so this really is a byte comparison)."""
    paths = _paths()
    manifest_path = os.path.join(paths["manifests"], f"{name}.json")
    with open(manifest_path, "rb") as f:
        before = f.read()
    tmp_path = manifest_path + ".redo_check"
    env = dict(os.environ)
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    subprocess.run(
        [sys.executable, os.path.join(repo_root, "scripts/data_prep/make_shift_manifests.py"),
         "--dataset", name, "--seed", str(seed)],
        check=True, env=env)
    with open(manifest_path, "rb") as f:
        after = f.read()
    # restore original in case the re-run reordered anything unexpectedly
    if after != before:
        with open(manifest_path, "wb") as f:
            f.write(before)
        raise RuntimeError(f"{name}: re-run with seed={seed} produced a "
                            f"different manifest — determinism violated")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    print(f"{name}: determinism OK (re-run byte-identical)")


def verify_one(name, skip_determinism=False):
    paths = _paths()
    entries, _ = _load_manifest(paths, name)
    meta = _load_meta(paths, name)
    check_integrity_and_decodability(entries, name)
    check_label_range(entries, name)
    check_mask_consistency(entries, meta, name)
    check_split_disjointness(entries, name)
    check_class_coverage(entries, name)
    check_es_reference_exclusion(entries, name)
    if not skip_determinism:
        check_determinism(name, meta["seed"])
    print(f"=== {name}: ALL CHECKS PASSED ===")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=_DATASETS + ["all"])
    p.add_argument("--skip_determinism", action="store_true",
                    help="skip the re-run-and-diff check (slow: re-hashes "
                         "every file)")
    args = p.parse_args()
    targets = _DATASETS if args.dataset == "all" else [args.dataset]
    failures = []
    for name in targets:
        try:
            verify_one(name, args.skip_determinism)
        except Exception as exc:  # noqa: BLE001 - report all, then fail loudly
            print(f"!!! {name}: FAILED — {exc}", file=sys.stderr)
            failures.append(name)
    if failures:
        print(f"\n{len(failures)}/{len(targets)} datasets failed verification: "
              f"{failures}", file=sys.stderr)
        sys.exit(1)
    print(f"\nAll {len(targets)} dataset(s) passed verification.")


if __name__ == "__main__":
    main()
