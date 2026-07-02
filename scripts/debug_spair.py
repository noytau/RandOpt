"""One-shot script to inspect SPair-71k annotation format on the cluster."""
import json
import glob
import os

root = "/storage/noy/RandOpt/data/spair71k/SPair-71k"

print("=== Directory structure ===")
for d in sorted(os.listdir(root)):
    print(f"  {d}/")

print("\n=== PairAnnotation splits ===")
pair_dir = os.path.join(root, "PairAnnotation")
for split in sorted(os.listdir(pair_dir)):
    files = glob.glob(f"{pair_dir}/{split}/**/*.json", recursive=True)
    print(f"  {split}: {len(files)} files")

print("\n=== First annotation JSON ===")
files = sorted(glob.glob(f"{pair_dir}/trn/**/*.json", recursive=True))
d = json.load(open(files[0]))
print(f"File: {files[0]}")
print(f"Keys: {list(d.keys())}")
print()
for k, v in d.items():
    print(f"  {k}: {repr(v)[:150]}")

print("\n=== Image path check ===")
cat = d.get("category", "")
src = d.get("src_imname", d.get("src_path", ""))
for img_dir in ["JPEGImages", "ImageData", "images"]:
    path = os.path.join(root, img_dir, cat, src)
    exists = os.path.exists(path)
    print(f"  {img_dir}/{cat}/{src}: {'EXISTS' if exists else 'missing'}")
