import os
from collections import Counter

# 1) Update this to point at your dataset root
DATASET_ROOT = r"C:\Users\venka\Desktop\finalMiniProject\dataset"

# 2) The 6 classes in numeric order
CLASS_NAMES = [
    "red blood cell",
    "leukocyte",
    "trophozoite",
    "ring",
    "schizont",
    "gametocyte",
]

# 3) Which splits to include
SPLITS = ["train"]

def count_labels(root, splits, class_names):
    counts = Counter()
    for split in splits:
        # adjust if your labels live under "images" folder
        label_dir = os.path.join(root, split, "labels")
        if not os.path.isdir(label_dir):
            print(f"⚠️  Warning: '{label_dir}' not found, skipping")
            continue

        for fn in os.listdir(label_dir):
            if not fn.lower().endswith(".txt"):
                continue
            path = os.path.join(label_dir, fn)
            with open(path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        # some files may have "0.0" etc., so cast via float
                        cls_id = int(float(parts[0]))
                    except Exception:
                        # skip malformed lines
                        continue
                    if 0 <= cls_id < len(class_names):
                        counts[cls_id] += 1

    # Print summary
    total = sum(counts.values())
    print("\nSample counts by class (all splits combined):\n")
    for idx, name in enumerate(class_names):
        cnt = counts[idx]
        pct = cnt / total * 100 if total else 0
        print(f"  {idx:>2d} {name:15s}: {cnt:6d}  ({pct:5.2f}%)")
    print(f"\n  Total annotated cells: {total}")

if __name__ == "__main__":
    count_labels(DATASET_ROOT, SPLITS, CLASS_NAMES)
