import os
import shutil
import random
import cv2
import albumentations as A
import yaml
from tqdm import tqdm

# ---------------------------
# Configuration
# ---------------------------
# Input dataset directories
INPUT_IMAGES_DIR = "malaria_dataset/malaria/images"
INPUT_LABELS_DIR = "malaria_dataset/malaria/labels"

# Output base directory for split data
OUTPUT_BASE = "dataset"
SPLIT_DIRS = {
    "train": {"images": os.path.join(OUTPUT_BASE, "train", "images"),
              "labels": os.path.join(OUTPUT_BASE, "train", "labels")},
    "val":   {"images": os.path.join(OUTPUT_BASE, "val", "images"),
              "labels": os.path.join(OUTPUT_BASE, "val", "labels")},
    "test":  {"images": os.path.join(OUTPUT_BASE, "test", "images"),
              "labels": os.path.join(OUTPUT_BASE, "test", "labels")}
}

# Create output directories if they don't exist.
for split in SPLIT_DIRS:
    for folder in SPLIT_DIRS[split].values():
        os.makedirs(folder, exist_ok=True)

# Infected cell classes (YOLO class indices)
# 0: red blood cell, 1: leukocyte, 2: trophozoite, 3: ring, 4: schizont, 5: gametocyte
INFECTED_CLASSES = {2, 3, 4, 5}

# ---------------------------
# Step 1: Filter for Infected Images
# ---------------------------
def is_infected(label_path):
    """
    Returns True if the label file (YOLO format) contains any infected cell class.
    Each label line is expected to be: <class> <xc> <yc> <w> <h>
    """
    if not os.path.exists(label_path):
        return False
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    if cls in INFECTED_CLASSES:
                        return True
                except ValueError:
                    continue
    return False

# Get list of all images in input folder that are infected.
all_images = [f for f in os.listdir(INPUT_IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
infected_images = []
for img in all_images:
    label_file = os.path.splitext(img)[0] + ".txt"
    label_path = os.path.join(INPUT_LABELS_DIR, label_file)
    if is_infected(label_path):
        infected_images.append(img)

print(f"Found {len(infected_images)} infected images out of {len(all_images)} total images.")

# ---------------------------
# Step 2: Split Infected Images
# ---------------------------
random.shuffle(infected_images)
total = len(infected_images)
train_split = int(0.8 * total)
val_split = int(0.9 * total)  # 80% train, 10% val, 10% test

train_images = infected_images[:train_split]
val_images   = infected_images[train_split:val_split]
test_images  = infected_images[val_split:]

print(f"Dataset split: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test infected images.")

def copy_files(image_list, split):
    for img_file in tqdm(image_list, desc=f"Copying {split} files"):
        label_file = os.path.splitext(img_file)[0] + ".txt"
        src_img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
        src_label_path = os.path.join(INPUT_LABELS_DIR, label_file)
        dst_img_path = os.path.join(SPLIT_DIRS[split]["images"], img_file)
        dst_label_path = os.path.join(SPLIT_DIRS[split]["labels"], label_file)
        shutil.copy2(src_img_path, dst_img_path)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"Warning: Label file not found for {img_file}")

copy_files(train_images, "train")
copy_files(val_images, "val")
copy_files(test_images, "test")

# ---------------------------
# Step 3: Augmentation for Infected Images in Training Set (in-place)
# ---------------------------
# Define an augmentation pipeline using Albumentations.
augment = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

def clamp_bbox(bbox):
    """
    Clamp each value in the bbox to the range [0.0, 1.0].
    bbox is expected to be [xc, yc, w, h].
    """
    return [max(0.0, min(val, 1.0)) for val in bbox]

def augment_image(image_path, label_path, aug_count=3):
    """
    Reads an image and its YOLO-format label file,
    applies augmentation, and saves augmented image and label
    into the same training directories.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image {image_path}")
        return

    # Read labels (each line: <class> <xc> <yc> <w> <h>)
    boxes, class_labels = [], []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:])
                    boxes.append([xc, yc, w, h])
                    class_labels.append(cls)
                except ValueError:
                    continue

    # Proceed only if the image is infected (should be true by filtering)
    if not any(cls in INFECTED_CLASSES for cls in class_labels):
        return

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for i in range(aug_count):
        try:
            augmented = augment(image=image, bboxes=boxes, class_labels=class_labels)
        except ValueError as e:
            # If there's a ValueError, clamp input boxes and try again.
            boxes_fixed = [clamp_bbox(box) for box in boxes]
            try:
                augmented = augment(image=image, bboxes=boxes_fixed, class_labels=class_labels)
            except Exception as e2:
                print(f"Failed to augment {image_path}: {e2}")
                continue

        aug_image = augmented['image']
        aug_bboxes = augmented['bboxes']
        aug_class_labels = augmented['class_labels']

        # Clamp augmented boxes
        aug_bboxes = [clamp_bbox(list(bbox)) for bbox in aug_bboxes]

        aug_img_name = f"{base_name}_aug_{i}.jpg"
        aug_label_name = f"{base_name}_aug_{i}.txt"

        # Save augmented image in the same train images directory.
        cv2.imwrite(os.path.join(SPLIT_DIRS["train"]["images"], aug_img_name), aug_image)
        # Save augmented label file.
        with open(os.path.join(SPLIT_DIRS["train"]["labels"], aug_label_name), "w") as out_f:
            for bbox, cls in zip(aug_bboxes, aug_class_labels):
                out_f.write(f"{cls} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

# Apply augmentation to each training image.
for img_file in tqdm(train_images, desc="Augmenting train images"):
    label_file = os.path.splitext(img_file)[0] + ".txt"
    img_path = os.path.join(SPLIT_DIRS["train"]["images"], img_file)
    label_path = os.path.join(SPLIT_DIRS["train"]["labels"], label_file)
    if os.path.exists(img_path) and os.path.exists(label_path):
        augment_image(img_path, label_path, aug_count=3)
    else:
        print(f"Skipping augmentation for {img_file}: missing image or label.")

print("Dataset splitting and augmentation complete!")

# ---------------------------
# Step 4: Create YOLOv8 Data Config (data.yaml)
# ---------------------------
data_yaml = {
    "train": os.path.abspath(SPLIT_DIRS["train"]["images"]),
    "val": os.path.abspath(SPLIT_DIRS["val"]["images"]),
    "nc": 6,
    "names": ["red blood cell", "leukocyte", "trophozoite", "ring", "schizont", "gametocyte"]
}

with open("data.yaml", "w") as f:
    yaml.dump(data_yaml, f)

print("data.yaml created.")
