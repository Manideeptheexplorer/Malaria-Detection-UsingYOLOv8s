import torch
from ultralytics import YOLO

# Load the trained model (use your trained checkpoint)
model = YOLO('runs/detect/train2/weights/best.pt')  # Adjust path if needed

# Run inference on the test dataset
results = model.val(
    data='data.yaml',   # Path to your dataset YAML file
    split='train'        # Explicitly specify test set
)

# Extract metrics
mp = results.box.mp  # Mean Precision (access as attribute)
mr = results.box.mr  # Mean Recall (access as attribute)
map50 = results.box.map50  # mAP@50 (access as attribute)
map50_95 = results.box.map  # mAP@50-95 (access as attribute)

# Compute F1 Score
f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0

# Display results
print("\nTest Set Metrics:")
print(f"mAP@50: {map50:.3f}")
print(f"mAP@50-95: {map50_95:.3f}")
print(f"Precision: {mp:.3f}")
print(f"Recall: {mr:.3f}")
print(f"F1 Score: {f1:.3f}")
