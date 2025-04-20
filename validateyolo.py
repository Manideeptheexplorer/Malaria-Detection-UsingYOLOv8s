from ultralytics import YOLO
import os

# Path to the trained model
model_path = "runs/detect/train2/weights/best.pt"

# Path to your dataset YAML file
data_yaml = "data.yaml"  # Replace with actual path

# Load the model
model = YOLO(model_path)

# Run validation
metrics = model.val(data=data_yaml, split="val", save_json=True, save_txt=True, save_hybrid=True)

# Print metrics
print("\n📊 Validation Metrics:")
print(f"Precision:      {metrics.box.mp:.4f}")  # Mean precision across all classes
print(f"Recall:         {metrics.box.mr:.4f}")   # Mean recall across all classes
print(f"mAP@0.5:        {metrics.box.map50:.4f}")      # Mean AP at IoU threshold 0.5
print(f"mAP@0.5:0.95:   {metrics.box.map:.4f}")        # Mean AP across IoU thresholds from 0.5 to 0.95

# You can also access confusion matrix and per-class metrics
if hasattr(metrics, "confusion_matrix"):
    print("\n🧩 Confusion Matrix:")
    print(metrics.confusion_matrix.matrix)

print("\n✅ Validation completed successfully.")
