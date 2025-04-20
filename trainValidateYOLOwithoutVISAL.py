import torch
from ultralytics import YOLO                                                       

# --- Monkey patch torch.load to force weights_only=False ---
_original_torch_load = torch.load  # Keep the original reference

def custom_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False  # Force loading of the entire checkpoint
    return _original_torch_load(*args, **kwargs)

torch.load = custom_torch_load

# Load the YOLOv8 model checkpoint (YOLOv8s)
model = YOLO('yolov8s.pt')
print("YOLOv8 model loaded successfully with weights_only=False.")

# ---------------------------
# Train the Model
# ---------------------------
results = model.train(
    data='data.yaml',     # Data configuration file
    epochs=100,           # Maximum epochs
    imgsz=640,            # Training image size
    batch=16,             # Batch size (adjust as per your GPU)
    optimizer='SGD',      # Optimizer choice; can also try 'Adam'
    momentum=0.937,       # Momentum (relevant for SGD)
    weight_decay=5e-4,    # Weight decay for regularization
    lr0=0.01,             # Initial learning rate
    patience=10,          # Early stopping patience                            
    verbose=True          # Verbose logging
)

# ---------------------------
# Validate the Model
# ---------------------------
metrics = model.val(data='data.yaml')                                              

# The metrics are available in the 'box' attribute.
# We can retrieve:
# - mAP50: via metrics.box.map50()
# - mAP50-95: via metrics.box.map()
# - Mean precision: via metrics.box.mp()
# - Mean recall: via metrics.box.mr()
# We'll compute F1 score as: 2*(P*R)/(P+R)

mp = metrics.box.mp()  # Mean Precision
mr = metrics.box.mr()  # Mean Recall
f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0

print("\nValidation Metrics:")
print(f"mAP50: {metrics.box.map50():.3f}")
print(f"mAP50-95: {metrics.box.map():.3f}")
print(f"Precision: {mp:.3f}")
print(f"Recall: {mr:.3f}")
print(f"F1 Score: {f1:.3f}")                                                                                      
