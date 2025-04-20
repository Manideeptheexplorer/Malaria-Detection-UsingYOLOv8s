import torch
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
import torch.nn as nn

# Custom VISAL + Fewshot Loss
class VISALFewshotLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.base_loss = v8DetectionLoss(model.model)  # pass top-level model
        # Class sample counts (from your earlier analysis)
        self.class_counts = torch.tensor([190569, 234, 4744, 1551, 576, 428], dtype=torch.float32)
        self.class_weights = 1.0 / (self.class_counts + 1e-6)
        self.class_weights /= self.class_weights.sum()  # normalize

        # Margin per class (can be tuned)
        self.margins = torch.tensor([0.1, 0.3, 0.2, 0.25, 0.25, 0.3])

    def forward(self, preds, targets, anchors):
        loss_dict = self.base_loss(preds, targets, anchors)

        # Add VISAL-style margin loss
        cls_loss = loss_dict['cls']
        batch_size = targets.shape[0] if targets.numel() > 0 else 1

        margin_loss = 0.0
        for t in targets:
            cls = int(t[1])  # class index
            margin_loss += self.margins[cls] * self.class_weights[cls]
        margin_loss = margin_loss / batch_size

        # Combine losses
        total_loss = loss_dict['box'] + loss_dict['cls'] + loss_dict['dfl'] + margin_loss
        return total_loss, loss_dict  # must return a tuple

# Load model
print("🚀 Loading YOLOv8s model...")
model = YOLO("yolov8s.pt")

# Inject custom loss
print("🔧 Injecting VISAL+Fewshot loss...")
model.model.loss = VISALFewshotLoss(model)

# Freeze backbone to improve generalization
for name, param in model.model.named_parameters():
    if "backbone" in name:
        param.requires_grad = False
print("✅ Backbone layers frozen.")

# Train
print("🏋️ Starting training...")
results = model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.001,
    weight_decay=0.0005,
    optimizer='Adam',
    patience=10,
    device=0 if torch.cuda.is_available() else 'cpu',
    val=True,
    verbose=True
)

# Validation metrics
print("📊 Validation results:")
print(results)
