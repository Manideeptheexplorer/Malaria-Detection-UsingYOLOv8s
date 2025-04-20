import torch
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')  # Change path if necessary

# Define infected classes
infected_classes = ["trophozoite", "ring", "schizont", "gametocyte"]

# Function to check infection status
def check_infection(image_path):
    results = model(image_path)  # Run inference

    detected_classes = set()
    
    # Extract detected class names
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())  # Get class ID
            class_name = model.names[class_id]  # Convert ID to class name
            detected_classes.add(class_name)

    print(f"Detected Classes: {detected_classes}")

    # If any infected class is detected, mark as infected
    if any(cls in infected_classes for cls in detected_classes):
        print(f"Result: ❌ The image is INFECTED!")
    else:
        print(f"Result: ✅ The image is UNINFECTED!")

# Test on a sample image
image_path = r"C:\Users\venka\Desktop\finalMiniProject\malaria_dataset\malaria\images\07a96126-482e-4beb-b42a-0cb59e505372.png"  # Change this to your test image path
check_infection(image_path)
