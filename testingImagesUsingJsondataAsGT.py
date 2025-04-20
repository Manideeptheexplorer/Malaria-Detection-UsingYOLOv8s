import json
import os
from ultralytics import YOLO

# Load the trained YOLO model
model = YOLO('runs/detect/train2/weights/best.pt')  # Update path if needed

# Define infected classes
infected_classes = {"trophozoite", "ring", "schizont", "gametocyte"}

# Load test JSON file
test_json_path = "C:/Users/venka/Desktop/finalMiniProject/malaria_dataset/malaria/test.json"
test_images_root = "C:/Users/venka/Desktop/finalMiniProject/malaria_dataset/malaria"  # Root directory

# Load JSON data
with open(test_json_path, "r") as file:
    test_data = json.load(file)

# Debug: Check if JSON loaded correctly
if not test_data:
    print("Error: test.json is empty or not formatted correctly!")
    exit()

# Debug: Check a sample entry
print("Sample Entry:", test_data[0])

# Accuracy counters
correct_predictions = 0
total_images = 0

# Function to check infection status from YOLO detection
def check_infection(image_path):
    results = model(image_path)  # Run inference
    detected_classes = set()
    
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            class_name = model.names[class_id]
            detected_classes.add(class_name)

    return any(cls in infected_classes for cls in detected_classes)

# Function to check ground truth infection status
def check_ground_truth(image_data):
    return any(obj["category"] in infected_classes for obj in image_data["objects"])

# Loop through test images
for image_data in test_data:
    image_path = os.path.join(test_images_root, image_data["image"]["pathname"].lstrip("/"))

    if os.path.exists(image_path):  
        total_images += 1
        predicted_infected = check_infection(image_path)
        actual_infected = check_ground_truth(image_data)

        if predicted_infected == actual_infected:
            correct_predictions += 1
    else:
        print(f"Warning: Image Not Found {image_path}")

# Compute Accuracy
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0

# Print Final Results
print(f"Total Test Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Model Accuracy: {accuracy:.2f}%")
