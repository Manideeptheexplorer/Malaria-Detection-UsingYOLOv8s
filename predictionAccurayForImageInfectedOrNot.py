import os
from ultralytics import YOLO

# Load the trained model checkpoint.
model = YOLO("runs/detect/train2/weights/best.pt")
print("Trained YOLOv8 model loaded successfully.")

# Update these paths to your test images and labels directories.
test_img_dir = "malaria_dataset/malaria/images"    # e.g., "C:/Users/venka/Desktop/MINI PROJECT/datasets/dataset/test/images"
test_label_dir = "malaria_dataset/malaria/labels"    # e.g., "C:/Users/venka/Desktop/MINI PROJECT/datasets/dataset/test/labels"

# Define infected class indices (as per your dataset)
infected_classes = {2, 3, 4, 5}

def ground_truth_from_label(label_file):
    """
    Reads a YOLO-format label file and returns True if any label indicates an infected cell.
    Returns False if the file does not exist or no infected label is found.
    """
    if not os.path.exists(label_file):
        return False
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                cls = int(parts[0])
                if cls in infected_classes:
                    return True
    return False

def predict_infection(image_path):
    """
    Runs the model prediction on the given image.
    Returns True if any detected bounding box corresponds to an infected cell.
    """
    results = model.predict(source=image_path, conf=0.25, verbose=False)
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None and len(boxes) > 0:
            predicted_classes = boxes.cls.tolist()  # Get class predictions as a list
            # Ensure they are integers
            predicted_classes = [int(c) for c in predicted_classes]
            if any(c in infected_classes for c in predicted_classes):
                return True
    return False

# Variables to track accuracy.
total_images = 0
correct_predictions = 0

# Iterate over each image in the test directory.
for file in os.listdir(test_img_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        total_images += 1
        image_path = os.path.join(test_img_dir, file)
        # Assume label file has same base name with .txt extension.
        label_file = os.path.join(test_label_dir, os.path.splitext(file)[0] + ".txt")
        
        # Determine ground truth and model prediction.
        gt_infected = ground_truth_from_label(label_file)
        pred_infected = predict_infection(image_path)
        
        # Compare and count correct predictions.
        if gt_infected == pred_infected:
            correct_predictions += 1
        
        print(f"{file}: GT: {'infected' if gt_infected else 'uninfected'}, Pred: {'infected' if pred_infected else 'uninfected'}")

# Calculate and print overall accuracy.
accuracy = (correct_predictions / total_images) * 100 if total_images > 0 else 0
print(f"\nTotal Images: {total_images}")
print(f"Correct Predictions: {correct_predictions}")
print(f"Test Accuracy: {accuracy:.2f}%")
