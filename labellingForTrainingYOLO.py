import json
import os

# Mapping of cell categories to class indices.
category_map = {
    "red blood cell": 0,
    "leukocyte": 1,
    "trophozoite": 2,
    "ring": 3,
    "schizont": 4,
    "gametocyte": 5
}

# Path to your JSON file containing annotations.
json_file_path = "malaria_dataset/malaria/test.json"  # Update if necessary.

# Load the JSON data.
with open(json_file_path, "r") as f:
    data = json.load(f)

# Folder where you want to save the label files.
labels_output_dir = "labels"  # Change as needed.
os.makedirs(labels_output_dir, exist_ok=True)

# Process each image entry in the JSON.
for entry in data:
    # Get image details.
    image_info = entry["image"]
    # For YOLO, we need the image dimensions.
    img_height = image_info["shape"]["r"]
    img_width = image_info["shape"]["c"]
    
    # Extract the image filename from the pathname.
    image_pathname = image_info["pathname"]
    image_filename = os.path.basename(image_pathname)
    # Create label filename using the same base name with a .txt extension.
    label_filename = os.path.splitext(image_filename)[0] + ".txt"
    label_file_path = os.path.join(labels_output_dir, label_filename)
    
    lines = []
    
    # Process each object in the image.
    for obj in entry.get("objects", []):
        category = obj.get("category", "").lower()  # ensure lower case for matching
        if category not in category_map:
            # If category is not found in our mapping, skip it.
            continue
        
        cls = category_map[category]
        
        # Extract bounding box coordinates.
        r_min = obj["bounding_box"]["minimum"]["r"]
        c_min = obj["bounding_box"]["minimum"]["c"]
        r_max = obj["bounding_box"]["maximum"]["r"]
        c_max = obj["bounding_box"]["maximum"]["c"]
        
        # Compute YOLO-format values:
        # x_center and y_center are the center of the box, normalized.
        x_center = ((c_min + c_max) / 2.0) / img_width
        y_center = ((r_min + r_max) / 2.0) / img_height
        # width and height of the box normalized.
        box_width = (c_max - c_min) / img_width
        box_height = (r_max - r_min) / img_height
        
        # Create a line in YOLO format.
        line = f"{cls} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        lines.append(line)
    
    # Write the label file.
    with open(label_file_path, "w") as out_file:
        out_file.write("\n".join(lines))
    
    print(f"Label file created: {label_file_path}")
