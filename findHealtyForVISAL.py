import json

# Define the set of infected cell categories
infected_categories = {'trophozoite', 'ring', 'schizont', 'gametocyte'}

# Load the JSON data
with open('malaria_dataset\\malaria\\training.json', 'r') as file:
    data = json.load(file)

# Iterate through each image entry
for image_entry in data:
    # Check if any object in the image is an infected cell
    if not any(obj['category'] in infected_categories for obj in image_entry.get('objects', [])):
        print(f"Image without infected cells: {image_entry['image']['pathname']}")
