import os
from PIL import Image
import math
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import csv

# Load the processor and model
processor = AutoProcessor.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

model = AutoModelForCausalLM.from_pretrained(
    'cyan2k/molmo-7B-O-bnb-4bit',
    trust_remote_code=True,
    torch_dtype='auto',
    device_map='auto'
)

# Initialize model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
index_file = "scenarios_index.index"
# scenarios_file = "output_latest.txt"
payload_file = "id_to_payload.json"

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load the FAISS index
index = faiss.read_index("scenarios_index.index")

# Load the mapping of indices to JSON entries
with open("id_to_payload.json", "r") as f:
    id_to_payload = {int(k): v for k, v in json.load(f).items()}

# Function to retrieve impedance parameters of the best-matching scenario
def rag_retrieve_impedance(vlm_response, index, id_to_payload):
    # Embed VLM response
    response_embedding = embedding_model.encode(vlm_response).reshape(1, -1).astype(np.float32)
    
    # Query FAISS for all distances and indices
    distances, indices = index.search(response_embedding, k=1)  # k=1 to get the best match
    
    # FAISS returns L2 distances (smaller is closer)
    best_idx = np.argmin(distances[0])  # Find the index with the smallest distance
    best_distance = distances[0][best_idx]
    
    # Get impedance parameters and metadata
    matched_entry = id_to_payload[indices[0][best_idx]]
    impedance_params = matched_entry["impedance_parameters"]
    scenario = matched_entry["scenario"]
    match_text = matched_entry["text"]
    
    return impedance_params, scenario, match_text, best_distance


def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate the Euclidean distance between two points.
    """
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def extract_coordinates(generated_text):
    """
    Extract coordinates from the generated output text.
    The format might vary, so adjust the extraction logic based on the expected format.
    """
    # Example: Extract coordinates from points
    coordinates = []
    # Look for coordinates in the format of 'x="..", y=".."'
    import re
    matches = re.findall(r'x\d*=\s*["]*([0-9.]+)["]*\s*y\d*=\s*["]*([0-9.]+)', generated_text)
    
    for match in matches:
        x, y = float(match[0]), float(match[1])
        coordinates.append((x, y))
    
    return coordinates

def find_objects(image_path, is_hard_obstacle):
    """
    Process the mission description to find object coordinates on the map and return optimized coordinates using TSP.
    """
    if is_hard_obstacle:
        description = "This image is a top view of a scenario in which tripod stand with black base is obstacle. Point the center of each tripod stand's position in the image and tell me how many position you point."
    else:
        description = "This image is a top view of a scenario in which human is obstacle. Point the center of each human's position in the image and tell me how many position you point."

    # Process the image and text
    inputs = processor.process(
        images=[Image.open(image_path)],
        text=description
    )

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Extract coordinates from the generated text
    coordinates = extract_coordinates(generated_text)
    
    return coordinates

def compute_distances_for_image(coordinates):
    """
    Given the list of coordinates, compute the distances as per the rules.
    """
    distances = []
    
    if len(coordinates) == 2:
        # Two points (x1, y1) and (x2, y2)
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        distance = euclidean_distance(x1, y1, x2, y2)
        distances.append(distance)
        
    elif len(coordinates) == 3:
        # Three points (x1, y1), (x2, y2), and (x3, y3)
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]
        distance = euclidean_distance(x2, y2, x3, y3)
        distances.append(distance)
        
    elif len(coordinates) == 4:
        # Four points (x1, y1), (x2, y2), (x3, y3), (x4, y4)
        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]
        x4, y4 = coordinates[3]
        
        # Compute two distances: (x1, y1) -> (x2, y2) and (x3, y3) -> (x4, y4)
        distance1 = euclidean_distance(x1, y1, x2, y2)
        distance2 = euclidean_distance(x3, y3, x4, y4)
        
        distances.append(distance1)
        distances.append(distance2)
    
    return distances

def find_objects_crop(image_path, is_hard_obstacle, dist):
    """
    Process the mission description to find object coordinates on the map and return optimized coordinates using TSP.
    """
    if dist == "close":
        if is_hard_obstacle:
            description = "Analyze the drone arena image and identify all tripod stand as obstacle in the scene. Count the total number of tripod stand, specifying number of tripod stand before the gate and number of tripod stand after the gate. If there are no obstacles before the gate, mention that explicitly. Calculate the relative distances between each tripod stand and describe their spacing as 'closely spaced.' The spacing should be based on the distance between their feet in the image. Output in this format: 'Total obstacles: [total number], Hard (tripod stand), Positions: Before gate: [number of tripod stand before the gate], After gate: [number of tripod stand after the gate], Spacing: [Closely spaced based on distance], Distances between tripod stand: tripod stand1-tripod stand2: Distance in pixels/units."
        else:
            description =  "Analyze the drone arena image and identify all human obstacles in the scene. Count the total number of humans, specifying number of humans before the gate and number of humans after the gate. If there are no obstacles before the gate, mention that explicitly. Calculate the relative distances between each human and describe their spacing as 'closely spaced.' The spacing should be based on the distance between their feet in the image. Output in this format: 'Total obstacles: [total number], Soft (human), Positions: Before gate: [number of humans before the gate], After gate: [number of humans after the gate], Spacing: [Closely spaced based on distance], Distances between humans: human1-human2: Distance in pixels/units."
    elif dist == "far":
        if is_hard_obstacle:
            description = "Analyze the drone arena image and identify all tripod stand as obstacle in the scene. Count the total number of tripod stand, specifying number of tripod stand before the gate and number of tripod stand after the gate. If there are no obstacles before the gate, mention that explicitly. Calculate the relative distances between each tripod stand and describe their spacing as 'widely spaced.' The spacing should be based on the distance between their feet in the image. Output in this format: 'Total obstacles: [total number], Hard (tripod stand), Positions: Before gate: [number of tripod stand before the gate], After gate: [number of tripod stand after the gate], Spacing: [Widely spaced based on distance], Distances between tripod stand: tripod stand1-tripod stand2: Distance in pixels/units."
        else:
            description =  "Analyze the drone arena image and identify all human obstacles in the scene. Count the total number of humans, specifying number of humans before the gate and number of humans after the gate. If there are no obstacles before the gate, mention that explicitly. Calculate the relative distances between each human and describe their spacing as 'widely spaced.' The spacing should be based on the distance between their feet in the image. Output in this format: 'Total obstacles: [total number], Soft (human), Positions: Before gate: [number of humans before the gate], After gate: [number of humans after the gate], Spacing: [Widely spaced based on distance], Distances between humans: human1-human2: Distance in pixels/units."
    else:
        if is_hard_obstacle:
            description = "Analyze the drone arena image and identify all tripod stand as obstacle in the scene. Count the total number of tripod stand, specifying number of tripod stand before the gate and number of tripod stand after the gate. If there are no obstacles before the gate, mention that explicitly. Calculate the relative distances between each tripod stand and describe their spacing. The spacing should be based on the distance between their feet in the image. Output in this format: 'Total obstacles: [total number], Hard (tripod stand), Positions: Before gate: [number of tripod stand before the gate], After gate: [number of tripod stand after the gate]], Spacing: [no distance], Distances between tripod stand: tripod stand1-tripod stand2: Distance in pixels/units."
        else:
            description =  "Analyze the drone arena image and identify all human obstacles in the scene. Count the total number of humans, specifying number of humans before the gate and number of humans after the gate. If there are no obstacles before the gate, mention that explicitly. Calculate the relative distances between each human and describe their spacing. The spacing should be based on the distance between their feet in the image. Output in this format: 'Total obstacles: [total number], Soft (human), Positions: Before gate: [number of humans before the gate], After gate: [number of humans after the gate], Spacing: [no distance], Distances between humans: human1-human2: Distance in pixels/units."


    # Process the image and text
    inputs = processor.process(
        images=[Image.open(image_path)],
        text=description
    )

    inputs = {k: v.to(model.device).unsqueeze(0) for k, v in inputs.items()}

    # Generate output; maximum 200 new tokens; stop generation when <|endoftext|> is generated
    output = model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=2000, stop_strings="<|endoftext|>"),
        tokenizer=processor.tokenizer
    )

    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return generated_text

def process_images_in_folder_crop(folder_path, num_images=2, dist="small"):
    """
    Process the images in the given folder and generate coordinates.
    """
    generated_outputs = []
    # Loop through the first 'num_images' images in the folder
    # for i in range(1, num_images + 1):
    for obstacle_type in ['hard', 'soft']:
        # Construct image path and check if the file exists
        image_name = f'cropped_{obstacle_type}_obstacle.jpg'
        image_path = os.path.join(folder_path, image_name)
        
        if os.path.exists(image_path):
            print(f"Processing {image_name}...")
            # Determine if it's a hard or soft obstacle
            is_hard_obstacle = (obstacle_type == 'hard')
            generated_output = find_objects_crop(image_path, is_hard_obstacle, dist)
            generated_outputs.append((image_name, generated_output))
        else:
            print(f"Image {image_name} not found.")
    
    return generated_outputs


def process_images_in_folder(folder_path_1, folder_path_2, num_images=10):
    """
    Process the images in the given folder and generate coordinates.
    """
    generated_outputs = []
    distances = []

    # Loop through the first 'num_images' images in the folder
    # for i in range(1, num_images + 1):
    for obstacle_type in ['hard', 'soft']:
        # Construct image path and check if the file exists
        image_name = f'{obstacle_type}_obstacle.jpg'
        image_path = os.path.join(folder_path_1, image_name)
        
        if os.path.exists(image_path):
            print(f"Processing {image_name}...")
            # Determine if it's a hard or soft obstacle
            is_hard_obstacle = (obstacle_type == 'hard')
            coordinates = find_objects(image_path, is_hard_obstacle)

            if coordinates:
                image_distances = compute_distances_for_image(coordinates)
                print(image_distances)

                # Calculate average distance
                avg_distance = sum(image_distances) / len(image_distances) if image_distances else 0

                # Now check the average distance against your thresholds
                if avg_distance > 1 and avg_distance < 20:
                    image_name = f'cropped_{obstacle_type}_obstacle.jpg'
                    image_path = os.path.join(folder_path_2, image_name)
                    
                    if os.path.exists(image_path):
                        print(f"Processing {image_name}...")
                        # Determine if it's a hard or soft obstacle
                        is_hard_obstacle = (obstacle_type == 'hard')
                        generated_output = find_objects_crop(image_path, is_hard_obstacle, dist="close")
                    else:
                        print(f"Image {image_name} not found.")
                elif avg_distance > 20:
                    image_name = f'cropped_{obstacle_type}_obstacle.jpg'
                    image_path = os.path.join(folder_path_2, image_name)
                    
                    if os.path.exists(image_path):
                        print(f"Processing {image_name}...")
                        # Determine if it's a hard or soft obstacle
                        is_hard_obstacle = (obstacle_type == 'hard')
                        generated_output = find_objects_crop(image_path, is_hard_obstacle, dist="far")
                    else:
                        print(f"Image {image_name} not found.")
                else:
                    image_name = f'cropped_{obstacle_type}_obstacle.jpg'
                    image_path = os.path.join(folder_path_2, image_name)
                    
                    if os.path.exists(image_path):
                        print(f"Processing {image_name}...")
                        # Determine if it's a hard or soft obstacle
                        is_hard_obstacle = (obstacle_type == 'hard')
                        generated_output = find_objects_crop(image_path, is_hard_obstacle, dist="nil")
                    else:
                        print(f"Image {image_name} not found.")

                generated_outputs.append((image_name, generated_output))
        else:
            print(f"Image {image_name} not found.")

    return generated_outputs


# Folder path where the images are stored
folder_path_1 = 'dataset_images'  # Adjust the path as needed
folder_path_2 = 'cropped_images'  # Adjust the path as needed

# Make sure the destination folder exists, create it if not
os.makedirs(folder_path_2, exist_ok=True)

# Define the DPI and crop dimensions (as per the original code)
dpi = 300
cm_to_inch = 2.54
crop_in_inches = 2.5 / cm_to_inch  # 1.1 cm in inches
crop_in_pixels = int(dpi * crop_in_inches)

# Define the lengths to crop from left and right sides (in pixels)
crop_left_pixels = crop_in_pixels   # Example value for cropping from the left
crop_right_pixels = crop_in_pixels  # Example value for cropping from the right

for filename in os.listdir(folder_path_1):
    if filename.endswith('.jpg'):
        # Construct full file path
        file_path = os.path.join(folder_path_1, filename)
        
        # Load the image
        image = Image.open(file_path)

        # Get image dimensions
        width, height = image.size

        # Crop the image: (left, upper, right, lower)
        cropped_image = image.crop((crop_left_pixels, 0, width - crop_right_pixels, height))

        # Save the cropped image in the destination folder
        cropped_filename = os.path.join(folder_path_2, f'cropped_{filename}')
        cropped_image.save(cropped_filename)

print("Image cropping completed!")

# Process the first 10 images in the folder
generated_outputs = process_images_in_folder(folder_path_1, folder_path_2, num_images=10)

# Print the results for each processed image
with open("output.txt",'w') as fid:
    for image_name, output in generated_outputs:
        print(f"Generated Output for {image_name}:\n{output}\n")
        fid.write(output + '\n')
        print('----------------------------------------------------')


        impedance_params, scenario, matched_text, score = rag_retrieve_impedance(output, index, id_to_payload)
        print(f"Best Matching Scenario: {scenario}")
        print(f"Impedance Parameters (Best Match): {impedance_params}")
        #print(f"Impedance Parameter Values: {impedance_params.values()}")
        print("-" * 50)
                

        # Writing to a CSV file
        with open("impedance_parameters.csv", mode="w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            
            # Writing the header (keys of the dictionary)
            writer.writerow(["Parameter", "Value"])
            
            # Writing the impedance parameters as key-value pairs
            for key, value in impedance_params.items():
                writer.writerow([key, value])

        print("Impedance parameters saved to 'impedance_parameters.csv'.")