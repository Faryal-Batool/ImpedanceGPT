
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import csv


# Initialize model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index_file = "scenarios_index.index"
scenarios_file = "output_latest.txt"
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

# Load VLM responses from text file
def load_responses(file_path=scenarios_file):
    with open(file_path, "r") as f:
        responses = [line.strip() for line in f if line.strip()]
    return responses

# Test the RAG system with responses
if __name__ == "__main__":
 
    # Load responses
    vlm_responses = load_responses(scenarios_file)
    print(f"Loaded {len(vlm_responses)} VLM responses:\n")

    # Process each response
    for i, response in enumerate(vlm_responses, 1):
        print(f"Response {i}: '{response}'")
        impedance_params, scenario, matched_text, score = rag_retrieve_impedance(response, index, id_to_payload)
        #print(f"Best Matching Scenario: {scenario}")
        print(f"Impedance Parameters (Best Match): {impedance_params}")
        print(f"Impedance Parameter Values: {impedance_params.values()}")
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


