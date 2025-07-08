import json
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
json_file = "scenarios_description.json"

# Load the knowledge base
with open(json_file, "r") as f:
   database = json.load(f)


# Embed all scenario texts
embeddings = np.array([embedding_model.encode(entry["text"]) for entry in database], dtype=np.float32)

# Create FAISS index (FlatL2 for exact search)
dimension = 384  # all-MiniLM-L6-v2 output size
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)  # Add all 20 embeddings

# Map indices to JSON entries
id_to_payload = {i: entry for i, entry in enumerate(database)}

# Save FAISS index to a file
faiss.write_index(index, "scenarios_index.index")

# Save the mapping of indices to JSON entries (for later retrieval)
with open("id_to_payload.json", "w") as f:
    json.dump({i: entry for i, entry in enumerate(database)}, f)

print(f"Embedded {index.ntotal} scenarios into FAISS.")




