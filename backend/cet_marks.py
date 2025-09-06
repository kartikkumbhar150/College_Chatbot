import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load input JSON
json_file = "data/mht_cet_cutoff.json"
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# Prepare documents
documents = []
for record in data:
    text = (
        f"Branch: {record['Branch']}, "
        f"Category Level: {record['Category Level']}, "
        f"Category: {record['Category']}, "
        f"Cutoff Rank: {record['Cutoff Rank']}, "
        f"Cutoff Percentile: {record['Cutoff Percentile']}"
    )
    documents.append(text)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
embeddings = model.encode(documents, convert_to_numpy=True).astype("float32")

# Normalize embeddings for cosine similarity
faiss.normalize_L2(embeddings)

# Build FAISS index (cosine similarity)
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings)

print(f"✅ FAISS index created with {index.ntotal} embeddings")

# Save index + docs
faiss.write_index(index, "data/cutoff_index.faiss")
with open("data/cutoff_documents.json", "w", encoding="utf-8") as f:
    json.dump(documents, f, indent=2, ensure_ascii=False)

print("Files generated:")
print(" - cutoff_index.faiss (FAISS vector index)")
print(" - cutoff_documents.json (text data)")

# -------- TEST SEARCH --------
index_loaded = faiss.read_index("data/cutoff_index.faiss")

with open("data/cutoff_documents.json", "r", encoding="utf-8") as f:
    documents_loaded = json.load(f)

query = "Computer Engineering cutoff for OBC"
query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
faiss.normalize_L2(query_embedding)

k = 3
scores, indices = index_loaded.search(query_embedding, k)

print("\n🔎 Query:", query)
for i, idx in enumerate(indices[0]):
    print(f"{i+1}. {documents_loaded[idx]} (score={scores[0][i]:.4f})")
