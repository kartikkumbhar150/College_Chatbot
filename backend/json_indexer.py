import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os

DATA_PATH = "data/qa.json"
INDEX_PATH = "data/json_faiss_index.bin"
META_PATH = "data/json_faiss_meta.pkl"

def build_json_index():
    # Load Q&A JSON
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        qa_data = json.load(f)

    questions = [q["question"] for q in qa_data]
    answers = [q["answer"] for q in qa_data]

    # Embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(questions, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index + metadata
    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"questions": questions, "answers": answers}, f)

    print("JSON FAISS index built successfully")

if __name__ == "__main__":
    build_json_index()
