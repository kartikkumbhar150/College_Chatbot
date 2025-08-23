from sentence_transformers import SentenceTransformer
import faiss
import pickle

# Load your scraped text
with open("college_data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i+chunk_size]))
    return chunks

chunks = chunk_text(text)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = model.encode(chunks)

# Build FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save index + chunks
faiss.write_index(index, "college_index.faiss")
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print(" Data indexed in FAISS")
