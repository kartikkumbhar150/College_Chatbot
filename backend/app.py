import os
import pickle
import faiss
import numpy as np
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import tempfile
from groq import Groq

# ✅ local imports
from embeddings_indexer import read_source_files, preprocess, chunk_text_with_sections
from groq_client import groq_generate

# ============ Setup ============
load_dotenv()

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
DATA_DIR.mkdir(exist_ok=True)

FAISS_INDEX_FILE = DATA_DIR / "faiss_index.bin"
FAISS_META_FILE = DATA_DIR / "faiss_meta.pkl"

EMBED_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app, resources={r"/*": {"origins": "*"}})

# In-memory conversation history {session_id: [{q, a}, ...]}
HISTORY = {}

# Groq client
client = Groq(api_key=GROQ_API_KEY)


# ============ JSON serialization helper ============
def to_serializable(obj):
    """Recursively convert numpy types to native Python types for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


# ============ Load FAISS Index ============
if FAISS_INDEX_FILE.exists() and FAISS_META_FILE.exists():
    try:
        faiss_index = faiss.read_index(str(FAISS_INDEX_FILE))
        with open(FAISS_META_FILE, "rb") as f:
            metadata = pickle.load(f)
        dim = faiss_index.d
        embed_model = SentenceTransformer(EMBED_MODEL_NAME)
        print(f"✅ FAISS index loaded with {len(metadata)} entries, dim={dim}")
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load FAISS index/metadata: {e}")
else:
    raise RuntimeError("❌ FAISS index or metadata not found. Run `embeddings_indexer.py` first.")


# ============ Retrieval ============
def retrieve(query: str, top_k: int = 4):
    """Retrieve top_k most relevant chunks from FAISS index."""
    q_emb = embed_model.encode(query, normalize_embeddings=True)
    q_emb = np.array([q_emb]).astype("float32")

    D, I = faiss_index.search(q_emb, top_k)
    results = []
    for idx, score in zip(I[0], D[0]):
        if idx < 0:
            continue
        meta = metadata[idx]
        results.append({
            "id": meta.get("id", int(idx)),
            "text": meta["text"],
            "score": float(score)
        })
    return results


# ============ Prompt Builder ============
def build_prompt(question, retrieved_docs, history):
    system = (
        "You are an expert assistant for Dr. D. Y. Patil Institute of Technology.\n\n"
        "Formatting Rules:\n"
        "- Use Markdown formatting.\n"
        "- Headings must use ## (example: ## Placement Records).\n"
        "- Subheadings use ### (example: ### Training and Placement Team).\n"
        "- Use bullet points with - (hyphen).\n"
        "- Use numbered lists with 1., 2., 3.\n"
        "- Keep one blank line between sections.\n"
        "- End with a **Summary** section.\n\n"
        "Content Rules:\n"
        "- Always explicitly write 'Dr. D. Y. Patil Institute of Technology'.\n"
        "- Never replace the college name with 'the college' or 'the institute'.\n"
        "- Integrate prior conversation naturally if referenced.\n"
        "- If the question is unrelated to the college, politely decline.\n"
    )

    hist_text = ""
    if history:
        for h in history[-5:]:
            hist_text += f"Q: {h.get('q')}\nA: {h.get('a')}\n"

    sources_text = "\n\n---\n\n".join([d['text'] for d in retrieved_docs])

    user_prompt = (
        f"{hist_text}\n\n"
        f"Question: {question}\n\n"
        f"Context documents:\n{sources_text}\n\n"
        "Answer in the exact required structure, using Markdown formatting."
    )
    return system, user_prompt


# ============ API Routes ============
@app.route("/api/query", methods=["POST"])
def api_query():
    data = request.json or {}
    q = data.get("q", "").strip()
    session_id = data.get("session_id", "default")

    if not q:
        return jsonify({"error": "No query provided"}), 400

    # 🔹 Special commands
    q_lower = q.lower()
    if q_lower in ("stop", "exit", "okay stop", "ok stop", "wait"):
        return jsonify({
            "answer": "[stopped]",
            "retrieved": [],
            "history": HISTORY.get(session_id, [])
        })

    if q_lower in ("clear", "clear history", "reset"):
        HISTORY[session_id] = []
        return jsonify({
            "answer": "✅ History cleared.",
            "retrieved": [],
            "history": []
        })

    # 🔹 Retrieval
    try:
        retrieved = retrieve(q, top_k=6)
    except Exception as e:
        return jsonify({"error": f"Retrieval failed: {str(e)}"}), 500

    hist = HISTORY.get(session_id, [])
    system, user_prompt = build_prompt(q, retrieved, hist)

    try:
        answer = groq_generate(system, user_prompt, max_tokens=800, temperature=0.1)
    except Exception as e:
        return jsonify({"error": f"Groq API error: {str(e)}"}), 500

    # 🔹 Update history
    hist.append({"q": q, "a": answer})
    HISTORY[session_id] = hist[-10:]

    return jsonify({
        "answer": answer,
        "retrieved": retrieved,
        "history": HISTORY[session_id]
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    session_id = request.args.get("session_id", "default")
    return jsonify(to_serializable(HISTORY.get(session_id, [])))


@app.route("/api/scrape", methods=["POST"])
def api_scrape():
    try:
        from scraper import crawl
        text = crawl()
        out_file = DATA_DIR / "college.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(text)
        return jsonify({"status": "scraped", "chars": len(text)})
    except Exception as e:
        return jsonify({"error": f"Scraping failed: {str(e)}"}), 500


@app.route("/")
def frontend_index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(app.static_folder, path)


# ============ Main ============
if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "0.0.0.0")
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    print(f"🚀 Running Flask server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug)
