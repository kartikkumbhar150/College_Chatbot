from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
import os, uuid, faiss, pickle, sqlite3
from collections import defaultdict, deque
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# ============ Setup ============
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)

# Secrets / model config
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey_change_me")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

# Groq client
client = Groq(api_key=GROQ_API_KEY)

# ============ Load FAISS + chunks ============
INDEX_PATH = "college_index.faiss"
CHUNKS_PATH = "chunks.pkl"

index = faiss.read_index(INDEX_PATH)
with open(CHUNKS_PATH, "rb") as f:
    chunks = pickle.load(f)

# Embedding model must match what you used to build FAISS
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============ SQLite Persistence ============
DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS history (
            cid TEXT,
            role TEXT,
            content TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def save_message(cid, role, content):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("INSERT INTO history (cid, role, content) VALUES (?, ?, ?)", (cid, role, content))
    conn.commit()
    conn.close()

def load_history(cid, limit=20):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content FROM history WHERE cid = ? ORDER BY rowid DESC LIMIT ?", (cid, limit*2))
    rows = c.fetchall()
    conn.close()
    return [{"role": r, "content": c} for r, c in reversed(rows)]  # reverse to get correct order

def clear_history(cid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("DELETE FROM history WHERE cid = ?", (cid,))
    conn.commit()
    conn.close()

# ============ Retrieval ============
def retrieve_context(query, history_text="", top_k=3):
    q = (history_text + " " + query).strip()
    q_emb = embedder.encode([q], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    selected = []
    for i in I[0]:
        if 0 <= i < len(chunks):
            selected.append(chunks[i])
    return "\n".join(selected)

def recent_history_text(history, turns=3):
    hist = history[-2*turns:]
    return " ".join([f"{m['role'].capitalize()}: {m['content']}" for m in hist])

def get_or_create_cid():
    cid = request.cookies.get("cid")
    if not cid:
        cid = str(uuid.uuid4())
    return cid

# ============ Routes ============
@app.route("/")
def home():
    resp = make_response(render_template("index.html"))
    cid = get_or_create_cid()
    if not request.cookies.get("cid"):
        resp.set_cookie("cid", cid, httponly=True, secure=True, samesite="Lax")
    return resp

@app.route("/chat", methods=["POST"])
def chat():
    cid = get_or_create_cid()
    data = request.json or {}
    user_msg = (data.get("message") or "").strip()

    # Block empty input
    if not user_msg:
        return jsonify({"answer": "⚠️ Please enter a valid question."})

    # Handle very short/greeting inputs gracefully
    if len(user_msg.split()) < 2:
        return jsonify({"answer": "👋 Hello! Please ask me something about Dr. D. Y. Patil Institute of Technology."})

    # Load full history (from DB)
    history = load_history(cid)

    # Add user turn to history
    save_message(cid, "user", user_msg)
    history.append({"role": "user", "content": user_msg})

    # Retrieval (history-aware)
    hist_text = recent_history_text(history, turns=3)
    context = retrieve_context(user_msg, hist_text, top_k=3)

    # Build LLM messages
    system_prompt = (
        "You are a helpful assistant for Dr. D. Y. Patil Institute of Technology.\n"
        "Always mention the college name in answers.\n"
        "Answer in clean Markdown with headings, numbered steps, and bullet points.\n"
        "Use prior chat turns if the user refers to people/things mentioned earlier.\n"
        "Prefer retrieved context over memory if they conflict.\n"
        "If the user asks about something unrelated to the college, politely decline."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)  # keep all turns (up to 20)
    messages.append({
        "role": "user",
        "content": f"Relevant context:\n{context}\n\nCurrent question: {user_msg}"
    })

    # Call LLM
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=2000,
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()
    except Exception as e:
        answer = f"Sorry, I couldn't generate a response right now. ({e})"

    # Save assistant turn
    save_message(cid, "assistant", answer)

    # Return response
    resp = make_response(jsonify({"answer": answer}))
    if not request.cookies.get("cid"):
        resp.set_cookie("cid", cid, httponly=True, secure=True, samesite="Lax")
    return resp

@app.route("/reset", methods=["POST"])
def reset():
    cid = get_or_create_cid()
    clear_history(cid)
    return jsonify({"message": "Chat history cleared."})

@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    app.run(debug=True)
