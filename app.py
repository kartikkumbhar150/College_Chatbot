from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import faiss, pickle, os
from sentence_transformers import SentenceTransformer
from groq import Groq 
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load FAISS index + text chunks
index = faiss.read_index("college_index.faiss")
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

# Embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# OpenAI client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)
CORS(app)

def retrieve_context(query, top_k=3):
    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    D, I = index.search(q_emb, top_k)
    return " ".join([chunks[i] for i in I[0] if i < len(chunks)])

@app.route("/")
def home():
    return render_template("index.html")   

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_msg = data.get("message", "")
    lang = data.get("lang", "en")

    if not user_msg.strip():
        return jsonify({"answer": "⚠️ Please enter a valid question."})

    context = retrieve_context(user_msg)

    prompt = f"""
    You are a helpful assistant for Dr. D. Y. Patil Institute of Technology.
    Always mention the college name in answers.
    Format rules:
    - Use markdown headings (###, ####) for sections
    - Use numbered lists for steps
    - Use bullet points (-) for sub-points
    - Keep sentences concise and structured
    Context: {context}
    Question: {user_msg}
    """

    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=8000
    )

    answer = response.choices[0].message.content.strip()
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
