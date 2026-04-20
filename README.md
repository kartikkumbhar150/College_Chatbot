# 🎓 College Chatbot (Voice + Text RAG Assistant)

A full-stack college assistant for **Dr. D. Y. Patil Institute of Technology** that supports:

- **Voice and text chat** in the browser.
- **Multilingual input flow** (English/Hindi/Marathi speech capture with translation to English before backend query).
- **Hybrid answer strategy**:
  1. **Cutoff-specific semantic search** (MHT-CET data).
  2. **Direct semantic Q&A lookup** (`qa.json`).
  3. **General RAG fallback** (FAISS retrieval + Groq LLM).

The backend is Flask-based, retrieval uses FAISS + sentence-transformers, and the frontend is a lightweight static web UI.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [How the Answering Pipeline Works](#how-the-answering-pipeline-works)
4. [Prerequisites](#prerequisites)
5. [Installation & Setup](#installation--setup)
6. [Environment Variables](#environment-variables)
7. [Data Preparation](#data-preparation)
8. [Build Indexes](#build-indexes)
9. [Run the Application](#run-the-application)
10. [API Reference](#api-reference)
11. [Frontend Usage Guide](#frontend-usage-guide)
12. [Operational Notes](#operational-notes)
13. [Troubleshooting](#troubleshooting)
14. [Security & Production Recommendations](#security--production-recommendations)
15. [Quick Start (Minimal)](#quick-start-minimal)

---

## Architecture Overview

```text
[Browser UI]
  - Speech Recognition (Web Speech API)
  - Optional translation bridge (LibreTranslate)
  - Markdown rendering for answers
        |
        v
[Flask Backend: /api/query]
  1) Cutoff FAISS search (if cutoff/rank/CET/marks intent)
  2) Semantic QA JSON lookup
  3) General FAISS retrieval + Groq generation
        |
        +--> [FAISS general index + metadata]
        +--> [Cutoff FAISS index + cutoff documents]
        +--> [qa.json loaded in memory + in-memory FAISS]
```

---

## Repository Structure

```text
College_Chatbot/
├── README.md
├── backend/
│   ├── app.py                     # Flask API + retrieval/generation orchestration
│   ├── scraper.py                 # Crawls college website and creates college.txt
│   ├── embeddings_indexer.py      # Builds/loads/searches general FAISS index
│   ├── json_indexer.py            # Optional script to index qa.json separately
│   ├── cet_marks.py               # Builds cutoff FAISS index from MHT-CET JSON
│   ├── groq_client.py             # Async Groq API client
│   ├── requirements.txt           # Python dependencies
│   └── data/                      # You create this folder and data artifacts here
└── frontend/
    ├── index.html                 # Chat UI skeleton
    ├── main.js                    # Voice, text input, API calls, TTS, rendering
    └── styles.css                 # UI styles
```

---

## How the Answering Pipeline Works

When the frontend sends `POST /api/query`, the backend does the following in order:

### 1) Cutoff-first routing (admission intent)
If user query contains admission keywords (e.g., `cutoff`, `rank`, `cet`, `marks`), backend searches `cutoff_index.faiss` first and returns a markdown table for the detected branch/category when available.

### 2) Semantic Q&A lookup (`qa.json`)
If no cutoff answer is returned, backend semantically matches the question against predefined Q&A using sentence embeddings and cosine similarity threshold.

### 3) General RAG fallback
If no direct Q&A hit:
- Query is embedded.
- Top chunks retrieved from the general FAISS index.
- Prompt is built with recent session history + retrieved context.
- Groq LLM generates the response.

### 4) Session memory behavior
- Session history is tracked by `session_id`.
- Up to last 10 Q/A pairs are retained per session in memory.
- `clear/reset` commands clear current session history.

---

## Prerequisites

- **Python 3.10+** (recommended)
- `pip`
- Internet access (required for:
  - downloading embedding models,
  - calling Groq API,
  - optional LibreTranslate in browser flow)
- Groq API key

Optional but recommended:
- Virtual environment (`venv`)

---

## Installation & Setup

From repository root:

```bash
cd /workspace/College_Chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

---

## Environment Variables

Create file: `backend/.env`

```env
# Required for LLM generation
GROQ_API_KEY=your_groq_api_key_here

# Optional: Groq model
GROQ_MODEL=llama-3.1-8b-instant

# Flask runtime
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
FLASK_DEBUG=false

# Embedding/indexer tuning (optional)
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
QA_EMBEDDING_MODEL=intfloat/e5-base-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
CHUNK_SIZE=300
CHUNK_OVERLAP=30
MIN_CHUNK_WORDS=20
HNSW_M=32
EF_CONSTRUCTION=200
EF_SEARCH=50
BATCH_SIZE=64
```

> Important: `GROQ_API_KEY` is mandatory for `app.py` because `groq_client.py` validates it at import time.

---

## Data Preparation

Create data directory:

```bash
mkdir -p backend/data
```

Expected files/artifacts in `backend/data/`:

### Required for app startup
- `qa.json` (list of objects with `question` and `answer` keys)
- `faiss_index.bin` and `faiss_meta.pkl` (general retrieval index + metadata)

### Optional but used if present
- `mht_cet_cutoff.json` (input source for cutoff indexing script)
- `cutoff_index.faiss` and `cutoff_documents.json` (built from script)
- `college.txt` (crawler output text source)

### `qa.json` format example

```json
[
  {
    "question": "What are college timings?",
    "answer": "College timings are ..."
  },
  {
    "question": "Where is the admissions office?",
    "answer": "The admissions office is ..."
  }
]
```

---

## Build Indexes

Run these commands **from `backend/`**.

```bash
cd backend
```

### A) Crawl website content (optional source generation)

```bash
python scraper.py
```

This produces `data/college.txt`.

### B) Build general FAISS retrieval index (required)

```bash
python embeddings_indexer.py --build
```

This generates:
- `data/faiss_index.bin`
- `data/faiss_meta.pkl`
- `data/docs_chunks.json`

### C) Build cutoff FAISS index (optional but recommended for admission queries)

```bash
python cet_marks.py
```

Requires `data/mht_cet_cutoff.json` and produces:
- `data/cutoff_index.faiss`
- `data/cutoff_documents.json`

### D) (Optional) standalone Q&A index script

```bash
python json_indexer.py
```

> Note: `app.py` already builds an in-memory Q&A index at startup from `qa.json`, so this script is optional for the current runtime pipeline.

---

## Run the Application

From `backend/`:

```bash
python app.py
```

Server default URL:

- `http://0.0.0.0:5000` (or `http://localhost:5000` from your machine)

Frontend is served by Flask static routing (`frontend/` folder).

---

## API Reference

### `POST /api/query`

Submit a query.

**Request JSON**

```json
{
  "q": "What is the cutoff for Computer Engineering OBC?",
  "session_id": "session-abc123"
}
```

**Response JSON (success)**

```json
{
  "answer": "...",
  "retrieved": [
    {
      "id": 12,
      "text": "...",
      "score": 0.83
    }
  ],
  "history": [
    {"q": "...", "a": "..."}
  ]
}
```

**Special command queries**

- Stop responses: `stop`, `exit`, `ok stop`, `wait`
- Clear history: `clear`, `clear history`, `reset`

### `GET /api/history?session_id=...`
Returns in-memory Q/A history list for session.

### `GET /api/health`
Returns backend health summary:
- status
- whether FAISS loaded
- whether cutoff index loaded
- count of Q&A rows loaded from `qa.json`

---

## Frontend Usage Guide

- Open app in browser.
- Click **Start Listening** or type in input box.
- Voice wake words include variants like `hello`, `hi`, `hello dit`, etc.
- You can switch speech recognition language from dropdown:
  - English (India)
  - Hindi
  - Marathi
- Bot output supports markdown rendering.

### Voice flow highlights

- Continuous recognition with interim transcript bubble.
- Interrupt command support (`stop`, `wait`, etc.).
- Text-to-speech reads bot answers.
- If user starts speaking while bot speaks, bot speech is canceled.

---

## Operational Notes

1. **In-memory session store**
   - Session history is not persisted across restarts.
2. **Model startup overhead**
   - First startup can take time due to model loading.
3. **Embedding dimension consistency is mandatory**
   - If you switch embedding model, rebuild FAISS index.
4. **Cutoff behavior**
   - If cutoff index files are absent, app still runs and skips cutoff search.
5. **CORS**
   - Configured as `*` in current backend.

---

## Troubleshooting

### Error: `Q&A file not found: backend/data/qa.json`
Create `backend/data/qa.json` with valid Q/A entries.

### Error: `Index or metadata not found. Run with --build first.`
Run:

```bash
cd backend
python embeddings_indexer.py --build
```

### Error: `Please set GROQ_API_KEY in .env`
Set `GROQ_API_KEY` in `backend/.env`.

### Error: dimension mismatch between model and index
You changed `EMBEDDING_MODEL` after index creation. Rebuild:

```bash
python embeddings_indexer.py --build
```

### Browser speech recognition not working
- Use Chromium-based browser.
- Ensure microphone permission is granted.
- Some browsers/devices may not fully support `SpeechRecognition`.

### Translation service unavailable
Frontend translation calls an external LibreTranslate endpoint. If unavailable, flow falls back to original text.

---

## Security & Production Recommendations

Current project is development-friendly. For production, consider:

- Restrict CORS to trusted origins.
- Put Flask behind production server (gunicorn/uvicorn + reverse proxy).
- Add request auth/rate limiting.
- Add observability (structured logs, metrics, tracing).
- Persist chat histories in DB if needed.
- Containerize with pinned model versions for reproducibility.

---

## Quick Start (Minimal)

If you just want to run quickly:

```bash
cd /workspace/College_Chatbot
python -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
mkdir -p backend/data
# add backend/data/qa.json
cd backend
python embeddings_indexer.py --build
python app.py
```

Then open `http://localhost:5000`.

---

## Maintainer Notes

- If you add new data files in `backend/data`, rebuild indexes accordingly.
- Keep prompts in `app.py` concise and policy-aligned.
- Prefer validating `api/health` before frontend debugging.
