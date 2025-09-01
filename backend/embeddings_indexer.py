import os
import glob
import json
import pickle
import re
import argparse
import hashlib
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

import numpy as np
from tqdm import tqdm
import faiss

# ---- Load ENV ----
load_dotenv()

# ---- Paths / defaults ----
BASE = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TEXT_FILE = os.path.join(DATA_DIR, "college.txt")
CHUNKS_FILE = os.path.join(DATA_DIR, "docs_chunks.json")
FAISS_INDEX_FILE = os.path.join(DATA_DIR, "faiss_index.bin")
FAISS_META_FILE = os.path.join(DATA_DIR, "faiss_meta.pkl")

# ---- Models & hyperparams ----
EMBED_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
QA_EMBED_MODEL_NAME = os.getenv(
    "QA_EMBEDDING_MODEL",
    "sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "30"))
MIN_CHUNK_WORDS = int(os.getenv("MIN_CHUNK_WORDS", "20"))

HNSW_M = int(os.getenv("HNSW_M", "32"))
EF_CONSTRUCTION = int(os.getenv("EF_CONSTRUCTION", "200"))
EF_SEARCH = int(os.getenv("EF_SEARCH", "50"))

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))


# ---- Utilities ----
def read_source_files() -> List[Dict[str, Any]]:
    if os.path.exists(TEXT_FILE):
        files = [TEXT_FILE]
    else:
        files = []
        for pattern in ("*.txt", "*.md", "*.html"):
            files.extend(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {DATA_DIR}. Place college.txt or other files there.")
    result = []
    for fp in sorted(files):
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            result.append({"source": os.path.basename(fp), "text": f.read()})
    return result


def preprocess(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r'\n{3,}', '\n\n', text)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return "\n\n".join(paragraphs)


def is_heading(paragraph: str) -> bool:
    p = paragraph.strip()
    if len(p) < 120 and (p.endswith(":") or p.isupper() or re.match(r'^[\d\.\sA-Z\-]{1,60}$', p)):
        return True
    return False


def chunk_text_with_sections(text: str,
                             chunk_size: int = CHUNK_SIZE,
                             overlap: int = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    paragraphs = text.split("\n\n")
    chunks, cur_words, cur_len, last_heading = [], [], 0, None

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if is_heading(para):
            last_heading = para
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sent in sentences:
            words = sent.split()
            if cur_len + len(words) > chunk_size and cur_words:
                chunk_text = " ".join(cur_words).strip()
                if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
                    chunks.append({"text": chunk_text, "section": last_heading})
                if overlap > 0:
                    overlap_words = " ".join(cur_words).split()[-overlap:]
                    cur_words, cur_len = overlap_words.copy(), len(overlap_words)
                else:
                    cur_words, cur_len = [], 0
            cur_words.extend(words)
            cur_len += len(words)

    if cur_words:
        chunk_text = " ".join(cur_words).strip()
        if len(chunk_text.split()) >= MIN_CHUNK_WORDS:
            chunks.append({"text": chunk_text, "section": last_heading})

    seen, unique_chunks = set(), []
    for c in chunks:
        h = hashlib.sha1(c["text"].strip().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_chunks.append(c)
    return unique_chunks


# ---- Index build / load / search ----
def build_index(embedding_model_name: str = EMBED_MODEL_NAME,
                use_qa_model: bool = False):
    print("Reading source files...")
    sources = read_source_files()

    print("Preprocessing and chunking...")
    all_chunks = []
    for s in sources:
        text = preprocess(s["text"])
        for ch in chunk_text_with_sections(text, CHUNK_SIZE, CHUNK_OVERLAP):
            all_chunks.append({"source": s["source"], "section": ch.get("section"), "text": ch["text"]})
    if not all_chunks:
        raise RuntimeError("No chunks produced. Check your data and chunking settings.")

    print(f"Total chunks: {len(all_chunks)}")

    model_name = QA_EMBED_MODEL_NAME if use_qa_model else embedding_model_name
    print(f"Loading embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)
    print(f"Embedding dim: {embedder.get_sentence_embedding_dimension()}")

    embeddings = []
    texts = [c["text"] for c in all_chunks]
    for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Encoding"):
        embs = embedder.encode(texts[i:i+BATCH_SIZE],
                               show_progress_bar=False,
                               convert_to_numpy=True,
                               batch_size=BATCH_SIZE)
        embeddings.append(embs)
    arr = np.vstack(embeddings).astype("float32")
    faiss.normalize_L2(arr)

    dim = arr.shape[1]
    print(f"Creating FAISS index (dim={dim}, M={HNSW_M})")
    index = faiss.IndexHNSWFlat(dim, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = EF_CONSTRUCTION
    index.hnsw.efSearch = EF_SEARCH
    index_id_map = faiss.IndexIDMap(index)

    ids = np.arange(len(all_chunks)).astype("int64")
    index_id_map.add_with_ids(arr, ids)

    print("Saving index + metadata...")
    faiss.write_index(index_id_map, FAISS_INDEX_FILE)
    with open(FAISS_META_FILE, "wb") as f:
        pickle.dump(all_chunks, f)
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print("Index build complete.")


def load_index_and_meta(embed_model_name: str = EMBED_MODEL_NAME,
                        index_path: str = FAISS_INDEX_FILE,
                        meta_path: str = FAISS_META_FILE):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError("Index or metadata not found. Run with --build first.")

    print("Loading FAISS index + metadata...")
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    embedder = SentenceTransformer(embed_model_name)
    model_dim = embedder.get_sentence_embedding_dimension()
    index_dim = index.d
    print(f"Model dim={model_dim}, Index dim={index_dim}")

    if model_dim != index_dim:
        raise ValueError(
            f"Dimension mismatch! Model={model_dim}, Index={index_dim}. "
            f"Rebuild the index with this model."
        )

    return index, meta, embedder


def search(query: str,
           top_k: int = 5,
           rerank: bool = True,
           embed_model_name: str = EMBED_MODEL_NAME,
           cross_encoder_model: Optional[str] = CROSS_ENCODER_MODEL):
    index, meta, embedder = load_index_and_meta(embed_model_name)
    qv = embedder.encode(query, convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(qv)

    D, I = index.search(np.array([qv]), top_k)
    candidates = [{"id": int(idx), "score": float(sc), "meta": meta[int(idx)]}
                  for idx, sc in zip(I[0], D[0]) if idx != -1]

    if rerank and cross_encoder_model:
        try:
            cross = CrossEncoder(cross_encoder_model)
            pairs = [[query, c["meta"]["text"]] for c in candidates]
            rerank_scores = cross.predict(pairs)
            for c, new_sc in zip(candidates, rerank_scores):
                c["rerank_score"] = float(new_sc)
            candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        except Exception as e:
            print(f"Cross-encoder rerank failed: {e}")

    return candidates


# ---- CLI ----
def main():
    parser = argparse.ArgumentParser(description="FAISS chatbot indexer + search")
    parser.add_argument("--build", action="store_true", help="Build embeddings + FAISS index")
    parser.add_argument("--use_qa_model", action="store_true", help="Use QA-optimized embedding model")
    parser.add_argument("--search", type=str, help="Run a quick search query")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_rerank", action="store_true")
    args = parser.parse_args()

    if args.build:
        build_index(use_qa_model=args.use_qa_model)
        return
    if args.search:
        results = search(args.search, top_k=args.top_k, rerank=(not args.no_rerank))
        for i, r in enumerate(results, 1):
            meta = r["meta"]
            print(f"\nResult {i}: score={r.get('rerank_score', r['score']):.4f} source={meta.get('source')}")
            print(meta.get("text", "")[:300], "...")
        return
    parser.print_help()


if __name__ == "__main__":
    main()
