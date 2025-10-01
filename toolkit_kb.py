#!/usr/bin/env python3
# toolkit_kb.py
"""
Runtime KB retrieval for your agent.

- Loads the artifacts produced by kb_build.py (chunks.jsonl, embeddings.npy,
  manifest.json, and optionally faiss.index).
- Provides a simple function `kb_search(...)` that returns structured hits
  (rank, score, doc_id, source, chunk_id, text).
- No subprocess; pure Python. Fast to call from your agent.
"""

from __future__ import annotations
import json, pathlib, re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Optional FAISS
try:
    import faiss  # type: ignore
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False


def _load_manifest(d: pathlib.Path) -> Dict[str, Any]:
    with open(d / "manifest.json", "r") as fh:
        return json.load(fh)

def _load_chunks(d: pathlib.Path) -> List[Dict[str, Any]]:
    out = []
    with open(d / "chunks.jsonl", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out

def _load_embeddings(d: pathlib.Path) -> np.ndarray:
    return np.load(d / "embeddings.npy")

def _load_faiss(d: pathlib.Path):
    return faiss.read_index(str(d / "faiss.index"))

def _ensure_model(name: str, device: str = "cpu") -> SentenceTransformer:
    # Use CPU by default to keep the agent lightweight/stable
    return SentenceTransformer(name, device=device)

def _cosine_search_numpy(q_vec: np.ndarray, db: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # Assumes L2-normalized embeddings
    scores = db @ q_vec  # [N]
    if k >= len(scores):
        idx = np.argsort(-scores)
        return idx, scores[idx]
    part = np.argpartition(-scores, k-1)[:k]
    order = part[np.argsort(-scores[part])]
    return order, scores[order]

def kb_search(
    index_dir: str,
    query: str,
    k: int = 5,
    engine: str = "auto",          # "auto" | "faiss" | "numpy"
    doc_id: Optional[str] = None,  # restrict to a single document id
    model_name: Optional[str] = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Search the local KB and return top-k chunks.

    Returns:
      {
        "engine": "faiss" | "numpy",
        "k": int,
        "count_chunks": int,
        "hits": [
          {"rank": int, "score": float, "doc_id": str, "source": str,
           "chunk_id": int, "text": str}
        ]
      }
    """
    idx_dir = pathlib.Path(index_dir)
    if not idx_dir.exists():
        raise FileNotFoundError(f"Index dir not found: {idx_dir}")

    manifest = _load_manifest(idx_dir)
    chunks = _load_chunks(idx_dir)
    embeddings = _load_embeddings(idx_dir).astype(np.float32)
    if embeddings.shape[0] != len(chunks):
        raise RuntimeError("embeddings/chunks length mismatch")

    model_name = model_name or manifest.get("embedding_model") or "sentence-transformers/all-MiniLM-L6-v2"
    model = _ensure_model(model_name, device=device)

    q = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    qv = q[0]

    engine_used = "numpy"
    index = None
    if engine in ("auto", "faiss"):
        if HAVE_FAISS and (idx_dir / "faiss.index").exists():
            try:
                index = _load_faiss(idx_dir)
                engine_used = "faiss"
            except Exception:
                if engine == "faiss":
                    raise

    # Over-sample if filtering
    k_eff = k * 5 if doc_id else k

    if engine_used == "faiss":
        D, I = index.search(q, k_eff)
        idxs = I[0]
        sims = D[0]
    else:
        idxs, sims = _cosine_search_numpy(qv, embeddings, k_eff)

    hits = []
    seen = 0
    for rank0, (i, score) in enumerate(zip(idxs, sims), start=1):
        if i < 0 or i >= len(chunks):
            continue
        rec = dict(chunks[i])
        if doc_id and rec.get("doc_id") != doc_id:
            continue
        hits.append({
            "rank": len(hits) + 1,
            "score": float(score),
            "doc_id": rec.get("doc_id"),
            "source": rec.get("source"),
            "chunk_id": rec.get("chunk_id"),
            "text": rec.get("text"),
        })
        if len(hits) >= k:
            break

    return {
        "engine": engine_used,
        "k": k,
        "count_chunks": len(chunks),
        "hits": hits,
    }
