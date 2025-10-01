#!/usr/bin/env python3
"""
kb_query.py — Query one or more KB shards built by kb_build.py.

- Supports single --index-dir or multiple --index-dirs (shards)
- Engine: FAISS (default, fast) or NumPy fallback (loads embeddings.npy)
- Merges top-k across shards and prints concise results
- Returns JSON with --json for programmatic use

Examples
--------
python kb_query.py --index-dir kb_out --query "What is the main phenomenological advantage of heavy neutrinos?"

# Query multiple shards
python kb_query.py --index-dirs kb_store/2025-08-22_TopBSM kb_store/2025-08-24_2HDM \
  --query "2HDM alignment limit" --k 8
"""
import os, json, argparse, heapq
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Keep tokenizers quiet in forked contexts
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

def load_chunks(chunks_path: Path) -> List[Dict[str, Any]]:
    items = []
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except Exception:
                pass
    return items

def load_shard(index_dir: Path, engine: str):
    manifest = json.loads(Path(index_dir, "manifest.json").read_text(encoding="utf-8"))
    chunks = load_chunks(Path(index_dir, "chunks.jsonl"))
    model_name = manifest["model"]
    normalize = bool(manifest.get("normalize", True))
    emb_dim = int(manifest["emb_dim"])

    if engine == "faiss":
        faiss_index = faiss.read_index(str(Path(index_dir, "faiss.index")))
        embs = None
    else:
        faiss_index = None
        embs = np.load(Path(index_dir, "embeddings.npy"), mmap_mode="r")

    return {
        "dir": str(index_dir),
        "manifest": manifest,
        "chunks": chunks,
        "faiss": faiss_index,
        "embs": embs,
        "normalize": normalize,
        "emb_dim": emb_dim,
        "model": model_name,
    }

def search_shard(shard, qvec: np.ndarray, topk: int) -> List[Tuple[float, int, Dict[str, Any]]]:
    # returns list of (score, global_id, chunk_obj) where global_id is row in this shard
    scores, ids = None, None
    if shard["faiss"] is not None:
        D, I = shard["faiss"].search(qvec, topk)
        scores, ids = D[0], I[0]
    else:
        # NumPy cosine/IP on normalized vectors
        V = np.asarray(shard["embs"])
        # qvec is normalized row vector (1,D), V is (N,D)
        scores = (V @ qvec[0]).astype(np.float32)
        ids = np.argsort(-scores)[:topk]
        scores = scores[ids]

    out = []
    for s, idx in zip(scores, ids):
        if idx < 0 or idx >= len(shard["chunks"]):
            continue
        out.append((float(s), int(idx), shard["chunks"][idx]))
    return out

def main():
    ap = argparse.ArgumentParser(description="Query one or more KB indices.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--index-dir", help="Single index directory")
    g.add_argument("--index-dirs", nargs="+", help="Multiple index directories (shards)")

    ap.add_argument("--query", required=True, help="Query string")
    ap.add_argument("--k", type=int, default=5, help="Top-k merged results")
    ap.add_argument("--engine", choices=["auto", "faiss", "numpy"], default="auto",
                    help="Search engine backend")
    ap.add_argument("--json", action="store_true", help="Print raw JSON results")
    ap.add_argument("--snippet", type=int, default=240, help="Characters to show from each chunk")
    args = ap.parse_args()

    shard_dirs = [Path(args.index_dir)] if args.index_dir else [Path(p) for p in args.index_dirs]

    # Load first manifest to pick model and check consistency
    first_manifest = json.loads(Path(shard_dirs[0], "manifest.json").read_text(encoding="utf-8"))
    model_name = first_manifest["model"]
    normalize = bool(first_manifest.get("normalize", True))
    emb_dim = int(first_manifest["emb_dim"])

    # Engine selection
    engine = args.engine
    if engine == "auto":
        # prefer FAISS if all shards have faiss.index
        ok = all(Path(sd, "faiss.index").exists() for sd in shard_dirs)
        engine = "faiss" if ok else "numpy"

    # Load shards
    shards = [load_shard(sd, engine) for sd in shard_dirs]
    # Basic compatibility check
    for sh in shards:
        if sh["model"] != model_name or sh["emb_dim"] != emb_dim or bool(sh["normalize"]) != bool(normalize):
            raise RuntimeError(f"Incompatible shard {sh['dir']} (model/dim/normalize differ)")

    # Encoder
    encoder = SentenceTransformer(model_name)
    qvec = encoder.encode([args.query], normalize_embeddings=normalize, convert_to_numpy=True).astype(np.float32)

    # Collect candidates from each shard and merge top-k
    candidates = []
    per_shard_k = max(args.k, 10)  # grab a few more per shard, then merge
    for shard in shards:
        res = search_shard(shard, qvec, per_shard_k)
        for (score, ridx, chunk) in res:
            candidates.append((score, shard, ridx, chunk))

    # Merge (max-heap by score)
    top = heapq.nlargest(args.k, candidates, key=lambda t: t[0])

    results = []
    for score, shard, ridx, chunk in top:
        text = chunk.get("text", "")
        results.append({
            "score": float(score),
            "doc_id": chunk.get("doc_id"),
            "source": chunk.get("source"),
            "page": chunk.get("page"),
            "index_dir": shard["dir"],
            "snippet": (text[:args.snippet] + ("…" if len(text) > args.snippet else "")),
            "hash": chunk.get("hash"),
        })

    if args.json:
        print(json.dumps({"query": args.query, "topk": args.k, "results": results}, ensure_ascii=False, indent=2))
    else:
        print(f"\nQuery: {args.query}\nModel : {model_name}  (engine={engine})\n")
        for i, r in enumerate(results, 1):
            print(f"[{i}] score={r['score']:.4f}  doc={r['doc_id']}  page={r['page']}  dir={Path(r['index_dir']).name}")
            print(f"    source: {r['source']}")
            print(f"    {r['snippet']}\n")

if __name__ == "__main__":
    main()
