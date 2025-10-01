#!/usr/bin/env python3
"""
kb_build.py — Build (or append to) a local physics knowledge base.

Features
- Ingest from arXiv IDs (--ids), local PDFs (--pdfs), or plain text files (--txts)
- Chunking with overlap; skip obvious reference sections
- Page- and chunk-hash dedup to avoid runaway duplication
- Embedding with SentenceTransformers (normalized)
- FAISS IndexFlatIP + parallel embeddings.npy (for numpy fallback)
- Incremental appends with --append (no clobber)
- Atomic writes where possible; tqdm progress bars

Examples
--------
# New build from arXiv IDs
python kb_build.py --ids 1506.02624 1807.06209 --out kb_out

# Append more IDs later
python kb_build.py --ids 2403.01234 2410.05678 --out kb_out --append

# Mix inputs
python kb_build.py --pdfs local1.pdf local2.pdf --txts notes.txt --out kb_out
"""
import os, io, re, json, time, hashlib, argparse, tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Iterable, Optional

# Avoid tokenizers fork warnings / deadlocks (macOS, Jupyter, etc.)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
from tqdm import tqdm
import requests
import faiss
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader

# ----------------------------- I/O helpers -----------------------------

def atomic_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: dict):
    data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
    atomic_write(path, data)

# ----------------------------- Text utils -----------------------------

REF_RE = re.compile(r"^(references|bibliography)\s*$", re.I)

def normalize_text(s: str) -> str:
    # light normalization for hashing/dedup
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def extract_pdf_text(pdf_path: Path, max_pages: Optional[int]) -> List[Tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        if max_pages is not None and i >= max_pages:
            break
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        pages.append((i + 1, normalize_text(txt)))
    return pages

def trim_after_references(pages: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    # Stop at first page whose first nonblank line is "References/Bibliography"
    out = []
    for pno, txt in pages:
        first_nonblank = None
        for line in txt.splitlines():
            if line.strip():
                first_nonblank = line.strip()
                break
        if first_nonblank and REF_RE.match(first_nonblank):
            break
        out.append((pno, txt))
    return out

def chunk_text(text: str, max_chars: int, overlap: int) -> List[str]:
    # robust char-based sliding window to avoid token-dependent behavior
    text = normalize_text(text)
    if not text:
        return []
    chunks = []
    start = 0
    N = len(text)
    while start < N:
        end = min(start + max_chars, N)
        chunks.append(text[start:end])
        if end == N:
            break
        start = max(0, end - overlap)
    return chunks

# ----------------------------- ArXiv fetch ----------------------------

def fetch_arxiv_pdf(arxiv_id: str, tmpdir: Path) -> Path:
    # accept bare like "1506.02624" or "arXiv:1506.02624"
    arxiv_id = arxiv_id.replace("arXiv:", "").strip()
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    out = tmpdir / f"{arxiv_id}.pdf"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    out.write_bytes(r.content)
    return out

# ----------------------------- Build core -----------------------------

def ensure_index(index_dir: Path, model_name: str, emb_dim: int, normalize: bool, append: bool):
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    index_path = index_dir / "faiss.index"
    emb_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"

    if append and manifest_path.exists():
        manifest = read_json(manifest_path)
        # Sanity checks
        if manifest.get("model") != model_name:
            raise RuntimeError(f"Model mismatch: {manifest.get('model')} vs {model_name}")
        if manifest.get("emb_dim") != emb_dim:
            raise RuntimeError(f"Embedding dim mismatch: {manifest.get('emb_dim')} vs {emb_dim}")
        if bool(manifest.get("normalize", True)) != bool(normalize):
            raise RuntimeError("Normalization policy mismatch with existing index.")

        # Load FAISS
        faiss_index = faiss.read_index(str(index_path)) if index_path.exists() else faiss.IndexFlatIP(emb_dim)
        # Load existing embeddings length (or 0)
        if emb_path.exists():
            try:
                old = np.load(emb_path, mmap_mode="r")
                base = int(old.shape[0])
            except Exception:
                base = 0
        else:
            base = 0

        # Collect existing chunk hashes to dedup
        seen_hashes = set()
        if chunks_path.exists():
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line)
                        if "hash" in obj:
                            seen_hashes.add(obj["hash"])
                    except Exception:
                        pass

        return faiss_index, manifest, base, seen_hashes

    # Fresh build
    manifest = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "emb_dim": emb_dim,
        "normalize": bool(normalize),
        "index_type": "IndexFlatIP",
        "docs": []  # list of {doc_id, source, pages, chunks}
    }
    faiss_index = faiss.IndexFlatIP(emb_dim)
    base = 0
    seen_hashes = set()
    return faiss_index, manifest, base, seen_hashes

def append_embeddings(emb_path: Path, new_vecs: np.ndarray):
    # Append to embeddings.npy (or create)
    if emb_path.exists():
        old = np.load(emb_path, mmap_mode="r")
        allv = np.concatenate([np.asarray(old), new_vecs], axis=0)
    else:
        allv = new_vecs
    np.save(emb_path, allv)

def ingest_sources(args) -> List[Tuple[str, str, List[Tuple[int, str]]]]:
    """
    Returns list of (doc_id, source, pages[(pno, text)]).
    doc_id is derived from file stem or arXiv id.
    """
    out = []
    tmpdir = Path(tempfile.mkdtemp(prefix="kb_arxiv_"))
    try:
        # arXiv
        for aid in args.ids or []:
            pdf = fetch_arxiv_pdf(aid, tmpdir)
            pages = extract_pdf_text(pdf, args.max_pages)
            pages = trim_after_references(pages) if args.trim_references else pages
            out.append((aid.replace("arXiv:", ""), str(pdf), pages))
        # Local PDFs
        for p in args.pdfs or []:
            p = Path(p)
            pages = extract_pdf_text(p, args.max_pages)
            pages = trim_after_references(pages) if args.trim_references else pages
            out.append((p.stem, str(p), pages))
        # Plain text files
        for t in args.txts or []:
            p = Path(t)
            text = normalize_text(p.read_text(encoding="utf-8", errors="ignore"))
            pages = [(1, text)]
            out.append((p.stem, str(p), pages))
    finally:
        # keep temp PDFs around? We used tmpdir; they can be deleted now.
        pass
    return out

def main():
    ap = argparse.ArgumentParser(description="Build or append a local FAISS KB from arXiv IDs, PDFs, or text files.")
    ap.add_argument("--out", required=True, help="Index directory (e.g. kb_out)")
    ap.add_argument("--append", action="store_true", help="Append to existing index instead of overwriting")

    ap.add_argument("--ids", nargs="*", default=None, help="arXiv IDs (e.g. 1506.02624 1807.06209)")
    ap.add_argument("--pdfs", nargs="*", default=None, help="Local PDF paths")
    ap.add_argument("--txts", nargs="*", default=None, help="Local .txt or .md files")

    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--max-pages", type=int, default=None, help="Limit number of PDF pages (debug)")
    ap.add_argument("--max-chars", type=int, default=1600, help="Chunk size in characters")
    ap.add_argument("--overlap", type=int, default=200, help="Chunk overlap in characters")
    ap.add_argument("--min-chars", type=int, default=200, help="Drop tiny chunks (< min-chars)")
    ap.add_argument("--trim-references", action="store_true", help="Trim pages after References/Bibliography")
    args = ap.parse_args()

    index_dir = Path(args.out)
    index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = index_dir / "manifest.json"
    index_path = index_dir / "faiss.index"
    emb_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.jsonl"

    if not args.append and index_dir.exists():
        # Fresh build — if anything exists, confirm overwrite by moving aside
        # (simpler: we just create/overwrite files we manage)
        pass

    # Load encoder early to know emb_dim
    model = SentenceTransformer(args.model)
    emb_dim = int(model.get_sentence_embedding_dimension())
    normalize = True  # we always normalize embeddings for cosine/IP

    faiss_index, manifest, base, seen_hashes = ensure_index(index_dir, args.model, emb_dim, normalize, args.append)

    sources = ingest_sources(args)
    if not sources:
        print("[INFO] Nothing to ingest.")
        return

    # We'll write chunks incrementally in append mode
    chunks_f = open(chunks_path, "a", encoding="utf-8")

    new_vecs_all = []
    new_chunks_count = 0

    for doc_id, source, pages in sources:
        # Page-level dedup: skip pages identical to any previous page in this doc
        page_hashes = set()
        kept_chunks = []
        # Build big text with separators to avoid cross-page sentence fusion
        # But we also keep page numbers for metadata.
        for pno, ptxt in pages:
            if not ptxt.strip():
                continue
            ph = sha1(ptxt[:4000])  # cheap page hash
            if ph in page_hashes:
                continue
            page_hashes.add(ph)

            # Chunk this page
            chs = chunk_text(ptxt, args.max_chars, args.overlap)
            for ch in chs:
                if len(ch) < args.min_chars:
                    continue
                h = sha1(doc_id + f"#{pno}#" + ch[:2000])
                if h in seen_hashes:
                    continue
                kept_chunks.append((pno, ch, h))

        if not kept_chunks:
            # Register doc even if skipped? Only if new pages exist—skip here
            continue

        # Embed batch for this doc
        texts = [c[1] for c in kept_chunks]
        vecs = model.encode(texts, batch_size=64, normalize_embeddings=normalize, show_progress_bar=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        faiss_index.add(vecs)
        new_vecs_all.append(vecs)
        new_chunks_count += len(kept_chunks)

        # Write chunks.jsonl lines
        for (pno, ch, h) in kept_chunks:
            line = {
                "doc_id": doc_id,
                "source": source,
                "page": pno,
                "hash": h,
                "text": ch
            }
            chunks_f.write(json.dumps(line, ensure_ascii=False) + "\n")
            seen_hashes.add(h)

        # Update doc entry in manifest
        manifest["docs"].append({
            "doc_id": doc_id,
            "source": source,
            "pages": len(pages),
            "chunks": len(kept_chunks)
        })

    chunks_f.close()

    # Update embeddings.npy
    if new_vecs_all:
        new_vecs = np.concatenate(new_vecs_all, axis=0)
        append_embeddings(emb_path, new_vecs)

    # Save FAISS + manifest
    faiss.write_index(faiss_index, str(index_path))
    write_json(manifest_path, manifest)

    print(f"[DONE] Indexed {new_chunks_count} new chunks "
          f"from {len(sources)} source(s).")
    print(f"       Index dir: {index_dir}")
    print(f"       Model    : {args.model} (dim={emb_dim}, normalized={normalize})")
    if new_vecs_all:
        print(f"       Vectors  : +{sum(v.shape[0] for v in new_vecs_all)} (total ~{faiss_index.ntotal})")
    else:
        print("       No new vectors added (dedup likely skipped duplicates).")


if __name__ == "__main__":
    main()
