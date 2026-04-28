#!/usr/bin/env python3
"""
build_index.py
--------------
Builds the FAISS vector store from Master_final.csv using NVIDIA NIM embeddings.

Run once after mix_gpu_train.py.
Saves intermediate embeddings as checkpoint so index can be rebuilt
without re-embedding (saves 3-5 min on subsequent runs).

Output:
    rag_pipeline/faiss_index/        — LangChain FAISS index + docstore
    rag_pipeline/matrices/emb_nvidia.npy  — embedding checkpoint
"""

import sys
import ast
import numpy as np
import pandas as pd
from pathlib import Path
# #!/usr/bin/env python3
import os
import torch as _torch
if _torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.insert(0, str(Path(__file__).parent))
from config import META_PATH, INDEX_DIR, MATRICES_DIR, FAISS_DOC_TEMPLATE, get_embedder

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Embedding checkpoint — avoids re-embedding on index rebuild
# Named _nvidia to distinguish from old nomic-embed-text cache (incompatible vectors)
NVIDIA_CACHE = MATRICES_DIR / 'emb_nvidia.npy'


# ── Helpers ───────────────────────────────────────────────────
def _parse_list(val, max_items: int = None) -> list:
    if pd.isna(val):
        return []
    try:
        r = ast.literal_eval(str(val))
        if isinstance(r, list):
            return r[:max_items] if max_items else r
        return [str(r)]
    except Exception:
        return [str(val)]


def make_doc_text(row) -> str:
    """Build document text using the structured template from config."""
    genres   = ', '.join(_parse_list(row.get('genres')))
    cast     = ', '.join(_parse_list(row.get('top_cast'), max_items=3))
    keywords = ', '.join(_parse_list(row.get('movie_keywords'), max_items=10))
    overview = str(row.get('overview', '') or '')
    if overview == 'nan':
        overview = ''

    return FAISS_DOC_TEMPLATE.format(
        title        = str(row['title']),
        year         = int(row['year']) if pd.notna(row.get('year')) else 'unknown',
        overview     = overview,
        keywords     = keywords,
        genres       = genres,
        director     = str(row.get('director', '') or ''),
        cast         = cast,
        vote_average = float(row.get('vote_average', 0) or 0),
        runtime      = int(row['runtime']) if pd.notna(row.get('runtime')) and row['runtime'] > 0 else 'unknown',
    )


def make_metadata(row) -> dict:
    return {
        'movie_id':     int(row['item_id']),
        'title':        str(row['title']),
        'genres':       str(row.get('genres', '')),
        'director':     str(row.get('director', '')),
        'year':         int(row['year']) if pd.notna(row.get('year')) else 0,
        'vote_average': float(row.get('vote_average', 0.0) or 0.0),
        'popularity':   float(row.get('popularity', 0.0) or 0.0),
        'overview':     str(row.get('overview', ''))[:300],
    }


# ── Build ─────────────────────────────────────────────────────
def build():
    print("Loading metadata...")
    meta = pd.read_csv(META_PATH)
    meta.rename(columns={'movie_id': 'item_id'}, inplace=True)
    meta = meta.reset_index(drop=True)
    print(f"  {len(meta)} movies")

    print("\nConstructing documents...")
    docs = [
        Document(page_content=make_doc_text(row), metadata=make_metadata(row))
        for _, row in meta.iterrows()
    ]
    print(f"  {len(docs)} documents ready")
    print(f"\nSample (movie_id=1):\n  {docs[0].page_content[:250]}\n")

    # Use NVIDIA NIM passage embedder (asymmetric: "passage" for docs, "query" for search)
    embedder = get_embedder(input_type="passage")

    # ── Embedding checkpoint ──────────────────────────────────
    # Delete emb_nvidia.npy to force re-embedding (e.g. after doc changes).
    # NOTE: old emb_nomic.npy is INCOMPATIBLE — different model/dim, ignore it.
    if NVIDIA_CACHE.exists():
        print(f"Checkpoint found at {NVIDIA_CACHE} — loading cached embeddings...")
        cached_emb = np.load(NVIDIA_CACHE).tolist()

        if len(cached_emb) == len(docs):
            print("Building FAISS index from checkpoint (fast)...")
            texts     = [d.page_content for d in docs]
            metadatas = [d.metadata     for d in docs]
            vectorstore = FAISS.from_embeddings(
                text_embeddings=list(zip(texts, cached_emb)),
                embedding=embedder,
                metadatas=metadatas,
            )
        else:
            print(f"Checkpoint shape mismatch ({len(cached_emb)} vs {len(docs)}) — re-embedding...")
            NVIDIA_CACHE.unlink()
            vectorstore = _embed_and_save(docs, embedder)
    else:
        vectorstore = _embed_and_save(docs, embedder)

    vectorstore.save_local(str(INDEX_DIR))
    print(f"\nIndex saved to {INDEX_DIR}")

    # ── Verification ──────────────────────────────────────────
    print("\n" + "="*50)
    print("Verification (top 3 per query) — CHECK THESE BEFORE PROCEEDING:")
    test_queries = [
        "sci-fi film about time travel and space exploration",
        "Christopher Nolan psychological thriller",
        "animated Disney family movie for children",
        "crime drama with gangsters and moral corruption",
        "romantic comedy set in New York",
    ]
    for q in test_queries:
        results = vectorstore.similarity_search(q, k=3)
        print(f"\n  '{q}'")
        for r in results:
            print(f"    [{r.metadata['vote_average']:.1f}] "
                  f"{r.metadata['title']} ({r.metadata['year']})  "
                  f"{r.metadata['genres'][:60]}")

    print("\nDone. Run query_parser.py to test the LLM parser.")


def _embed_and_save(docs, embedder):
    texts     = [d.page_content for d in docs]
    metadatas = [d.metadata     for d in docs]
    print(f"Embedding {len(docs)} documents via NVIDIA NIM (llama-nemotron-embed-1b-v2)...")
    print("  Batching in chunks of 50 — ~35 API calls for 1682 movies...")

    raw_embs = embedder.embed_documents(texts)
    np.save(NVIDIA_CACHE, np.array(raw_embs, dtype=np.float32))
    print(f"  Checkpoint saved to {NVIDIA_CACHE}")

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip(texts, raw_embs)),
        embedding=embedder,
        metadatas=metadatas,
    )
    return vectorstore


if __name__ == '__main__':
    build()
