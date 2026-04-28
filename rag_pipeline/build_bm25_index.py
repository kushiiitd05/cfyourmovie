"""
build_bm25_index.py
-------------------
Builds a BM25 index from the existing FAISS docstore page_content.
Run once after build_index.py. Saves a small pickle (~2MB).

Why BM25 alongside FAISS dense?
  FAISS (dense): great for semantic queries  "something emotional and slow-paced"
  BM25 (keyword): great for exact matches    "Spielberg", "Tom Hanks", "Star Wars"
  Hybrid: best of both worlds.

Output:
    rag_pipeline/bm25_index/bm25.pkl   — BM25Okapi model + movie_id list
"""
import sys
import re
import pickle
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer. No NLTK needed."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]


def build(save_dir: Path = None):
    from config import INDEX_DIR, RAG_DIR
    from langchain_community.vectorstores import FAISS
    from config import get_embedder

    if save_dir is None:
        save_dir = RAG_DIR / 'bm25_index'
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / 'bm25.pkl'

    print("Loading FAISS docstore...")
    embedder = get_embedder(input_type='query')
    vs = FAISS.load_local(str(INDEX_DIR), embedder,
                          allow_dangerous_deserialization=True)
    docs = list(vs.docstore._dict.values())
    print(f"  {len(docs)} documents found")

    # Sort by movie_id for deterministic ordering
    docs.sort(key=lambda d: d.metadata.get('movie_id', 0))

    movie_ids  = [d.metadata['movie_id'] for d in docs]
    corpus     = [_tokenize(d.page_content) for d in docs]

    print("Building BM25 index...")
    from rank_bm25 import BM25Okapi
    bm25 = BM25Okapi(corpus)

    payload = {'bm25': bm25, 'movie_ids': movie_ids}
    with open(out_path, 'wb') as f:
        pickle.dump(payload, f)

    print(f"BM25 index saved → {out_path}  ({out_path.stat().st_size // 1024} KB)")
    print(f"Sample tokenized (movie 0): {corpus[0][:15]}")
    return out_path


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    build()
