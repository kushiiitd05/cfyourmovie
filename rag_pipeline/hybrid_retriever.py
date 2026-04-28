"""
hybrid_retriever.py
-------------------
Fuses BM25 (keyword) + FAISS (dense) retrieval via Reciprocal Rank Fusion.

Usage:
    retriever = HybridRetriever()
    results = retriever.search("Spielberg thrillers", query_embedding, alpha=0.5, top_k=30)

alpha controls blend:
    0.0 → pure BM25 keyword
    0.5 → balanced (default)
    1.0 → pure FAISS dense (equivalent to P4)

RRF formula (Cormack et al. 2009):
    score(doc) = Σ  weight_i / (k + rank_i)
    where k=60 (standard constant that dampens rank differences)
"""
import sys
import pickle
import logging
import re
from pathlib import Path
from functools import lru_cache

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

_RRF_K = 60  # standard RRF constant


def _tokenize(text: str) -> list[str]:
    """Must match build_bm25_index.py tokenizer exactly."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return [t for t in text.split() if len(t) > 1]


class HybridRetriever:
    """
    Loads BM25 + FAISS once, exposes a single search() interface.
    Thread-safe for read operations.
    """

    def __init__(self, bm25_alpha: float = 0.5):
        """
        bm25_alpha: default blend (0=pure BM25, 1=pure FAISS).
        Can be overridden per-call.
        """
        self.default_alpha = bm25_alpha
        self._bm25 = None
        self._movie_ids = None
        self._vectorstore = None
        self._id_to_doc = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        from config import RAG_DIR, INDEX_DIR, get_embedder
        from langchain_community.vectorstores import FAISS

        # Load BM25
        bm25_path = RAG_DIR / 'bm25_index' / 'bm25.pkl'
        if not bm25_path.exists():
            log.warning(f"[hybrid] BM25 index not found at {bm25_path} — building now...")
            from build_bm25_index import build
            build()
        with open(bm25_path, 'rb') as f:
            payload = pickle.load(f)
        self._bm25 = payload['bm25']
        self._movie_ids = payload['movie_ids']
        log.info(f"[hybrid] BM25 loaded — {len(self._movie_ids)} movies")

        # Load FAISS
        log.info("[hybrid] Loading FAISS index...")
        embedder = get_embedder(input_type='query')
        self._vectorstore = FAISS.load_local(
            str(INDEX_DIR), embedder, allow_dangerous_deserialization=True
        )
        for _, doc in self._vectorstore.docstore._dict.items():
            mid = doc.metadata.get('movie_id')
            if mid is not None:
                self._id_to_doc[int(mid)] = doc
        log.info(f"[hybrid] FAISS loaded — {len(self._id_to_doc)} movies")
        self._loaded = True

    # ── BM25 retrieval ────────────────────────────────────────
    def _bm25_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Returns [(movie_id, bm25_score), ...] sorted descending."""
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Pair with movie_ids, sort by score
        ranked = sorted(
            zip(self._movie_ids, scores),
            key=lambda x: x[1], reverse=True
        )[:top_k]
        return [(int(mid), float(s)) for mid, s in ranked if s > 0]

    # ── FAISS retrieval ───────────────────────────────────────
    def _faiss_search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """Returns [(movie_id, faiss_score), ...] sorted descending (lower=closer in FAISS)."""
        raw = self._vectorstore.similarity_search_with_score(query, k=top_k)
        # FAISS L2 distance: lower = more similar. Invert for ranking.
        # Normalize to [0,1] range within result set
        if not raw:
            return []
        scores = [float(s) for _, s in raw]
        s_min, s_max = min(scores), max(scores)
        s_range = s_max - s_min if s_max != s_min else 1.0
        results = []
        for (doc, score) in raw:
            mid = doc.metadata.get('movie_id')
            if mid is not None:
                # Invert: high score = high similarity
                norm = 1.0 - (float(score) - s_min) / s_range
                results.append((int(mid), norm))
        return sorted(results, key=lambda x: x[1], reverse=True)

    # ── RRF fusion ────────────────────────────────────────────
    @staticmethod
    def _rrf_fuse(
        bm25_ranked: list[tuple[int, float]],
        faiss_ranked: list[tuple[int, float]],
        alpha: float,
        k: int = _RRF_K,
    ) -> dict[int, float]:
        """
        Reciprocal Rank Fusion.
        alpha weights FAISS contribution; (1-alpha) weights BM25.
        Returns {movie_id: rrf_score}.
        """
        scores: dict[int, float] = {}

        # BM25 contribution
        bm25_weight = 1.0 - alpha
        for rank, (mid, _) in enumerate(bm25_ranked):
            scores[mid] = scores.get(mid, 0.0) + bm25_weight / (k + rank + 1)

        # FAISS contribution
        faiss_weight = alpha
        for rank, (mid, _) in enumerate(faiss_ranked):
            scores[mid] = scores.get(mid, 0.0) + faiss_weight / (k + rank + 1)

        return scores

    # ── Public API ────────────────────────────────────────────
    def search(
        self,
        query: str,
        alpha: float = None,
        top_k: int = 30,
        candidate_k: int = 60,
    ) -> list[tuple[int, float, object]]:
        """
        Hybrid BM25 + FAISS search via RRF.

        Args:
            query:       natural language query (text used for both BM25 and FAISS)
            alpha:       0=BM25 only, 0.5=balanced, 1=FAISS only (default: self.default_alpha)
            top_k:       how many final results to return
            candidate_k: how many candidates each engine retrieves before fusion

        Returns:
            [(movie_id, rrf_score, doc), ...] sorted by rrf_score descending
        """
        self._load()
        if alpha is None:
            alpha = self.default_alpha

        bm25_results  = self._bm25_search(query, top_k=candidate_k)
        faiss_results = self._faiss_search(query, top_k=candidate_k)

        log.debug(f"[hybrid] BM25: {len(bm25_results)} hits | FAISS: {len(faiss_results)} hits")

        fused = self._rrf_fuse(bm25_results, faiss_results, alpha=alpha)
        top = sorted(fused, key=fused.get, reverse=True)[:top_k]

        results = []
        for mid in top:
            doc = self._id_to_doc.get(mid)
            results.append((mid, fused[mid], doc))

        log.info(f"[hybrid] '{query[:50]}' → {len(results)} results (alpha={alpha})")
        return results

    def get_doc(self, movie_id: int):
        self._load()
        return self._id_to_doc.get(movie_id)


# ── Module-level singleton (loaded lazily on first call) ──────
_retriever: HybridRetriever = None


def get_retriever(alpha: float = 0.5) -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(bm25_alpha=alpha)
    return _retriever


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    import logging as _log
    _log.basicConfig(level=logging.INFO)

    r = HybridRetriever(bm25_alpha=0.5)

    test_queries = [
        ("Spielberg action adventure", 0.3),   # keyword-heavy → lower alpha (more BM25)
        ("dark psychological thriller with moral ambiguity", 0.7),  # semantic → higher alpha
        ("Tom Hanks comedy", 0.4),
        ("romantic drama 90s", 0.5),
    ]

    for query, alpha in test_queries:
        print(f"\n{'─'*60}")
        print(f"QUERY: {query!r}  (alpha={alpha})")
        results = r.search(query, alpha=alpha, top_k=5)
        for mid, score, doc in results:
            title = doc.metadata['title'] if doc else str(mid)
            print(f"  [{score:.4f}] {title}")
