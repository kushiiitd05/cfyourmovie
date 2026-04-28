"""
nvidia_reranker.py
------------------
Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (local CPU, ~130MB).

Flow: FAISS top-30 → reranker → top-k by semantic relevance.
Used as a pre-filter in P2/P3/P4 before Mix_GPU fusion.

Cross-encoder vs bi-encoder (FAISS):
  FAISS bi-encoder: query → embed separately → fast vector search
  Cross-encoder:    (query, doc) pair → single relevance score → more accurate
  Typical pipeline: FAISS retrieves 30, cross-encoder reranks to 20.

Note: NVIDIA NIM reranker endpoint (llama-nemotron-rerank-1b-v2) requires a
downloadable-only model — not available via API on this key. Cross-encoder runs
directly without any API call (~200ms for 30 docs on M2 CPU).
"""
import logging

log = logging.getLogger(__name__)

# ── Cross-encoder (loaded lazily on first use) ─────────────────
_cross_encoder = None

def _get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        try:
            from sentence_transformers import CrossEncoder
            # Switch to BGE-reranker-base: modern, robust, and confirms works in this env
            _cross_encoder = CrossEncoder(
                'BAAI/bge-reranker-base',
                max_length=512,
                device='cpu',
            )
            log.info("[reranker] Cross-encoder loaded: BAAI/bge-reranker-base (CPU)")
        except Exception as e:
            log.warning(f"[reranker] Cross-encoder unavailable: {e}")
            _cross_encoder = None
    return _cross_encoder


def _cross_encode_faiss(query: str, faiss_results: list, top_k: int) -> list:
    """Rerank (Document, score) tuples using local cross-encoder."""
    model = _get_cross_encoder()
    if model is None:
        return [(doc, s, 0.0) for doc, s in faiss_results[:top_k]]

    # Ensure query is a valid string
    safe_query = str(query or "unknown").strip()[:500]
    if not safe_query: safe_query = "unknown query"

    pairs = [(safe_query, doc.page_content[:512]) for doc, _ in faiss_results]
    
    log.debug(f"[reranker] Running CE on {len(pairs)} FAISS pairs. Sample q: {safe_query[:50]}...")
    scores = model.predict(pairs)

    # Check for NaNs just in case, but BGE is confirmed stable
    clean_scores = [float(s) if float(s) == float(s) else 0.0 for s in scores]


    ranked = sorted(
        zip(faiss_results, clean_scores),
        key=lambda x: x[1], reverse=True
    )[:top_k]

    result = [(doc, fscore, float(ce_score)) for (doc, fscore), ce_score in ranked]
    log.info(f"[reranker] cross-encoder: {len(faiss_results)} → {len(result)} (CPU)")
    return result


def _cross_encode_movies(query: str, movies: list, text_key: str, top_k: int) -> list:
    """Rerank movie dict list using local cross-encoder."""
    model = _get_cross_encoder()
    if model is None:
        for m in movies:
            m.setdefault("nvidia_rank_score", 0.0)
        return movies[:top_k] if top_k else movies

    # Ensure query is a valid string
    safe_query = str(query or "unknown").strip()[:500]
    if not safe_query: safe_query = "unknown query"

    # Use overview → title → movie_id as fallback
    # CRITICAL: empty strings produce NaN logits from ms-marco — always supply non-empty text
    texts = [
        (str(m.get(text_key) or m.get("overview") or m.get("title") or m.get("movie_id", "unknown")).strip()
         or str(m.get("title") or m.get("movie_id", "unknown")))[:512]
        for m in movies
    ]
    pairs = [(safe_query, t if t else "unknown film") for t in texts]
    
    log.info(f"[reranker] Running CE on {len(pairs)} movie pairs. Q: {safe_query[:60]}...")
    if pairs:
        log.debug(f"[reranker] First pair: ({pairs[0][0][:50]}, {pairs[0][1][:50]})")
    
    scores = model.predict(pairs)

    for m, s in zip(movies, scores):
        val = float(s)
        m["nvidia_rank_score"] = round(val if val == val else 0.0, 4)


    result = sorted(movies, key=lambda m: m.get("nvidia_rank_score", -999), reverse=True)
    if top_k:
        result = result[:top_k]
    log.info(f"[reranker] cross-encoder movies: {len(movies)} → {len(result)} (CPU)")
    return result


# ── Public API ────────────────────────────────────────────────

def rerank_faiss_results(
    query: str,
    faiss_results: list,
    top_k: int = 15,
) -> list:
    """
    Rerank FAISS (Document, score) tuples using local cross-encoder.
    Returns list of (Document, faiss_score, rerank_score) tuples.
    """
    if not faiss_results:
        return faiss_results
    return _cross_encode_faiss(query, faiss_results, top_k)


def rerank_movies(
    query: str,
    movies: list[dict],
    text_key: str = "overview",
    top_k: int = None,
) -> list[dict]:
    """
    Rerank a list of movie dicts using local cross-encoder.
    Adds 'nvidia_rank_score' to each movie dict.
    Input: (query, movie[text_key]) pairs — text_key defaults to 'overview'.
    """
    if not movies:
        return movies
    return _cross_encode_movies(query, movies, text_key, top_k)
