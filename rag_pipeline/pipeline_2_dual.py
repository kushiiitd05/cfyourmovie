"""
pipeline_2_dual.py
------------------
Pipeline 2: Dual-Engine Hybrid (Mix_GPU + FAISS → fused score → LLM).

Fixes applied:
  - Parsed filters (genre, director, year) NOW passed to mix_recommend() — was a bug
  - Prebuilt O(1) movie_id lookup dict (no repeated FAISS calls for metadata)
  - Robust empty-candidate handling if one engine returns nothing
  - Logging throughout
"""
import os
import torch as _t
if _t.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import json
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import get_llm, get_embedder, INDEX_DIR, DEFAULT_ALPHA, DEFAULT_BETA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from query_parser import parse_query
from mix_gpu_infer import recommend as mix_recommend, score_for_items
from web_search_fallback import should_trigger_fallback, run as web_fallback_run

log = logging.getLogger(__name__)

# ── Load FAISS once ───────────────────────────────────────────
log.info("[P2] Loading FAISS index...")
_embedder    = get_embedder(input_type="query")
_vectorstore = FAISS.load_local(str(INDEX_DIR), _embedder,
                                allow_dangerous_deserialization=True)
_llm         = get_llm(temperature=0.3)

# Prebuilt O(1) lookup
_id_to_doc = {}
for _, doc in _vectorstore.docstore._dict.items():
    mid = doc.metadata.get('movie_id')
    if mid is not None:
        _id_to_doc[int(mid)] = doc
log.info(f"[P2] Ready — {len(_id_to_doc)} movies in lookup")


# ── Score fusion ──────────────────────────────────────────────
def fuse_scores(mix_scores: dict, faiss_scores: dict,
                alpha: float, beta: float) -> dict:
    """
    Z-normalize both score sets independently, then fuse.
    Returns {movie_id: fused_score} for all candidate movies.
    """
    all_ids    = sorted(set(mix_scores) | set(faiss_scores))
    mix_vals   = np.array([mix_scores.get(mid,  0.0) for mid in all_ids], dtype=np.float32)
    faiss_vals = np.array([faiss_scores.get(mid, 0.0) for mid in all_ids], dtype=np.float32)

    mix_norm   = (mix_vals   - mix_vals.mean())   / (mix_vals.std()   + 1e-8)
    faiss_norm = (faiss_vals - faiss_vals.mean()) / (faiss_vals.std() + 1e-8)

    fused = alpha * mix_norm + beta * faiss_norm
    return dict(zip(all_ids, fused.tolist()))


# ── Prompt ────────────────────────────────────────────────────
_SYSTEM = """You are a movie recommendation assistant for a 1682-film database (MovieLens 100K, mostly 1990s).

These recommendations combined two signals:
1. Personalized model trained on this user's rating history
2. Semantic similarity to the query description

Rankings are FINAL. Do not reorder, rescore, or add any films not listed.

User's request: {user_query}

Final recommendations (personalization + semantic fusion):
{ranked_movies}

Instructions:
- 2 sentences noting the blend of personal taste and semantic relevance
- Top 3: one sentence each explaining the match
- If user asked for something specific: address how well the database matched it"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Explain these recommendations."),
])
_chain = _prompt | _llm | StrOutputParser()


# ── Main function ─────────────────────────────────────────────
def run(user_id: int, query: str, n: int = 10,
        alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> dict:

    # Step 1: Parse
    parsed = parse_query(query)
    n      = parsed.get('n', n)

    # Step 2A: Mix_GPU — top-20 WITH filters (bug fix: filters were missing before)
    mix_results = mix_recommend(
        user_id        = user_id,
        n              = 20,
        genre_filter   = parsed.get('genre_filter'),
        director_filter= parsed.get('director_filter'),
        year_min       = parsed.get('year_range', {}).get('min'),
        year_max       = parsed.get('year_range', {}).get('max'),
    )
    mix_scores = {m['movie_id']: m['mix_score'] for m in mix_results}
    log.info(f"[P2] Mix_GPU returned {len(mix_scores)} candidates for user={user_id}")

    # Step 2B: FAISS — top-30 semantic candidates, then NVIDIA reranker → top-20
    from nvidia_reranker import rerank_faiss_results
    faiss_raw = _vectorstore.similarity_search_with_score(
        parsed['semantic_description'], k=30
    )
    faiss_reranked = rerank_faiss_results(parsed['semantic_description'], faiss_raw, top_k=20)
    faiss_results = [(doc, fscore) for doc, fscore, *_ in faiss_reranked]
    faiss_scores = {doc.metadata['movie_id']: float(score) for doc, score in faiss_results}
    ce_scores = {doc.metadata['movie_id']: float(ce) for doc, _, ce in faiss_reranked}
    log.info(f"[P2] FAISS returned {len(faiss_scores)} candidates")

    # Robustness: if one engine returned nothing, use the other exclusively
    if not mix_scores and not faiss_scores:
        log.error("[P2] Both engines returned empty — returning empty result")
        return {"pipeline": "P2_DualEngine", "user_id": user_id, "query": query,
                "movies": [], "explanation": "No candidates found.", "parsed": parsed}

    if not mix_scores:
        log.warning("[P2] Mix_GPU empty — using FAISS only (α forced to 0)")
        alpha, beta = 0.0, 1.0
    elif not faiss_scores:
        log.warning("[P2] FAISS empty — using Mix_GPU only (β forced to 0)")
        alpha, beta = 1.0, 0.0

    # Step 3A: Score FAISS-only candidates via Mix_GPU matrix lookup
    faiss_only = set(faiss_scores) - set(mix_scores)
    if faiss_only:
        mix_scores.update(score_for_items(user_id, list(faiss_only)))

    # Step 3B: Fuse
    fused   = fuse_scores(mix_scores, faiss_scores, alpha, beta)
    top_ids = sorted(fused, key=fused.get, reverse=True)[:n]

    # Step 3C: Build result list using O(1) lookup
    movies = []
    for mid in top_ids:
        doc = _id_to_doc.get(mid)
        if doc:
            m = {
                'movie_id':     mid,
                'title':        doc.metadata['title'],
                'year':         doc.metadata.get('year'),
                'genres':       doc.metadata.get('genres', ''),
                'director':     doc.metadata.get('director', ''),
                'vote_average': doc.metadata.get('vote_average', 0),
                'overview':     doc.metadata.get('overview', '')[:300],
                'mix_score':    round(float(mix_scores.get(mid, 0.0)), 4),
            }
        else:
            m = {'movie_id': mid, 'title': str(mid), 'mix_score': 0.0}

        m['faiss_score'] = round(float(faiss_scores.get(mid, 0.0)), 4)
        m['fused_score'] = round(float(fused[mid]), 4)
        
        # Robust NaN guard for JSON compliance (Python 3.14)
        raw_ce = ce_scores.get(mid, 0.0)
        m['nvidia_rank_score'] = round(float(raw_ce) if raw_ce == raw_ce else -99.0, 4)
        movies.append(m)

    # ── WEB FALLBACK: insufficient DB results → search the web ──
    trigger, reason = should_trigger_fallback(
        db_results=movies, faiss_scores=faiss_scores, n_requested=n,
    )
    if trigger:
        log.info(f"[P2] Web fallback triggered: {reason}")
        return web_fallback_run(
            query=query, user_id=user_id, n=n,
            semantic_description=parsed.get('semantic_description', ''),
            db_results=movies,
        )
    # ── END WEB FALLBACK ────────────────────────────────────────

    # Step 4: Explain
    explanation = _chain.invoke({
        "user_query":    query,
        "ranked_movies": json.dumps(movies, indent=2),
    })

    return {
        "pipeline": "P2_DualEngine", "user_id": user_id,
        "query": query, "parsed": parsed,
        "alpha": alpha, "beta": beta,
        "movies": movies, "explanation": explanation,
    }


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for uid, q in [(1, "Recommend movies"),
                   (42, "sci-fi about AI, user 42"),
                   (15, "crime dramas user 15")]:
        print(f"\n{'='*55}\nUSER:{uid}  QUERY:{q}")
        r = run(uid, q, n=5)
        print(f"EXPLANATION:\n{r['explanation']}")
        for m in r['movies']:
            print(f"  [f={m['fused_score']:.3f} m={m['mix_score']:.3f} "
                  f"r={m['faiss_score']:.3f}]  {m['title']} ({m['year']})")
