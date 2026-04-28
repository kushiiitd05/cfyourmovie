"""
pipeline_4_hyde.py
------------------
Pipeline 4: HyDE Hybrid — primary research contribution.
Based on Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (CMU 2022).

Fixes applied:
  - FAISS now queries on combined HyDE + semantic_description (not HyDE alone)
    → preserves original query intent while benefiting from hypothetical enrichment
  - HyDE quality validation: if output is too short/generic, falls back to semantic_description
  - Parsed filters passed to mix_recommend() — was missing (same bug as P2)
  - Prebuilt O(1) movie_id lookup dict
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
from pipeline_2_dual import fuse_scores   # same fusion logic
from nvidia_reranker import rerank_faiss_results

log = logging.getLogger(__name__)

# ── Load FAISS once ───────────────────────────────────────────
log.info("[P4] Loading FAISS index...")
_embedder = get_embedder(input_type="query")
_vectorstore = FAISS.load_local(str(INDEX_DIR), _embedder,
                                allow_dangerous_deserialization=True)
_llm      = get_llm(temperature=0.3)
_llm_hyde = get_llm(temperature=0.6)

_id_to_doc = {}
for _, doc in _vectorstore.docstore._dict.items():
    mid = doc.metadata.get('movie_id')
    if mid is not None:
        _id_to_doc[int(mid)] = doc
log.info(f"[P4] Ready — {len(_id_to_doc)} movies in lookup")


# ── HyDE prompt ───────────────────────────────────────────────
_HYDE_SYSTEM = """You are a film critic and creative screenwriter.

A user is looking for movie recommendations. Write EXACTLY ONE SENTENCE
describing the ideal fictional film that would perfectly satisfy their request.

The film does not need to exist. Write it as a festival catalogue description:
include genre, tone, setting, key themes, directorial style, emotional register.
It should read like movie metadata — not like a user query.

Example request: "Something like Inception but more emotional"
Example: "A cerebral science-fiction drama that uses nested dream architecture as a metaphor for grief, blending Nolan-esque visual ambition with the quiet devastation of a European character study."

Respond with ONLY the one sentence. No preamble. No film titles. No quotes."""

_hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", _HYDE_SYSTEM),
    ("human", "User request: {query}\nSemantic summary: {semantic}"),
])
_hyde_chain = _hyde_prompt | _llm_hyde | StrOutputParser()


# ── Explainer prompt ──────────────────────────────────────────
_SYSTEM = """You are a movie recommendation assistant for a 1682-film database (MovieLens 100K, mostly 1990s).

The system used:
1. A personalized model trained on this user's rating history
2. Semantic matching using a hypothetical ideal film description (HyDE method)

Hypothetical ideal film used for retrieval: "{hypothetical}"
User's original request: {user_query}

Final recommendations:
{ranked_movies}

Instructions:
- 1 sentence briefly referencing the hypothetical ideal concept and how retrieval worked
- Top 3: one sentence each linking to themes in the hypothetical description
- Note any gaps between the request and what the database could provide"""

_exp_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Explain these recommendations."),
])
_exp_chain = _exp_prompt | _llm | StrOutputParser()


# ── HyDE generation with quality check ───────────────────────
def generate_hyde(query: str, semantic_desc: str) -> str:
    """
    Generate hypothetical ideal film description.
    Quality check: if output is too short (<30 chars) or unchanged from input,
    treat as a failure and fall back to semantic_desc.
    """
    try:
        result = _hyde_chain.invoke({"query": query, "semantic": semantic_desc}).strip()

        # Quality validation
        if len(result) < 30:
            log.warning(f"[P4] HyDE output too short ({len(result)} chars) — fallback")
            return semantic_desc
        if result.lower().strip() == query.lower().strip():
            log.warning("[P4] HyDE output identical to input — fallback")
            return semantic_desc

        log.info(f"[P4] HyDE generated: {result[:100]}...")
        return result

    except Exception as e:
        log.warning(f"[P4] HyDE generation failed: {e} — using semantic_desc")
        return semantic_desc


# ── Main function ─────────────────────────────────────────────
def run(user_id: int, query: str, n: int = 10,
        alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA) -> dict:

    # Step 1: Parse
    parsed = parse_query(query)
    n      = parsed.get('n', n)

    # Step 1B: Generate HyDE (only for non-trivial queries)
    semantic_desc = parsed.get('semantic_description', query)
    if not parsed.get('hypothetical_ideal', '').strip():
        hypothetical = generate_hyde(query, semantic_desc)
    else:
        # Parser already generated one — validate and use it
        hypothetical = parsed['hypothetical_ideal']
        if len(hypothetical) < 30:
            hypothetical = generate_hyde(query, semantic_desc)

    # Step 2A: Mix_GPU WITH filters (bug fix applied)
    mix_results = mix_recommend(
        user_id        = user_id,
        n              = 20,
        genre_filter   = parsed.get('genre_filter'),
        director_filter= parsed.get('director_filter'),
        year_min       = parsed.get('year_range', {}).get('min'),
        year_max       = parsed.get('year_range', {}).get('max'),
    )
    mix_scores = {m['movie_id']: m['mix_score'] for m in mix_results}
    log.info(f"[P4] Mix_GPU returned {len(mix_scores)} candidates")

    # Step 2B: FAISS with COMBINED query (HyDE + original semantic_desc), then NVIDIA reranker → top-20
    # Combining preserves original intent while benefiting from hypothetical enrichment
    combined_query = f"{hypothetical} {semantic_desc}".strip()
    faiss_raw      = _vectorstore.similarity_search_with_score(combined_query, k=30)
    faiss_reranked = rerank_faiss_results(combined_query, faiss_raw, top_k=20)
    faiss_results  = [(doc, fscore) for doc, fscore, *_ in faiss_reranked]
    faiss_scores   = {doc.metadata['movie_id']: float(score) for doc, score in faiss_results}
    ce_scores      = {doc.metadata['movie_id']: float(ce) for doc, _, ce in faiss_reranked}
    log.info(f"[P4] FAISS returned {len(faiss_scores)} candidates (post-rerank)")

    # Robustness
    if not mix_scores and not faiss_scores:
        return {"pipeline": "P4_HyDE", "user_id": user_id, "query": query,
                "movies": [], "explanation": "No candidates found.",
                "hypothetical": hypothetical, "parsed": parsed}

    if not mix_scores:
        alpha, beta = 0.0, 1.0
    elif not faiss_scores:
        alpha, beta = 1.0, 0.0

    # Step 3: Score FAISS-only items, fuse, rank
    faiss_only = set(faiss_scores) - set(mix_scores)
    if faiss_only:
        mix_scores.update(score_for_items(user_id, list(faiss_only)))

    fused   = fuse_scores(mix_scores, faiss_scores, alpha, beta)
    top_ids = sorted(fused, key=fused.get, reverse=True)[:n]

    # Build result list
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
        
        # Robust NaN guard for JSON compliance
        raw_ce = ce_scores.get(mid, 0.0)
        m['nvidia_rank_score'] = round(float(raw_ce) if raw_ce == raw_ce else -99.0, 4)
        movies.append(m)

    # Score ALL movies (including CF-only) against query so condition 4 works
    from nvidia_reranker import rerank_movies as _rerank
    movies = _rerank(combined_query, movies, top_k=None)

    # ── WEB FALLBACK: insufficient DB results → search the web ──
    from web_search_fallback import should_trigger_fallback, run as web_fallback_run
    trigger, reason = should_trigger_fallback(
        db_results=movies, faiss_scores=faiss_scores, n_requested=n,
    )
    if trigger:
        log.info(f"[P4] Web fallback triggered: {reason}")
        return web_fallback_run(
            query=query, user_id=user_id, n=n,
            semantic_description=parsed.get('semantic_description', ''),
            db_results=movies,
        )
    # ── END WEB FALLBACK ────────────────────────────────────────

    # Step 4: Explain
    explanation = _exp_chain.invoke({
        "user_query":    query,
        "hypothetical":  hypothetical,
        "ranked_movies": json.dumps(movies, indent=2),
    })

    return {
        "pipeline": "P4_HyDE", "user_id": user_id,
        "query": query, "parsed": parsed,
        "hypothetical": hypothetical,
        "alpha": alpha, "beta": beta,
        "movies": movies, "explanation": explanation,
    }


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for uid, q in [
        (1,  "Something like Inception but more emotional"),
        (42, "dark crime drama with moral ambiguity"),
        (15, "feel-good comedies from the 90s"),
    ]:
        print(f"\n{'='*55}\nUSER:{uid}  QUERY:{q}")
        r = run(uid, q, n=5)
        print(f"HYPOTHETICAL: {r['hypothetical']}")
        print(f"EXPLANATION:\n{r['explanation']}")
        for m in r['movies']:
            print(f"  [f={m['fused_score']:.3f}]  {m['title']} ({m['year']})")
