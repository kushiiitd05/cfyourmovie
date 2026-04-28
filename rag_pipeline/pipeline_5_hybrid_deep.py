"""
pipeline_5_hybrid_deep.py
--------------------------
Pipeline 5: Hybrid-Deep — evolved from P4 (HyDE).

Improvements over P4:
  1. BM25 + FAISS hybrid retrieval via RRF  (improves recall for keyword queries)
  2. Sub-query decomposition (2 sub-questions per query, merged candidates)
  3. Cross-encoder reranker moved AFTER CF fusion  (better final ordering)
  4. Adaptive alpha: semantic queries → higher FAISS weight, keyword → higher BM25

Scoring authority:
  - Mix_GPU CF scoring is the PRIMARY ranking signal (unchanged from P4)
  - RAG (hybrid BM25+FAISS) expands the candidate pool and scores content relevance
  - Reranker is a final post-fusion quality filter, not the scorer

Architecture:
    query
      → parse_query          (existing)
      → generate_hyde        (from P4)
      → decompose_query      (NEW: 2 sub-questions via LLM)
      → hybrid_retrieve      (NEW: BM25+FAISS RRF for each sub-query + HyDE query)
      → deduplicate pool
      → mix_recommend        (CF scoring — unchanged)
      → score_for_items      (CF score for FAISS-only items — unchanged)
      → fuse_scores          (Z-norm alpha*CF + beta*RAG — unchanged)
      → rerank_movies        (AFTER fusion — improvement over P4)
      → web_fallback         (unchanged)
      → LLM synthesis
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

from config import get_llm, DEFAULT_ALPHA, DEFAULT_BETA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from query_parser import parse_query
from mix_gpu_infer import recommend as mix_recommend, score_for_items
from pipeline_2_dual import fuse_scores
from nvidia_reranker import rerank_movies
from hybrid_retriever import get_retriever
from web_search_fallback import should_trigger_fallback, run as web_fallback_run

log = logging.getLogger(__name__)

# ── LLM setup ─────────────────────────────────────────────────
_llm      = get_llm(temperature=0.2)
_llm_hyde = get_llm(temperature=0.6)

# ── HyDE prompt (same as P4) ──────────────────────────────────
_HYDE_SYSTEM = """You are a film critic and creative screenwriter.

A user is looking for movie recommendations. Write EXACTLY ONE SENTENCE
describing the ideal fictional film that would perfectly satisfy their request.

The film does not need to exist. Write it as a festival catalogue description:
include genre, tone, setting, key themes, directorial style, emotional register.
It should read like movie metadata — not like a user query.

Respond with ONLY the one sentence. No preamble. No film titles. No quotes."""

_hyde_prompt = ChatPromptTemplate.from_messages([
    ("system", _HYDE_SYSTEM),
    ("human", "User request: {query}\nSemantic summary: {semantic}"),
])
_hyde_chain = _hyde_prompt | _llm_hyde | StrOutputParser()


# ── Sub-query decomposer ──────────────────────────────────────
_DECOMPOSE_SYSTEM = """You are a search query decomposer for a movie recommendation system.

DATABASE: MovieLens 100K — 1682 films, mostly 1990s.

Given a user's movie request, generate exactly 2 search sub-queries that together
cover the full intent of the request from different angles.

Each sub-query should:
- Be a short search phrase (5-15 words)
- Cover a different aspect of the request
- Be suitable as input to both keyword search AND semantic search

Output ONLY valid JSON. No markdown. No explanation.

Schema:
{{"sub_queries": ["<sub-query 1>", "<sub-query 2>"], "alpha_hints": [<0.0-1.0>, <0.0-1.0>]}}

alpha_hints: how FAISS-semantic vs BM25-keyword each sub-query should be.
  0.0 = pure keyword (director name, actor name, exact title fragment)
  0.5 = balanced
  1.0 = pure semantic (mood, themes, emotions)

Examples:
  Input: "dark crime drama with strong female lead from the 90s"
  Output: {{"sub_queries": ["crime thriller neo-noir female protagonist", "dark atmospheric drama 1990s"], "alpha_hints": [0.7, 0.5]}}

  Input: "Spielberg adventure films"
  Output: {{"sub_queries": ["Spielberg director adventure action", "adventure family epic blockbuster"], "alpha_hints": [0.2, 0.7]}}"""

_decompose_prompt = ChatPromptTemplate.from_messages([
    ("system", _DECOMPOSE_SYSTEM),
    ("human", "User query: {query}\nSemantic description: {semantic}"),
])
_decompose_chain = _decompose_prompt | get_llm(temperature=0, json_mode=True) | JsonOutputParser()


# ── Explainer prompt ──────────────────────────────────────────
_EXPLAIN_SYSTEM = """You are a movie recommendation assistant for a 1682-film database (MovieLens 100K, mostly 1990s).

This recommendation used:
1. A personalized model trained on this user's rating history
2. Hybrid search: BM25 keyword + semantic FAISS retrieval
3. A hypothetical ideal film description to enrich semantic search

Search queries used: {sub_queries}
Hypothetical ideal film: "{hypothetical}"
User's original request: {user_query}

Final recommendations:
{ranked_movies}

Instructions:
- 1 sentence explaining the hybrid search approach used
- For top 3 films: one specific sentence each linking to the search queries or hypothetical description
- If results are limited, acknowledge what the database could/couldn't provide"""

_exp_prompt = ChatPromptTemplate.from_messages([
    ("system", _EXPLAIN_SYSTEM),
    ("human", "Explain these recommendations."),
])
_exp_chain = _exp_prompt | _llm | StrOutputParser()


# ── HyDE generation (reused from P4 logic) ───────────────────
def _generate_hyde(query: str, semantic_desc: str) -> str:
    try:
        result = _hyde_chain.invoke({"query": query, "semantic": semantic_desc}).strip()
        if len(result) < 30 or result.lower().strip() == query.lower().strip():
            log.warning("[P5] HyDE quality check failed — using semantic_desc")
            return semantic_desc
        log.info(f"[P5] HyDE: {result[:100]}...")
        return result
    except Exception as e:
        log.warning(f"[P5] HyDE failed: {e}")
        return semantic_desc


# ── Sub-query decomposition ───────────────────────────────────
def _decompose(query: str, semantic_desc: str) -> tuple[list[str], list[float]]:
    """
    Returns (sub_queries, alpha_hints).
    Falls back to [semantic_desc] with alpha=0.6 on any error.
    """
    try:
        result = _decompose_chain.invoke({"query": query, "semantic": semantic_desc})
        sqs = result.get("sub_queries", [])
        alphas = result.get("alpha_hints", [0.5, 0.5])

        # Validate
        if not isinstance(sqs, list) or len(sqs) < 1:
            raise ValueError(f"Invalid sub_queries: {sqs}")
        # Clamp alphas
        alphas = [max(0.0, min(1.0, float(a))) for a in alphas]
        # Pad if needed
        while len(alphas) < len(sqs):
            alphas.append(0.5)

        log.info(f"[P5] Sub-queries: {sqs}")
        return sqs[:2], alphas[:2]

    except Exception as e:
        log.warning(f"[P5] Decompose failed: {e} — using semantic fallback")
        return [semantic_desc], [0.6]


# ── Candidate pool builder ────────────────────────────────────
def _build_candidate_pool(
    hyde_query: str,
    sub_queries: list[str],
    alpha_hints: list[float],
    candidate_k: int = 30,
) -> dict[int, float]:
    """
    Run hybrid retrieval for HyDE query + each sub-query.
    Merge by taking max RRF score per movie_id.
    Returns {movie_id: max_rrf_score}.
    """
    retriever = get_retriever()
    all_scores: dict[int, float] = {}

    # HyDE query (high semantic weight since it's a hypothetical description)
    hyde_results = retriever.search(hyde_query, alpha=0.75, top_k=candidate_k)
    for mid, score, _ in hyde_results:
        all_scores[mid] = max(all_scores.get(mid, 0.0), score)

    # Sub-queries with their adaptive alphas
    for sq, alpha in zip(sub_queries, alpha_hints):
        sq_results = retriever.search(sq, alpha=alpha, top_k=candidate_k)
        for mid, score, _ in sq_results:
            all_scores[mid] = max(all_scores.get(mid, 0.0), score)

    log.info(f"[P5] Candidate pool: {len(all_scores)} unique movies")
    return all_scores


# ── Main function ─────────────────────────────────────────────
def run(
    user_id: int,
    query: str,
    n: int = 10,
    alpha: float = DEFAULT_ALPHA,
    beta: float = DEFAULT_BETA,
) -> dict:

    # Step 1: Parse query
    parsed = parse_query(query)
    n = parsed.get('n', n)
    semantic_desc = parsed.get('semantic_description', query)

    # Early exit: DB cannot satisfy this query (foreign films, post-1998 titles, etc.)
    if not parsed.get('db_satisfiable', True):
        constraint = parsed.get('constraint_notes', 'Query cannot be satisfied by the local database')
        log.info(f"[P5] DB unsatisfiable — routing directly to web fallback. Reason: {constraint}")
        result = web_fallback_run(
            query=query, user_id=user_id, n=n,
            semantic_description=semantic_desc,
            db_results=[],
            constraint_notes=constraint,
        )
        result['pipeline'] = 'P5_HybridDeep_WebFallback'
        result.setdefault('sub_queries', [])
        result.setdefault('hypothetical', semantic_desc)
        return result

    # Step 2: Generate HyDE
    hypothetical = parsed.get('hypothetical_ideal', '').strip()
    if len(hypothetical) < 30:
        hypothetical = _generate_hyde(query, semantic_desc)

    # Step 3: Decompose into sub-queries
    sub_queries, alpha_hints = _decompose(query, semantic_desc)

    # Step 4: Build candidate pool via hybrid retrieval
    # Uses HyDE doc + sub-queries, all with adaptive alpha
    rag_scores = _build_candidate_pool(
        hyde_query=hypothetical,
        sub_queries=sub_queries,
        alpha_hints=alpha_hints,
        candidate_k=30,
    )

    # Step 5: Mix_GPU CF scoring (primary signal — unchanged)
    mix_results = mix_recommend(
        user_id         = user_id,
        n               = 30,
        genre_filter    = parsed.get('genre_filter'),
        director_filter = parsed.get('director_filter'),
        year_min        = parsed.get('year_range', {}).get('min'),
        year_max        = parsed.get('year_range', {}).get('max'),
    )
    mix_scores = {m['movie_id']: m['mix_score'] for m in mix_results}
    log.info(f"[P5] Mix_GPU: {len(mix_scores)} candidates")

    if not mix_scores and not rag_scores:
        return {
            "pipeline": "P5_HybridDeep", "user_id": user_id, "query": query,
            "movies": [], "explanation": "No candidates found.",
            "hypothetical": hypothetical, "sub_queries": sub_queries,
            "alpha": alpha, "beta": beta,
        }

    # Robustness: if one engine returns nothing, weight other fully
    if not mix_scores:
        alpha, beta = 0.0, 1.0
    elif not rag_scores:
        alpha, beta = 1.0, 0.0

    # Step 6: Score CF for RAG-only items (items in rag_scores but not mix_scores)
    rag_only = set(rag_scores) - set(mix_scores)
    if rag_only:
        extra = score_for_items(user_id, list(rag_only))
        mix_scores.update(extra)
        log.info(f"[P5] Scored {len(rag_only)} RAG-only items via CF")

    # Step 7: Fuse (Z-norm alpha*CF + beta*RAG)
    fused = fuse_scores(mix_scores, rag_scores, alpha, beta)
    top_ids = sorted(fused, key=fused.get, reverse=True)[:n * 2]  # take 2x for reranker

    # Build movie list for reranker
    retriever = get_retriever()
    movies = []
    for mid in top_ids:
        doc = retriever.get_doc(mid)
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
                'rag_score':    round(float(rag_scores.get(mid, 0.0)), 4),
                'fused_score':  round(float(fused[mid]), 4),
            }
            movies.append(m)

    # Step 8: Cross-encoder rerank AFTER CF fusion (key improvement over P4)
    # P4 reranked before fusion; P5 reranks the final fused ranking
    # Combined query: hypothetical description captures intent richly
    rerank_query = f"{hypothetical} {semantic_desc}".strip()
    log.info(f"[P5] Final rerank query (len={len(rerank_query)}): {rerank_query[:100]}...")
    movies = rerank_movies(rerank_query, movies, top_k=n)

    # Guard: if cross-encoder returned NaN (-99.0) for all results (Python 3.14
    # compatibility issue with sentence-transformers), strip the key so
    # web_fallback condition 4 doesn't fire on a broken model signal.
    ce_scores = [m.get("nvidia_rank_score", 0) for m in movies if "nvidia_rank_score" in m]
    if ce_scores and all(s <= -90 for s in ce_scores):
        log.warning("[P5] Cross-encoder returned NaN for all results — skipping rerank signal")
        for m in movies:
            m.pop("nvidia_rank_score", None)

    # Step 9: Web fallback
    # Note: rag_scores uses RRF scale (0.01–0.02), not FAISS L2 — skip score threshold,
    # rely only on db_results count check.
    trigger, reason = should_trigger_fallback(
        db_results=movies,
        faiss_scores=None,
        n_requested=n,
    )
    if trigger:
        log.info(f"[P5] Web fallback triggered: {reason}")
        result = web_fallback_run(
            query=query, user_id=user_id, n=n,
            semantic_description=semantic_desc,
            db_results=movies,
        )
        result.setdefault('hypothetical', hypothetical)
        result.setdefault('sub_queries', sub_queries)
        result['pipeline'] = 'P5_HybridDeep_WebFallback'
        return result

    # Step 10: LLM synthesis
    explanation = _exp_chain.invoke({
        "user_query":    query,
        "hypothetical":  hypothetical,
        "sub_queries":   json.dumps(sub_queries),
        "ranked_movies": json.dumps(movies[:5], indent=2),  # top 5 for context
    })

    return {
        "pipeline":    "P5_HybridDeep",
        "user_id":     user_id,
        "query":       query,
        "parsed":      parsed,
        "hypothetical": hypothetical,
        "sub_queries": sub_queries,
        "alpha":       alpha,
        "beta":        beta,
        "movies":      movies,
        "explanation": explanation,
    }


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    tests = [
        (1,  "Something like Inception but more emotional"),
        (42, "dark crime drama with moral ambiguity"),
        (15, "feel-good comedies from the 90s"),
        (7,  "Spielberg adventure films"),          # keyword-heavy — BM25 should shine
        (3,  "slow-paced thoughtful drama about family"),  # semantic — FAISS should shine
    ]

    for uid, q in tests:
        print(f"\n{'='*60}\nUSER:{uid}  QUERY: {q}")
        r = run(uid, q, n=5)
        print(f"HYPOTHETICAL: {r['hypothetical']}")
        print(f"SUB-QUERIES:  {r['sub_queries']}")
        print(f"EXPLANATION:\n{r['explanation']}")
        for m in r['movies']:
            print(f"  [fused={m['fused_score']:.3f} rag={m['rag_score']:.3f}] {m['title']} ({m.get('year','')})")
