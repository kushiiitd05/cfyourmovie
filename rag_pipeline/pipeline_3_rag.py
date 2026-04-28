"""
pipeline_3_rag.py
-----------------
Pipeline 3: RAG-only baseline. No Mix_GPU. No personalization.
Ablation baseline — expected lowest MAP@10/nDCG@10 of the four pipelines.

Fixes applied:
  - Parsed filters (genre, director) now applied via FAISS metadata filtering
  - FAISS scores normalized before returning (consistent with P2/P4)
  - Prebuilt movie_id→doc lookup for O(1) access
"""
import os
import torch as _t
if _t.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import get_llm, get_embedder, INDEX_DIR
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from query_parser import parse_query
from nvidia_reranker import rerank_movies
from web_search_fallback import should_trigger_fallback, run as web_fallback_run

log = logging.getLogger(__name__)

# ── Load once ─────────────────────────────────────────────────
log.info("[P3] Loading FAISS index...")
_embedder    = get_embedder(input_type="query")
_vectorstore = FAISS.load_local(str(INDEX_DIR), _embedder,
                                allow_dangerous_deserialization=True)
_llm         = get_llm(temperature=0.3)
log.info("[P3] Ready")

# ── Explainer prompt ──────────────────────────────────────────
_SYSTEM = """You are a movie recommendation assistant for a 1682-film database (MovieLens 100K, mostly 1990s).

The system retrieved these films using semantic similarity to the user's query.
These are the ONLY movies you may reference. Do not mention any film not in this list.

Retrieved movies:
{retrieved_movies}

Instructions:
- 1-2 sentences introducing how these match the query
- For top 3: one sentence each with specific reasons
- Be honest about limited matches for very specific requests
- Note results are semantic similarity, not personalized to viewing history"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "User query: {query}\nExplain these {n} recommendations."),
])
_chain = _prompt | _llm | StrOutputParser()


# ── Main function ─────────────────────────────────────────────
def run(query: str, n: int = 10) -> dict:
    parsed = parse_query(query)
    n      = parsed.get('n', n)

    # Build metadata filter dict for FAISS if genre/director specified
    # FAISS LangChain supports equality filters on metadata fields
    faiss_filter = {}
    if parsed.get('director_filter'):
        faiss_filter['director'] = parsed['director_filter']

    # Retrieve — use semantic_description; genre filter applied post-retrieval
    # (FAISS metadata filter on list fields is unreliable — filter manually)
    search_k   = max(n * 3, 30) if parsed.get('genre_filter') else n
    raw_results = _vectorstore.similarity_search_with_score(
        parsed['semantic_description'], k=search_k,
        filter=faiss_filter if faiss_filter else None,
    )

    # Post-retrieval genre filter
    if parsed.get('genre_filter'):
        gf = {g.strip().lower() for g in parsed['genre_filter']}
        raw_results = [
            (doc, score) for doc, score in raw_results
            if gf.intersection(
                {g.lower() for g in _parse_genres(doc.metadata.get('genres', ''))}
            )
        ]

    # Limit to n (retrieve extra for reranker, then cut to n)
    rerank_k = max(n, 20)
    raw_results = raw_results[:rerank_k]

    # Normalize FAISS scores to [0,1] within this result set
    scores = [float(s) for _, s in raw_results]
    s_min, s_max = min(scores, default=0), max(scores, default=1)
    s_range = s_max - s_min if s_max != s_min else 1.0

    movies = []
    for (doc, score), norm_s in zip(raw_results,
                                     [(s - s_min) / s_range for s in scores]):
        movies.append({
            'movie_id':     doc.metadata['movie_id'],
            'title':        doc.metadata['title'],
            'year':         doc.metadata.get('year'),
            'genres':       doc.metadata.get('genres', ''),
            'director':     doc.metadata.get('director', ''),
            'vote_average': doc.metadata.get('vote_average', 0),
            'faiss_score':  round(float(score), 4),
            'faiss_score_norm': round(norm_s, 4),
            'overview':     doc.metadata.get('overview', '')[:250],
        })

    # Cross-encoder reranker — rescore by semantic relevance, keep top n
    movies = rerank_movies(parsed['semantic_description'], movies, top_k=n)

    # Web fallback: if FAISS returned too few / weak results → search the web
    trigger, reason = should_trigger_fallback(db_results=movies, n_requested=n)
    if trigger:
        log.info(f"[P3] Web fallback triggered: {reason}")
        return web_fallback_run(
            query=query,
            user_id=None,
            n=n,
            semantic_description=parsed.get('semantic_description', ''),
            db_results=movies,
        )

    explanation = _chain.invoke({
        "retrieved_movies": json.dumps(movies, indent=2),
        "query": query, "n": n,
    })

    return {
        "pipeline": "P3_RAG_only", "query": query,
        "parsed": parsed, "movies": movies, "explanation": explanation,
    }


def _parse_genres(genres_str: str) -> list:
    import ast
    try:
        r = ast.literal_eval(genres_str)
        return r if isinstance(r, list) else [str(r)]
    except Exception:
        return [genres_str] if genres_str else []


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for q in ["dark psychological thriller", "Animated family movies", "crime drama"]:
        print(f"\n{'='*55}\nQUERY: {q}")
        r = run(q, n=5)
        print(f"EXPLANATION:\n{r['explanation']}")
        for m in r['movies']:
            print(f"  [{m['faiss_score_norm']:.3f}]  {m['title']} ({m['year']})")
