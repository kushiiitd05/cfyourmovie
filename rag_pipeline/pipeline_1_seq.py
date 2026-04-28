"""
pipeline_1_seq.py
-----------------
Pipeline 1: Sequential hybrid (Mix_GPU → content fetch → LLM explanation).

Fixes applied:
  - Prebuilt movie_id→doc dict at load time — O(1) lookup vs. N×FAISS calls
  - Parsed filters (genre, director, year) now passed to mix_recommend()
  - Fallback message when filters yield fewer than n results
  - Context truncation increased to 400 chars (was 300)
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
from mix_gpu_infer import recommend

log = logging.getLogger(__name__)

# ── Load FAISS + build O(1) lookup dict ───────────────────────
log.info("[P1] Loading FAISS index...")
_embedder    = get_embedder(input_type="query")
_vectorstore = FAISS.load_local(str(INDEX_DIR), _embedder,
                                allow_dangerous_deserialization=True)
_llm         = get_llm(temperature=0.3)

# Prebuilt dict: movie_id → Document  (built once, avoids N FAISS calls per request)
log.info("[P1] Building movie_id lookup dict...")
_id_to_doc = {}
for doc_id, doc in _vectorstore.docstore._dict.items():
    mid = doc.metadata.get('movie_id')
    if mid is not None:
        _id_to_doc[int(mid)] = doc
log.info(f"[P1] Lookup dict ready: {len(_id_to_doc)} entries")

# ── Prompt ────────────────────────────────────────────────────
_SYSTEM = """You are a movie recommendation assistant for a 1682-film database (MovieLens 100K, mostly 1990s).

The recommendation engine scored and ranked these movies for this specific user.
Rankings are FINAL. Do not reorder, rescore, or add any films not listed.

User's request: {user_query}
{filter_note}
Ranked recommendations (Mix_GPU score — higher is better match):
{ranked_movies}

Instructions:
- 2-3 sentences explaining why these films suit this user
- Top 3: one sentence each drawing on genres, director, overview, themes
- If user asked for specific director/genre: state how many exact matches found, explain remaining picks
- End with one sentence noting this is a 1682-film 1990s-focused database"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", _SYSTEM),
    ("human", "Please explain these recommendations."),
])
_chain = _prompt | _llm | StrOutputParser()


# ── Main function ─────────────────────────────────────────────
def run(user_id: int, query: str, n: int = 10) -> dict:
    # Step 1: Parse
    parsed = parse_query(query)
    n      = parsed.get('n', n)

    # Step 2: Mix_GPU with filters
    mix_results = recommend(
        user_id        = user_id,
        n              = n,
        genre_filter   = parsed.get('genre_filter'),
        director_filter= parsed.get('director_filter'),
        year_min       = parsed.get('year_range', {}).get('min'),
        year_max       = parsed.get('year_range', {}).get('max'),
    )

    filter_note = ""
    if len(mix_results) < n:
        filter_note = (
            f"Note: only {len(mix_results)} films matched your filters "
            f"(requested {n}).\n"
        )
        log.info(f"[P1] user={user_id} got {len(mix_results)}/{n} after filters")

    # Step 3: Enrich via O(1) dict lookup — NO extra FAISS calls
    enriched = []
    for movie in mix_results:
        doc     = _id_to_doc.get(movie['movie_id'])
        context = doc.page_content[:400] if doc else movie.get('overview', '')
        enriched.append({**movie, 'context': context})

    # Score ALL movies against query so CE-based web fallback condition works
    # P1 movies use 'context' key (not 'overview') — pass explicitly
    from nvidia_reranker import rerank_movies as _rerank
    enriched = _rerank(parsed.get('semantic_description', query), enriched, text_key='context', top_k=None)

    # Step 4: Web fallback if Mix_GPU returned too few / semantically irrelevant results
    from web_search_fallback import should_trigger_fallback, run as web_fallback_run
    trigger, reason = should_trigger_fallback(db_results=enriched, n_requested=n)
    if trigger:
        log.info(f"[P1] Web fallback triggered: {reason}")
        return web_fallback_run(
            query=query, user_id=user_id, n=n,
            semantic_description=parsed.get('semantic_description', ''),
            db_results=enriched,
        )

    # Step 5: Explain
    explanation = _chain.invoke({
        "user_query":    query,
        "filter_note":   filter_note,
        "ranked_movies": json.dumps(enriched, indent=2),
    })

    return {
        "pipeline": "P1_Sequential", "user_id": user_id,
        "query": query, "parsed": parsed,
        "movies": enriched, "explanation": explanation,
    }


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    for uid, q in [(1, "Give me recommendations"),
                   (42, "Action movies from the 90s"),
                   (7, "Movies by Steven Spielberg")]:
        print(f"\n{'='*55}\nUSER:{uid}  QUERY:{q}")
        r = run(uid, q, n=5)
        print(f"EXPLANATION:\n{r['explanation']}")
        for m in r['movies']:
            print(f"  [{m['mix_score']:.4f}]  {m['title']} ({m['year']})")
