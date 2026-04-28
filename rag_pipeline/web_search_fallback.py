"""
web_search_fallback.py
----------------------
Fallback pipeline triggered when the database cannot satisfy a query:
  - k results are below a confidence threshold, OR
  - FAISS similarity scores are too low (no semantic match), OR
  - Mix_GPU returns fewer results than requested due to strict filters

Flow:
    raw_user_query
        ↓
    [LLM] Prompt Architect → refined search query
        ↓
    [Tavily] Web search → top movie results from the web
        ↓
    [LLM] Extract structured movie list from web results (JSON)
        ↓
    [Mix_GPU] Score each web-retrieved movie against user's taste profile
        ↓
    Re-rank by Mix_GPU score (personalised web results)
        ↓
    [LLM] Natural language explanation
        ↓
    Output with clear "web-sourced" label

Trigger conditions (checked by caller — pipeline_2_dual.py or pipeline_4_hyde.py):
    1. len(db_results) < MIN_RESULTS_THRESHOLD
    2. max(faiss_scores) < FAISS_CONFIDENCE_THRESHOLD
    3. all(mix_score < MIX_CONFIDENCE_THRESHOLD for m in db_results)

Design notes:
    - LLM never scores. Mix_GPU scores. LLM only refines query and explains.
    - Web results are passed through Mix_GPU scoring so personalisation is preserved.
    - Results are clearly labelled "web-sourced" — no confusion with DB results.
    - Tavily free tier: 1000 calls/month. Get key at tavily.com (no credit card).
    - If TAVILY_API_KEY is not set, falls back to a graceful "not found" message.

Cite as:
    Inspired by LLM-based RS agent design (RecSys 2024, ACM DL 10.1145/3705328.3759334)
    and HyDE retrieval (Gao et al., CMU 2022).
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import get_llm, GROQ_API_KEY, GROQ_MODEL, NVIDIA_API_KEY, NVIDIA_BASE_URL, NVIDIA_CHAT_MODEL

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

log = logging.getLogger(__name__)

# ── Trigger thresholds (tune these) ──────────────────────────
MIN_RESULTS_THRESHOLD    = 3      # fewer than this → trigger fallback
FAISS_CONFIDENCE_THRESHOLD = 0.30 # below this cosine sim → weak semantic match
MIX_CONFIDENCE_THRESHOLD   = -0.5 # below this z-score → CF has no opinion

# ── LLM setup ─────────────────────────────────────────────────
_llm = get_llm(temperature=0.3)
_llm_json = get_llm(temperature=0, json_mode=True)

# ── Groq: fast, reliable — used for architect query + extraction ──

def _groq_client():
    from groq import Groq
    return Groq(api_key=GROQ_API_KEY)


def _groq_architect_query(query: str, semantic: str, constraint_notes: str = "") -> str:
    """Use Groq to craft a clean 5-12 word web search query. Avoids local LLM looping."""
    if not GROQ_API_KEY:
        return query[:150]
    try:
        constraint_ctx = f"\nIMPORTANT constraint to preserve: {constraint_notes}" if constraint_notes else ""
        resp = _groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a movie search query architect. "
                    "Output ONLY a 5-12 word search query string. Nothing else. No quotes. "
                    "Preserve any regional/language/cultural constraints from the user's request."
                )},
                {"role": "user", "content": f"User request: {query}\nContext: {semantic}{constraint_ctx}"},
            ],
            temperature=0,
        )
        sq = resp.choices[0].message.content.strip().split('\n')[0][:150]
        return sq
    except Exception as e:
        log.warning(f"[web_fallback] Groq architect failed: {e}")
        return query[:150]


def _groq_extract_movies(snippets: str, n: int) -> list[dict]:
    """Groq llama-3.3-70b extracts structured movie list — reliable JSON output."""
    if not GROQ_API_KEY:
        return []
    try:
        system = (
            f"Extract up to {n} movies from web snippets. "
            'Return ONLY a JSON object: {"movies":[{"title":"...","year":2023,"director":"...","genres":"...","overview":"...","source_url":"..."}]}'
        )
        resp = _groq_client().chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Snippets:\n{snippets[:3000]}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        import json as _json
        raw = resp.choices[0].message.content
        parsed = _json.loads(raw)
        # Groq returns {"movies": [...]} or the array directly
        if isinstance(parsed, list):
            movies = parsed
        elif isinstance(parsed, dict):
            movies = next((v for v in parsed.values() if isinstance(v, list)), [])
        else:
            movies = []
        return [m for m in movies if isinstance(m, dict) and m.get("title")][:n]
    except Exception as e:
        log.warning(f"[web_fallback] Groq extraction failed: {e}")
        return []


# ── STEP 1: Prompt Architect ──────────────────────────────────
# Refines the raw user query into an optimised web search query.
# Inspired by HyDE concept: generate a query that reads like what a
# movie review site would say, not what a user typed.

_ARCHITECT_SYSTEM = """You are a search query architect for a movie recommendation system.

The user's request could not be satisfied from our internal 1682-film database.
Your job: craft ONE precise web search query that will find the best matching movies
on the internet (IMDb, Letterboxd, Rotten Tomatoes, etc.).

Rules:
- The query should be 5–12 words
- Frame it as a search a movie reviewer or film site would use
- Include genre, tone, era, director if mentioned by user
- Do NOT include words like "recommend" or "suggest" — search engines work better with descriptive queries
- Output ONLY the search query string. Nothing else.

Example input:  "I want something like Inception but more emotional and slower paced"
Example output: "cerebral emotional science fiction films slow burn psychological drama"""

_architect_prompt = ChatPromptTemplate.from_messages([
    ("system", _ARCHITECT_SYSTEM),
    ("human", "Raw user query: {query}\nSemantic context: {semantic}"),
])
_architect_chain = _architect_prompt | _llm | StrOutputParser()


# ── STEP 2: Tavily web search ─────────────────────────────────
def _tavily_search(query: str, n: int = 10) -> list[dict]:
    """
    Search the web for movies matching the query using Tavily.
    Returns list of {title, url, content} dicts.
    Falls back to empty list if API key not set.
    """
    api_key = os.environ.get("TAVILY_API_KEY", "")
    if not api_key:
        log.warning("[web_fallback] TAVILY_API_KEY not set — web search disabled")
        return []

    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query + " movie recommendations list",
            max_results=min(n + 5, 10),
            search_depth="basic",
            topic="general",
        )
        results = response.get("results", [])
        log.info(f"[web_fallback] Tavily returned {len(results)} results for: {query!r}")
        return results

    except ImportError:
        log.error("[web_fallback] tavily-python not installed. Run: pip install tavily-python")
        return []
    except Exception as e:
        log.error(f"[web_fallback] Tavily search failed: {e}")
        return []


# ── STEP 3: Extract structured movie list from web results ────
_EXTRACTOR_SYSTEM = """You are a movie data extractor.

You will receive web search snippets about movies. Extract a structured list of movies mentioned.
Output ONLY a valid JSON array — no preamble, no markdown.

Each item in the array must have exactly these fields:
{{
  "title": "<movie title>",
  "year": <integer year or null>,
  "director": "<director name or empty string>",
  "genres": "<comma-separated genres or empty string>",
  "overview": "<1-2 sentence description from the snippet>",
  "source_url": "<URL where this was found>"
}}

Rules:
- Include at most {n} movies
- Only include movies clearly mentioned with a title
- Do not invent plot details not present in the snippets
- If year or director not mentioned, use null / empty string
- Deduplicate — same movie appearing in multiple snippets = one entry"""

_extractor_prompt = ChatPromptTemplate.from_messages([
    ("system", _EXTRACTOR_SYSTEM),
    ("human", "Web search snippets:\n{snippets}\n\nExtract up to {n} movies."),
])
_extractor_chain = _extractor_prompt | _llm_json | JsonOutputParser()


def _extract_movies_from_web(web_results: list[dict], n: int) -> list[dict]:
    """Parse Tavily results into a structured movie list using LLM."""
    if not web_results:
        return []

    snippets = "\n\n".join([
        f"[{r.get('url', '')}]\n{r.get('content', '')[:400]}"
        for r in web_results
    ])

    # Prefer Groq (reliable JSON, fast) — fall back to NVIDIA NIM chain via LLM
    groq_result = _groq_extract_movies(snippets, n)
    if groq_result:
        log.info(f"[web_fallback] Groq extracted {len(groq_result)} movies")
        return groq_result

    try:
        extracted = _extractor_chain.invoke({"snippets": snippets, "n": n})
        # JsonOutputParser may return dict or list depending on LLM output shape
        if isinstance(extracted, list):
            movies = extracted
        elif isinstance(extracted, dict):
            # Check if it's a wrapper: {"movies": [...]} or similar
            list_vals = [v for v in extracted.values() if isinstance(v, list)]
            if list_vals:
                movies = list_vals[0]
            elif extracted.get("title"):
                # Small LLM returned single movie object instead of array — wrap it
                movies = [extracted]
            else:
                movies = []
        else:
            movies = []
        log.info(f"[web_fallback] Extracted {len(movies)} movies from web results")
        return [m for m in movies if isinstance(m, dict) and m.get("title")][:n]
    except Exception as e:
        log.warning(f"[web_fallback] Movie extraction failed: {e}")
        return []


# ── STEP 4: LLM Explanation ───────────────────────────────────
_EXPLAIN_SYSTEM = """You are a movie recommendation assistant.

The user's query could not be fully satisfied from our internal 1682-film database
(MovieLens 100K, mostly 1990s films). The following movies were sourced from the web
and ranked by how well they match the user's known taste profile.

Web-sourced recommendations (ranked by personalised score):
{movies}

User's original query: {query}
Refined search query used: {search_query}

Instructions:
- Start with ONE sentence acknowledging the database limitation honestly
- Then naturally introduce these web-sourced recommendations
- For top 3: one sentence each explaining the match
- Mention that these are web recommendations and may not be in our local database
- Do NOT claim these are personalised the same way our DB recommendations are
- Keep it natural, friendly, and concise (4-6 sentences total)"""

_explain_prompt = ChatPromptTemplate.from_messages([
    ("system", _EXPLAIN_SYSTEM),
    ("human", "Explain these web-sourced recommendations."),
])
_explain_chain = _explain_prompt | _llm | StrOutputParser()


# ── Public trigger checker ────────────────────────────────────
def should_trigger_fallback(
    db_results: list[dict],
    faiss_scores: Optional[dict] = None,
    n_requested: int = 10,
) -> tuple[bool, str]:
    """
    Decide whether to trigger the web search fallback.

    Args:
        db_results    : movies returned from the main pipeline
        faiss_scores  : {movie_id: float} FAISS similarity scores (optional)
        n_requested   : number the user asked for

    Returns:
        (should_trigger: bool, reason: str)
    """
    # Condition 1: too few results (got less than what was requested)
    if len(db_results) < n_requested:
        return True, f"only {len(db_results)}/{n_requested} results from database"

    # Condition 2: weak FAISS confidence (no semantic match)
    if faiss_scores:
        max_faiss = max(faiss_scores.values(), default=0.0)
        if max_faiss < FAISS_CONFIDENCE_THRESHOLD:
            return True, f"max FAISS similarity {max_faiss:.3f} < threshold {FAISS_CONFIDENCE_THRESHOLD}"

    # Condition 3: all Mix_GPU scores are very weak
    if db_results:
        mix_scores = [m.get("mix_score", 0) or m.get("fused_score", 0)
                      for m in db_results]
        if mix_scores and max(mix_scores) < MIX_CONFIDENCE_THRESHOLD:
            return True, f"best mix_score {max(mix_scores):.3f} < threshold {MIX_CONFIDENCE_THRESHOLD}"

    # Condition 4: cross-encoder says results are semantically irrelevant
    # BAAI/llama-nemotron-rerank logits: typical range -6 to +6.
    # Trigger if max score < -2.0 — indicates DB returned plausible-looking but wrong results
    if db_results:
        rerank_scores = [
            s for m in db_results
            for s in [m.get("nvidia_rank_score")]
            if s is not None and s == s  # s == s is False for nan
        ]
        if rerank_scores and max(rerank_scores) < -2.0:
            return True, f"cross-encoder: max rerank score {max(rerank_scores):.3f} < -2.0 (semantic mismatch)"

    return False, ""


# ── Main public function ──────────────────────────────────────
def run(
    query: str,
    user_id: Optional[int] = None,
    n: int = 10,
    semantic_description: str = "",
    db_results: Optional[list[dict]] = None,
    constraint_notes: str = "",
) -> dict:
    """
    Web search fallback pipeline.

    Args:
        query                : raw user query
        user_id              : 1-indexed (None = cold start)
        n                    : number of results requested
        semantic_description : enriched query from the parser (used as search hint)
        db_results           : partial DB results to merge with (can be empty list)

    Returns:
        {
          "pipeline":      "WEB_FALLBACK",
          "user_id":       int or None,
          "query":         str,
          "search_query":  str,   ← the refined Tavily query
          "movies":        list,  ← web-retrieved + Mix_GPU scored
          "db_movies":     list,  ← whatever the DB returned (may be empty)
          "explanation":   str,
          "web_sourced":   True,
        }
    """
    log.info(f"[web_fallback] Triggered for user={user_id} query={query!r}")

    # Step 1: Craft refined search query — Groq preferred, local LLM fallback
    if GROQ_API_KEY:
        search_query = _groq_architect_query(query, semantic_description or query, constraint_notes)
    else:
        raw_sq = _architect_chain.invoke({"query": query, "semantic": semantic_description or query}).strip()
        search_query = raw_sq.split('\n')[0][:150].strip()
    log.info(f"[web_fallback] Refined query: {search_query!r}")

    # Step 2: Web search via Tavily
    web_results = _tavily_search(search_query, n=n)

    # Step 3: Extract structured movie list
    web_movies = _extract_movies_from_web(web_results, n=n)

    if not web_movies:
        # Tavily key missing or search returned nothing
        explanation = (
            f"Your query '{query}' could not be matched in our 1682-film database, "
            f"and the web search did not return usable results. "
            f"Try a broader query, or check that TAVILY_API_KEY is set in your environment."
        )
        return {
            "pipeline":    "WEB_FALLBACK",
            "user_id":     user_id,
            "query":       query,
            "search_query": search_query,
            "movies":      db_results or [],
            "db_movies":   db_results or [],
            "explanation": explanation,
            "web_sourced": True,
        }

    # Step 4: Score via Mix_GPU (personalise web results)
    from web_movie_scorer import score_web_movies_proper
    scored_movies = score_web_movies_proper(user_id, web_movies)

    # Step 5: Explain
    explanation = _explain_chain.invoke({
        "movies":       json.dumps(scored_movies[:5], indent=2),
        "query":        query,
        "search_query": search_query,
    })

    return {
        "pipeline":     "WEB_FALLBACK",
        "user_id":      user_id,
        "query":        query,
        "search_query": search_query,
        "movies":       scored_movies,
        "db_movies":    db_results or [],
        "explanation":  explanation,
        "web_sourced":  True,
    }


# ── Integration patch for pipeline_2_dual.py and pipeline_4_hyde.py ──
# Add this snippet inside pipeline_2_dual.run() and pipeline_4_hyde.run()
# AFTER the fused results are computed, BEFORE the LLM explanation:
#
#   from web_search_fallback import should_trigger_fallback, run as web_fallback_run
#
#   trigger, reason = should_trigger_fallback(
#       db_results=movies,
#       faiss_scores=faiss_scores,
#       n_requested=n,
#   )
#   if trigger:
#       log.info(f"[Pipeline] Web fallback triggered: {reason}")
#       return web_fallback_run(
#           query=query,
#           user_id=user_id,
#           n=n,
#           semantic_description=parsed.get("semantic_description", ""),
#           db_results=movies,
#       )


# ── Test ──────────────────────────────────────────────────────
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    # Test 1: query that won't match DB well
    print("\n" + "="*60)
    print("TEST 1: Obscure foreign film query")
    result = run(
        query="recent Korean psychological thriller like Parasite",
        user_id=1,
        n=5,
    )
    print(f"Search query used: {result['search_query']}")
    print(f"Web movies found:  {len(result['movies'])}")
    print(f"\nEXPLANATION:\n{result['explanation']}")
    for m in result["movies"]:
        print(f"  [{m.get('mix_score', 0):.3f}] {m['title']} ({m.get('year')}) "
              f"[{'DB' if m.get('in_db') else 'WEB'}]")

    # Test 2: should_trigger_fallback check
    print("\n" + "="*60)
    print("TEST 2: Trigger detection")
    mock_results = [{"title": "Toy Story", "mix_score": -0.8}]
    trigger, reason = should_trigger_fallback(mock_results, n_requested=10)
    print(f"Trigger: {trigger}  Reason: {reason}")
