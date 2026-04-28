"""
query_parser.py
---------------
LLM-based query parser shared across all four pipelines.
Converts free-text → structured JSON dict.

Backend: Groq (llama-3.3-70b-versatile) → OpenRouter → local Ollama
Uses get_llm() from config — no direct Ollama dependency.
"""
import sys
import json
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import get_llm

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

log = logging.getLogger(__name__)

# ── System prompt — database-aware ────────────────────────────
PARSER_SYSTEM = """You are a query parser for a movie recommendation system.

DATABASE: MovieLens 100K — 1682 movies, mostly 1990s (1920–1998). English-language, Western-centric.
NO Indian, Korean, Japanese, Chinese, French, or other non-English films in the database.
NO films released after 1998.
FIELDS available: title, year, genres, director, top_cast, overview, vote_average, popularity.

Output ONLY valid JSON — no markdown, no explanation.

Schema:
{{
  "user_id": <integer or null>,
  "item_id": <integer or null>,
  "director_filter": <string or null>,
  "genre_filter": <list of strings or null>,
  "year_range": {{"min": <int or null>, "max": <int or null>}},
  "mood_keywords": <list of strings or null>,
  "intent": "recommendation" | "information" | "similar_to",
  "n": <integer, default 10>,
  "semantic_description": "<rich restatement for semantic search>",
  "hypothetical_ideal": "<one vivid sentence: genre + tone + themes — empty string if only user_id given>",
  "db_satisfiable": <true or false>,
  "constraint_notes": "<why db cannot satisfy this, or empty string if it can>"
}}

Rules:
- user_id: extract integer from "user 42" / "user_id=42" / "for user 42"
- genre_filter: use ONLY: Action, Adventure, Animation, Children's, Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War, Western
- year_range: "90s" → min=1990 max=1999 | "before 2000" → max=1999 | "recent" → min=1995
- n: extract from "top 5" / "give me 3" — default 10
- semantic_description: always a rich restatement (never empty)
- hypothetical_ideal: descriptive/mood queries only; empty string "" for bare user_id queries
- Never include real movie titles in any field
- db_satisfiable: set to FALSE if query mentions: (1) Indian/Bollywood/Telugu/Tamil/Korean/Japanese/foreign language films, (2) specific named films not in a 1920-1998 English database (e.g. Baahubali, RRR, Parasite, Inception, Avengers), (3) year_range with min > 1998, (4) "recent"/"new"/"latest" films implying post-1998
- constraint_notes: when db_satisfiable=false, explain what constraint can't be met (e.g. "Indian/Telugu films not in MovieLens 100K database" or "Baahubali (2015) post-dates the 1998 database cutoff")"""

_prompt = ChatPromptTemplate.from_messages([
    ("system", PARSER_SYSTEM),
    ("human", "{query}"),
])

_DEFAULTS = {
    "user_id": None, "item_id": None,
    "director_filter": None, "genre_filter": None,
    "year_range": {"min": None, "max": None},
    "mood_keywords": None,
    "intent": "recommendation",
    "n": 10,
    "semantic_description": "",
    "hypothetical_ideal": "",
    "db_satisfiable": True,
    "constraint_notes": "",
}


def _validate(result: dict, original_query: str) -> dict:
    for key, default in _DEFAULTS.items():
        if key not in result:
            result[key] = default
    try:
        result['n'] = int(result['n'])
    except (TypeError, ValueError):
        result['n'] = 10
    if not str(result.get('semantic_description', '')).strip():
        result['semantic_description'] = original_query
    if not isinstance(result.get('hypothetical_ideal'), str):
        result['hypothetical_ideal'] = ''
    if not isinstance(result.get('year_range'), dict):
        result['year_range'] = {"min": None, "max": None}
    if not isinstance(result.get('db_satisfiable'), bool):
        result['db_satisfiable'] = True
    if not isinstance(result.get('constraint_notes'), str):
        result['constraint_notes'] = ''
    return result


def parse_query(query: str, max_retries: int = 3) -> dict:
    """
    Parse a natural language query into structured JSON.
    Uses Groq → OpenRouter → Ollama fallback chain.
    Never raises — returns safe defaults after all retries exhausted.
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            llm   = get_llm(temperature=0, json_mode=True)
            chain = _prompt | llm | JsonOutputParser()
            raw   = chain.invoke({"query": query})
            # Handle wrapped dict e.g. {"result": {...}}
            if isinstance(raw, dict) and len(raw) == 1:
                inner = list(raw.values())[0]
                if isinstance(inner, dict):
                    raw = inner
            return _validate(raw, query)
        except Exception as e:
            last_error = e
            log.warning(f"[parser] Attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                time.sleep(1.5 * attempt)

    log.error(f"[parser] All {max_retries} attempts failed. Fallback. Error: {last_error}")
    fallback = dict(_DEFAULTS)
    fallback['semantic_description'] = query
    return fallback


# ── Test ──────────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tests = [
        "I love psychological thrillers with dark atmosphere, user 1",
        "user_id=42, recommend action movies",
        "movies similar to Schindler's List",
        "top 5 horror movies from the 90s",
        "Christopher Nolan films",
        "something slow-paced and emotional",
        "42",
    ]
    for q in tests:
        print(f"\n{'─'*55}\nINPUT:  {q}")
        r = parse_query(q)
        print(f"  user={r['user_id']}  intent={r['intent']}  genres={r['genre_filter']}  n={r['n']}")
        print(f"  semantic: {r['semantic_description'][:80]}")
        print(f"  ideal:    {r['hypothetical_ideal'][:80]}")
