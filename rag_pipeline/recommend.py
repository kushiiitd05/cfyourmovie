"""
recommend.py
------------
Global router — single entry point for all 5 pipelines.

Usage:
    from recommend import recommend

    # Auto-select pipeline:
    result = recommend("dark psychological thriller", user_id=1)

    # Force a specific pipeline:
    result = recommend("dark psychological thriller", user_id=1, pipeline="P5")

    # Cold-start (no user_id → P3):
    result = recommend("crime drama with gangsters")

Pipeline selection logic (auto mode):
  - No user_id         → P3 (RAG-only, content similarity)
  - Simple query       → P1 (Mix_GPU sequential, fastest)
  - Standard query     → P2 (Dual-engine, Mix_GPU + FAISS fused)
  - Complex/rich query → P4 (HyDE hybrid, hypothetical doc expansion)

Pipeline overview:
  P1 — Sequential Mix_GPU       (fastest, CF-only)
  P2 — Dual-engine CF+FAISS     (balanced, personalized)
  P3 — RAG cold-start           (content-only, no user_id needed)
  P4 — HyDE hybrid              (best for complex/stylistic queries)
  P5 — Hybrid Deep (BEST)       (BM25+FAISS+HyDE+sub-queries+post-fusion rerank)
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger(__name__)

# ── Lazy pipeline imports (load on first use) ─────────────────
_P1 = _P2 = _P3 = _P4 = _P5 = None

def _load_pipeline(name: str):
    global _P1, _P2, _P3, _P4, _P5
    if name == "P1" and _P1 is None:
        import pipeline_1_seq as m; _P1 = m
    elif name == "P2" and _P2 is None:
        import pipeline_2_dual as m; _P2 = m
    elif name == "P3" and _P3 is None:
        import pipeline_3_rag as m; _P3 = m
    elif name == "P4" and _P4 is None:
        import pipeline_4_hyde as m; _P4 = m
    elif name == "P5" and _P5 is None:
        import pipeline_5_hybrid_deep as m; _P5 = m
    return {"P1": _P1, "P2": _P2, "P3": _P3, "P4": _P4, "P5": _P5}[name]


# ── Query complexity heuristic ────────────────────────────────
_HYDE_TRIGGERS = {
    "like", "similar to", "but more", "but less", "meets", "reminiscent",
    "à la", "ala", "vibe", "feel of", "style of", "tone of", "atmosphere of",
    "inspired by", "in the spirit of",
}

def _is_complex_query(query: str) -> bool:
    """Return True if query benefits from HyDE hypothetical expansion."""
    q = query.lower()
    if len(query) > 60:
        return True
    return any(t in q for t in _HYDE_TRIGGERS)


def _select_pipeline(query: str, user_id) -> str:
    """Return pipeline name based on query and user context."""
    if user_id is None:
        return "P3"
    if _is_complex_query(query):
        return "P4"
    # Short query with user → P2 (dual-engine, balanced)
    return "P2"


# ── Main entry point ──────────────────────────────────────────
def recommend(
    query: str,
    user_id: int = None,
    pipeline: str = "auto",
    n: int = 10,
    alpha: float = None,
    beta: float = None,
) -> dict:
    """
    Recommend movies for a query.

    Args:
        query    : Natural language request
        user_id  : MovieLens user ID (1-943). None = cold-start → P3.
        pipeline : "auto" | "P1" | "P2" | "P3" | "P4"
        n        : Number of recommendations to return
        alpha    : Mix_GPU weight for P2/P4 (None = use Optuna default)
        beta     : FAISS weight for P2/P4 (None = use Optuna default)

    Returns:
        dict with keys: pipeline, query, movies, explanation, (+ pipeline-specific keys)
    """
    if not query or not query.strip():
        return {"pipeline": "none", "query": query, "movies": [],
                "explanation": "Empty query."}

    chosen = pipeline if pipeline != "auto" else _select_pipeline(query, user_id)
    chosen = chosen.upper()

    if chosen not in ("P1", "P2", "P3", "P4", "P5"):
        raise ValueError(f"Unknown pipeline '{chosen}'. Use P1–P5 or 'auto'.")

    requested_n = n  # save before pipelines can override via query_parser
    log.info(f"[recommend] pipeline={chosen} user_id={user_id} query='{query[:60]}'")

    mod = _load_pipeline(chosen)

    def _slice(r: dict) -> dict:
        r["movies"] = r["movies"][:requested_n]
        return r

    # Build kwargs — each pipeline has slightly different signature
    if chosen == "P1":
        if user_id is None:
            log.warning("[recommend] P1 requires user_id — falling back to P3")
            mod = _load_pipeline("P3")
            return _slice(mod.run(query=query, n=n))
        return _slice(mod.run(user_id=user_id, query=query, n=n))

    elif chosen == "P2":
        if user_id is None:
            log.warning("[recommend] P2 requires user_id — falling back to P3")
            mod = _load_pipeline("P3")
            return _slice(mod.run(query=query, n=n))
        kwargs = {"user_id": user_id, "query": query, "n": n}
        if alpha is not None: kwargs["alpha"] = alpha
        if beta  is not None: kwargs["beta"]  = beta
        return _slice(mod.run(**kwargs))

    elif chosen == "P3":
        return _slice(mod.run(query=query, n=n))

    elif chosen == "P4":
        if user_id is None:
            log.warning("[recommend] P4 requires user_id — falling back to P3")
            mod = _load_pipeline("P3")
            return _slice(mod.run(query=query, n=n))
        kwargs = {"user_id": user_id, "query": query, "n": n}
        if alpha is not None: kwargs["alpha"] = alpha
        if beta  is not None: kwargs["beta"]  = beta
        return _slice(mod.run(**kwargs))

    else:  # P5
        if user_id is None:
            log.warning("[recommend] P5 requires user_id — falling back to P3")
            mod = _load_pipeline("P3")
            result = mod.run(query=query, n=n)
            result["movies"] = result["movies"][:requested_n]
            return result
        kwargs = {"user_id": user_id, "query": query, "n": n}
        if alpha is not None: kwargs["alpha"] = alpha
        if beta  is not None: kwargs["beta"]  = beta
        result = mod.run(**kwargs)
        result["movies"] = result["movies"][:requested_n]
        return result


# ── CLI / quick test ──────────────────────────────────────────
if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    test_cases = [
        {"query": "dark psychological thriller",        "user_id": 1},
        {"query": "something like Inception but deeper","user_id": 42},
        {"query": "feel-good 90s comedies",             "user_id": 15},
        {"query": "crime drama with gangsters",         "user_id": None},
    ]

    for tc in test_cases:
        print(f"\n{'='*60}")
        print(f"QUERY: {tc['query']}  |  user_id={tc['user_id']}")
        result = recommend(**tc, n=5)
        print(f"PIPELINE: {result['pipeline']}")
        for m in result["movies"][:5]:
            score = m.get("fused_score") or m.get("mix_score") or m.get("faiss_score_norm") or 0
            print(f"  [{score:+.3f}]  {m['title']} ({m.get('year','?')})")
        print(f"EXPLANATION: {result['explanation'][:200]}")
