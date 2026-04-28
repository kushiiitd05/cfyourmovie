"""
compare_all.py
--------------
Runs all 4 pipelines on the same test queries and prints a side-by-side
comparison table. Mix_GPU score is the ranking authority in P1/P2/P4.
P3 is RAG-only (no Mix_GPU) — ablation baseline.

Usage:
    python compare_all.py
"""
import sys, os, json, time, logging, textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.WARNING)   # suppress INFO noise during run

# ── Test queries ──────────────────────────────────────────────
QUERIES = [
    {"user_id": 1,  "query": "psychological thriller with dark atmosphere"},
    {"user_id": 42, "query": "sci-fi films about AI and technology"},
    {"user_id": 15, "query": "feel-good comedies from the 90s"},
    {"user_id": 50, "query": "crime drama with gangsters and moral corruption"},
    {"user_id": 1,  "query": "animated family movies"},
]
N = 5   # top-N per pipeline

SEP  = "═" * 78
SEP2 = "─" * 78

def _fmt_movies(movies, score_key="mix_score", alt_keys=("fused_score", "faiss_score_norm")):
    lines = []
    for i, m in enumerate(movies[:N], 1):
        score = m.get(score_key) or m.get(alt_keys[0]) or m.get(alt_keys[1]) or 0
        lines.append(f"  {i}. [{score:+.3f}] {m['title']} ({m.get('year','?')})"
                     f"  ★{m.get('vote_average', m.get('vote_avg','?'))}")
    return "\n".join(lines) if lines else "  (no results)"

def _wrap(text, width=72, indent="  "):
    return textwrap.fill(str(text), width=width, initial_indent=indent,
                         subsequent_indent=indent)

def _run_pipeline(label, fn, user_id, query, requires_user=False):
    t0 = time.time()
    if requires_user and user_id is None:
        return None, 0.0, "N/A (requires user_id — cold-start unsupported)"
    try:
        kwargs = {"query": query, "n": N}
        if user_id is not None:
            kwargs["user_id"] = user_id
        result = fn(**kwargs)
        elapsed = time.time() - t0
        return result, elapsed, None
    except Exception as e:
        return None, time.time() - t0, str(e)


def main():
    # ── Load pipelines once ───────────────────────────────────
    print("\nLoading all 4 pipelines (FAISS + matrices load once)...")
    t_load = time.time()

    import pipeline_1_seq  as P1
    import pipeline_2_dual as P2
    import pipeline_3_rag  as P3
    import pipeline_4_hyde as P4

    print(f"  Loaded in {time.time()-t_load:.1f}s\n")

    results_table = []  # list of dicts for final summary

    for qi, q_cfg in enumerate(QUERIES, 1):
        user_id = q_cfg["user_id"]
        query   = q_cfg["query"]
        uid_str = f"user={user_id}" if user_id else "cold-start"

        print(f"\n{SEP}")
        print(f"  QUERY {qi}/{len(QUERIES)}: \"{query}\"  |  {uid_str}")
        print(SEP)

        pipeline_results = {}

        # P1 — Sequential (Mix_GPU → content fetch → LLM)
        r1, t1, e1 = _run_pipeline("P1", P1.run, user_id, query, requires_user=True)
        if e1:
            print(f"\n[P1 Sequential]  ERROR: {e1}")
        else:
            mix_ok = any(m.get("mix_score", 0) != 0 for m in r1["movies"])
            print(f"\n[P1 Sequential]  {t1:.1f}s  |  Mix_GPU={'OK' if mix_ok else 'EMPTY(FAISS fallback)'}  |  LLM=OK")
            print(_fmt_movies(r1["movies"], "mix_score"))
            print(_wrap(r1["explanation"]))
            pipeline_results["P1"] = {"movies": r1["movies"], "time": t1}

        print(SEP2)

        # P2 — Dual Engine (Mix_GPU + FAISS fused)
        r2, t2, e2 = _run_pipeline("P2", P2.run, user_id, query, requires_user=True)
        if e2:
            print(f"\n[P2 DualEngine]  ERROR: {e2}")
        else:
            mix_ok2  = any(m.get("mix_score", 0) != 0 for m in r2["movies"])
            faiss_ok = any(m.get("faiss_score", 0) != 0 for m in r2["movies"])
            rerank_ok = any(m.get("nvidia_rank_score", 0.0) not in (None, 0.0) for m in r2["movies"])
            print(f"\n[P2 DualEngine]  {t2:.1f}s  |  Mix_GPU={'OK' if mix_ok2 else 'EMPTY→FAISS-only'}  |  FAISS={'OK' if faiss_ok else 'EMPTY'}  |  Reranker={'OK' if rerank_ok else 'fallback'}  |  LLM=OK")
            print(_fmt_movies(r2["movies"], "fused_score"))
            print(_wrap(r2["explanation"]))
            pipeline_results["P2"] = {"movies": r2["movies"], "time": t2}

        print(SEP2)

        # P3 — RAG only (no Mix_GPU — ablation baseline)
        r3, t3, e3 = _run_pipeline("P3", P3.run, None, query)
        if e3:
            print(f"\n[P3 RAG-only]    ERROR: {e3}")
        else:
            faiss_ok3  = len(r3["movies"]) > 0
            rerank_ok3 = any(m.get("nvidia_rank_score", 0.0) not in (None, 0.0) for m in r3["movies"])
            print(f"\n[P3 RAG-only]    {t3:.1f}s  |  FAISS={'OK' if faiss_ok3 else 'EMPTY'}  |  Reranker={'OK' if rerank_ok3 else 'fallback'}  |  LLM=OK  (ablation: no Mix_GPU)")
            print(_fmt_movies(r3["movies"], "faiss_score_norm"))
            print(_wrap(r3["explanation"]))
            pipeline_results["P3"] = {"movies": r3["movies"], "time": t3}

        print(SEP2)

        # P4 — HyDE Hybrid (hypothetical doc expansion + Mix_GPU + FAISS)
        r4, t4, e4 = _run_pipeline("P4", P4.run, user_id, query, requires_user=True)
        if e4:
            print(f"\n[P4 HyDE]        ERROR: {e4}")
        else:
            hyp       = r4.get("hypothetical", "")
            hyde_ok   = bool(hyp and len(hyp) > 30)
            mix_ok4   = any(m.get("mix_score", 0) != 0 for m in r4["movies"])
            faiss_ok4 = any(m.get("faiss_score", 0) != 0 for m in r4["movies"])
            rerank4   = any(m.get("nvidia_rank_score", 0.0) not in (None, 0.0) for m in r4["movies"])
            print(f"\n[P4 HyDE]        {t4:.1f}s  |  HyDE={'OK' if hyde_ok else 'FALLBACK'}  |  Mix_GPU={'OK' if mix_ok4 else 'EMPTY→FAISS'}  |  FAISS={'OK' if faiss_ok4 else 'EMPTY'}  |  Reranker={'OK' if rerank4 else 'fallback'}  |  LLM=OK")
            if hyp:
                print(f"  HyDE doc: {hyp[:120]}...")
            print(_fmt_movies(r4["movies"], "fused_score"))
            print(_wrap(r4["explanation"]))
            pipeline_results["P4"] = {"movies": r4["movies"], "time": t4}

        results_table.append({"query": query, "uid": uid_str, **pipeline_results})

    # ── Summary table ─────────────────────────────────────────
    print(f"\n\n{'#'*78}")
    print("  PIPELINE COMPARISON SUMMARY")
    print(f"{'#'*78}")
    print(f"\n{'Query':<42} {'P1':>6} {'P2':>6} {'P3':>6} {'P4':>6}  Overlap?")
    print(SEP2)
    for row in results_table:
        times = []
        overlaps = []
        sets = {}
        for p in ["P1","P2","P3","P4"]:
            if p in row:
                t = row[p]["time"]
                times.append(f"{t:.1f}s")
                sets[p] = {m["title"] for m in row[p]["movies"][:N]}
            else:
                times.append(" err")
        # overlap between P1 and P4 (both personalised hybrid)
        if "P1" in sets and "P4" in sets:
            ov = len(sets["P1"] & sets["P4"])
            overlap_str = f"P1∩P4={ov}/{N}"
        else:
            overlap_str = ""
        q_short = row["query"][:40]
        print(f"{q_short:<42} {times[0]:>6} {times[1]:>6} {times[2]:>6} {times[3]:>6}  {overlap_str}")

    print(f"\n{'─'*78}")
    print("  Ranking authority : Mix_GPU (P1/P2/P4) | FAISS cosine (P3 ablation)")
    print("  Reranker          : NVIDIA llama-nemotron-rerank-1b-v2 (P2/P3/P4)")
    print("  Embed             : NVIDIA llama-nemotron-embed-1b-v2")
    print("  LLM               : Groq llama-3.3-70b-versatile")
    print("  Component status  : Mix_GPU/FAISS/Reranker/HyDE shown per-run above")
    print(f"{'─'*78}\n")


if __name__ == "__main__":
    main()
