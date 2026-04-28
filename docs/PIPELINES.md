# Pipeline Design Reference

Five pipeline tiers, each building on the previous. Each serves a different purpose in the ablation study.

---

## P1 — Sequential (Baseline)

**Role:** Pure CF baseline with LLM explanation. No retrieval.

```
User query
     │
     ▼
 query_parser.py  →  {genre_filter, year_range, n}
     │
     ▼
 mix_gpu_infer.recommend(user_id, n=20)
     │
     ▼
 Fetch metadata for top-20 from Master_final.csv
     │
     ▼
 Groq LLM → ranked explanation
```

**Score formula:** `final = mix_score`

**Use when:** Evaluating pure CF signal strength. Fastest pipeline (2 LLM calls).

**Weakness:** No semantic query understanding. Cannot handle "something like X but Y" queries.

---

## P2 — Dual (Standard)

**Role:** CF + dense semantic retrieval, Z-norm fused.

```
User query
     │
     ├─────────────────────┐
     ▼                     ▼
 mix_gpu CF          FAISS dense retrieval
 (personalized)      (semantic embedding)
     │                     │
     │              nvidia_reranker.py
     │              (before fusion — P2 pattern)
     │                     │
     └────── Z-norm fusion ─┘
         α·z(mix) + β·z(faiss)
               │
         Groq LLM → explanation
```

**Score formula:** `final = α·z_norm(mix) + β·z_norm(faiss_dense)`

**Reranker position:** After FAISS, before CF fusion.

**Weakness:** Dense-only FAISS misses keyword queries ("Spielberg films"). Single query = single semantic angle.

---

## P3 — RAG-only (Ablation Control)

**Role:** Pure retrieval baseline — no CF signal. Used to isolate the value of personalization.

```
User query
     │
     ▼
 FAISS dense retrieval
     │
     ▼
 cross-encoder rerank on final list
     │
     ▼
 Groq LLM → explanation
```

**Score formula:** `final = faiss_dense_score`

**Use when:** Measuring retrieval quality independent of user preference. Ablation baseline.

**Key insight from ablation:** P3 achieves lower MAP@10 than P2 — proves CF signal adds measurable value.

---

## P4 — HyDE (Primary Research Contribution)

**Role:** Hypothetical Document Embeddings for zero-shot query enrichment.

Based on [Gao et al. 2022](https://arxiv.org/abs/2212.10496): embed a *hypothetical ideal document* instead of the raw query.

```
User query
     │
     ▼
 query_parser.py → {structured_query + hypothetical_ideal}
     │
     ├─────────────────────┐
     ▼                     ▼
 mix_gpu CF          HyDE generation (LLM call #2)
                     │
                     ▼
                "A cerebral 1990s thriller
                 with unreliable narrator
                 and claustrophobic atmosphere..."
                     │
                     ▼
                NVIDIA NIM embedding of HyDE text
                     │
                     ▼
                FAISS retrieval (30 candidates)
                     │
                nvidia_reranker.py (before fusion)
                     │
     └────── Z-norm fusion ─┘
         α·z(mix) + β·z(hyde_faiss)
               │
         Groq LLM → explanation
```

**Score formula:** `final = α·z_norm(mix) + β·z_norm(faiss_hyde)`

**LLM calls:** 3 (parser + HyDE generation + synthesis)

**Why HyDE works:** Vocabulary gap between "something sadder than Inception" and indexed film metadata. HyDE bridges it by generating text in the metadata's vocabulary domain.

**Limitation:** Still single dense vector → misses exact keyword matches.

---

## P5 — HybridDeep (Production System)

**Role:** Full system — HyDE + BM25 + sub-query decomposition + post-fusion reranking.

Solves P4's keyword gap by adding BM25 retrieval and decomposing complex queries.

```
User query
     │
     ▼
 query_parser.py → structured JSON
     │
     ▼
 Sub-query decomposition (LLM call #2)
 "Spielberg Cold War thriller with strong female lead"
     → sub_q1: "Steven Spielberg historical thriller film"
     → sub_q2: "Cold War spy drama female protagonist"
     │
     ├─────────────────────────────────┐
     │                                 │
     ▼                                 ▼
 mix_gpu CF                    Hybrid Retriever
                                for each sub-query:
                                  ├─ BM25 keyword search
                                  └─ FAISS dense (HyDE embed)
                                        │
                                   RRF merge (k=60)
                                   per-query → unified rank
                                   across sub-queries → RRF again
                                        │
     └──────────── Z-norm fusion ───────┘
           α·z(mix) + β·z(rrf_bm25_faiss)
           α=0.80, β=0.20
                       │
            nvidia_reranker.py
            (AFTER fusion — P5 innovation)
                       │
            web_search_fallback.py
            (if confidence < threshold)
                       │
            Groq LLM → ranked explanation
```

**Score formula:** `final = α·z_norm(mix) + β·z_norm(rrf_bm25_faiss)` then cross-encoder rerank

**LLM calls:** 4 (parser + sub-query decomp + HyDE generations + synthesis)

**Candidate pool:** ~64 (vs ~30 for P2/P4) — larger because multi-sub-query retrieval

**Key architectural innovations over P4:**
1. BM25 handles keyword queries that dense retrieval fails
2. Sub-query decomposition handles multi-intent queries
3. Reranker moved **post-fusion** — scores the final fused ranking, not an intermediate FAISS ranking
4. Adaptive α per sub-query (keyword-heavy sub-query gets higher BM25 weight)

---

## Pipeline Comparison Matrix

| | P1 | P2 | P3 | P4 | P5 |
|---|---|---|---|---|---|
| CF Signal | Only | Yes | No | Yes | Yes |
| FAISS | No | Dense | Dense | HyDE embed | HyDE embed |
| BM25 | No | No | No | No | Yes |
| RRF | No | No | No | No | Yes |
| HyDE | No | No | No | Yes | Yes |
| Sub-queries | No | No | No | No | Yes |
| Reranker | No | Before fusion | After CF | Before fusion | **After fusion** |
| Web Fallback | Yes | Yes | Yes | Yes | Yes |
| Candidate Pool | 20 | ~30 | ~30 | ~30 | ~64 |
| LLM Calls | 2 | 2 | 2 | 3 | 4 |
| Latency | Fastest | Fast | Fast | Slow | Slowest |
| Keyword Queries | Weak | Weak | Weak | Weak | Strong |
| Multi-intent | None | None | None | Partial | Best |
| Needs user_id | Yes | Yes | No | Yes | Yes |

---

## Choosing a Pipeline

```
Need fastest response?          → P1
Pure retrieval (no user data)?  → P3
Standard personalized query?    → P2 or P4
Keyword-heavy query?            → P5
Complex multi-intent query?     → P5
Production default?             → P5 (or auto-route via recommend.py)
```

`recommend.py` implements a global router that inspects the query and selects a pipeline automatically.
