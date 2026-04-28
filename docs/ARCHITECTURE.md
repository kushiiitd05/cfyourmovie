# System Architecture

## Overview

CfYourMovie is a two-tower recommendation architecture: a personalized **Collaborative Filtering (CF) tower** and a semantic **RAG retrieval tower**, fused by Z-normalized score combination, post-fusion cross-encoder reranking, and a Tavily web search fallback.

---

## Layer 1 — Data Layer

### Master_final.csv — The Enriched Dataset

Raw MovieLens 100K has 6.3% rating density and minimal metadata. The data engineering phase created `Master_final.csv` — a 99.5% complete enriched dataset:

| Source | Fields Added |
|---|---|
| MovieLens 100K | `movie_id`, `user_id`, `rating`, `title`, `genres` (pipe-separated) |
| IMDb Python API | `director`, `top_cast`, `runtime` |
| TMDB API | `overview`, `movie_keywords`, `vote_average`, `vote_count`, `popularity` |
| Kaggle dataset | `budget`, `revenue` (~50% complete — excluded from FAISS docs) |
| LLM web scraping | Gap-fill for remaining missing fields |

EDA findings that shaped the model:
- Rating distribution skews toward 3–4 stars (users rarely give extremes)
- Temporal decay matters: recency of ratings correlates with current preference
- Genre popularity is highly uneven — drama/comedy dominate; warps coverage if uncorrected
- Cold-start: 218 users have fewer than 20 ratings — EASE handles better than LightFM here

---

## Layer 2 — CF Tower (Mix_GPU)

### EASE Model

EASE ([Steck 2019](https://arxiv.org/abs/1905.03375)) is a closed-form regularized autoencoder. Given user-item interaction matrix X:

```
B* = I − P · diag(1/diag(P))
P = (XᵀX + λI)⁻¹
```

No SGD. No gradient instability. One matrix inversion. Empirically superior MAP@10 vs LightFM on this dataset.

### Three Fused Similarity Signals

**1. Structural Similarity** (`alpha_struct = 0.100`)
- Genre overlap (Jaccard), director match, year proximity
- Fast — pure Pandas computation over Master_final.csv

**2. Embedding Similarity** (`alpha_embed = 0.037`)
- `sentence-transformers/all-MiniLM-L6-v2` on concatenated `[title, genres, director, overview]`
- Cosine similarity in 384-dim embedding space
- Runs once at build time — cached as numpy array

**3. Temporal Decay**
- Half-life rating recency weighting: `weight = 2^(-Δdays / half_life_days)`
- `half_life_days = 218` (Optuna-tuned)
- Upweights recently-active ratings; downweights stale preferences

### Optuna Hyperparameter Search

200+ trials on NVIDIA A100 (IIIT Delhi server):

```
Objective: maximize nDCG@10 on 5-fold CV validation fold
Search space:
  lambda_ease:      LogUniform(10, 5000)
  half_life_days:   IntUniform(30, 730)
  negative_penalty: Uniform(0.5, 1.0)
  alpha_struct:     Uniform(0.0, 0.5)
  alpha_embed:      Uniform(0.0, 0.3)

Best found:
  lambda_ease:      2199.94
  half_life_days:   218
  negative_penalty: 0.967
  alpha_struct:     0.100
  alpha_embed:      0.037
```

### CF Output

```python
mix_gpu_infer.recommend(user_id, n=100)
# Returns: [(movie_id, mix_score), ...]   Z-normalized float scores
```

---

## Layer 3 — RAG Tower

### FAISS Dense Index

- Embedding model: NVIDIA NIM `llama-nemotron-embed-1b-v2`
  - 8192 token context, asymmetric `input_type` (passage vs query)
  - Hosted on NVIDIA AI Foundation Endpoints
- Documents: 1682 movie entries — `{title} ({year}) | {genres} | {director} | {overview} | keywords: {keywords}`
- Index type: `IndexFlatIP` (inner product, L2-normalized → cosine similarity)
- Stored as: `rag_pipeline/faiss_index/`

### BM25 Keyword Index

- `rank_bm25` — BM25Okapi implementation
- Indexed fields: title, director, cast, genres, keywords
- Serialized to `rag_pipeline/bm25_index/bm25.pkl`
- Critical for name-based queries: "Spielberg films", "Tom Hanks comedy", "Jurassic Park sequels"

### Hybrid Retrieval (RRF)

Reciprocal Rank Fusion ([Cormack et al. 2009](https://dl.acm.org/doi/10.1145/1571941.1572114)):

```python
RRF_score(d) = Σ 1 / (k + rank_i(d))   # k=60, standard RRF constant
```

Merges BM25 rank list + FAISS rank list → unified semantic+keyword ranking.

### Query Parser

`query_parser.py` converts natural language → structured JSON via Groq `llama-3.3-70b-versatile`:

```json
{
  "genre_filter": ["thriller", "drama"],
  "director_filter": null,
  "year_range": [1980, 1999],
  "n": 10,
  "semantic_description": "slow-burn psychological tension, unreliable narrator",
  "hypothetical_ideal": "A cerebral 1990s thriller where the protagonist discovers..."
}
```

---

## Layer 4 — Fusion & Reranking

### Z-Norm Score Fusion

Both CF and RAG scores are Z-normalized before fusion to put them on a common scale:

```python
z_mix  = (mix_score  - μ_mix)  / σ_mix
z_rag  = (rag_score  - μ_rag)  / σ_rag
fused  = α·z_mix + β·z_rag    # α=0.80, β=0.20
```

α:β ratio is Optuna-tuned. CF dominates (0.80) because it captures personalization signal that RAG cannot.

### Cross-Encoder Reranker

`cross-encoder/ms-marco-MiniLM-L-6-v2` applied **post-fusion** (architectural innovation vs P2/P4 where it ran before fusion):

```
Input:  (query, movie_metadata_string) pairs
Output: relevance logit per pair
```

Applied to top-N fused candidates → final ranked list.

### Web Search Fallback

Triggered when:
- `len(db_results) < MIN_RESULTS_THRESHOLD` (3)
- `max(faiss_score) < FAISS_CONFIDENCE_THRESHOLD` (0.30)
- `max(mix_score) < MIX_CONFIDENCE_THRESHOLD` (-0.5)

Tavily search fetches real-time results → `web_movie_scorer.py` scores them via CF model → merged with DB results.

---

## Layer 5 — LLM Synthesis

Cascaded LLM chain (ordered by speed/cost):

```
1. Groq llama-3.3-70b-versatile       → JSON mode, fastest
2. Gemini 2.0 Flash                    → fallback (rate limit)
3. NVIDIA NIM nemotron-super-120b      → 1M context fallback
4. OpenRouter (free models)            → last resort
```

Each LLM generates a ranked list with movie-level explanations referencing the user's query phrasing.

---

## Two-Model Embedding Design

Important architectural constraint:

| Use | Model | Reason |
|---|---|---|
| FAISS retrieval | NVIDIA nemotron-embed-1b-v2 | High-quality asymmetric embeddings for retrieval |
| Mix_GPU CB component | all-MiniLM-L6-v2 | Pre-trained content similarity, offline, fast |

These operate in **separate vector spaces**. Fusion uses scalar Z-normalized scores — no vector space conflicts.

---

## Hardware Split

| Stage | Hardware | Why |
|---|---|---|
| Optuna hyperparameter search | NVIDIA A100 (IIIT Delhi) | 200+ trials × 5-fold CV |
| EASE matrix factorization | A100 → saved as `.npy` | One-time closed-form solve |
| FAISS index build | A100 (NVIDIA NIM embeddings via API) | 1682 × 1b embedding calls |
| Inference (pipeline serving) | Mac M2 Pro | All indices loaded in RAM, fast |
| Cross-encoder reranker | Mac M2 Pro (CPU) | Small model, acceptable latency |
