<div align="center">

# CfYourMovie

### *Come Find Your Movie*

A complete intelligent recommendation system — enriched metadata, hybrid collaborative-and-content models, a full RAG pipeline, and a cinematic product UI.

*IIIT Delhi · Collaborative Filtering · 2025–26 · Group G22*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-green)](https://langchain.com)
[![FAISS](https://img.shields.io/badge/FAISS-Meta-blue)](https://github.com/facebookresearch/faiss)
[![Groq](https://img.shields.io/badge/LLM-Groq%20Llama--3.3--70B-orange)](https://groq.com)
[![EASE](https://img.shields.io/badge/CF-EASE%20%28Steck%202019%29-purple)](https://arxiv.org/abs/1905.03375)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

**nDCG@10: 0.3850 · Hit Rate@10: 86.95% · +8.0% over Microsoft Recommender benchmark**

[Architecture](#architecture) · [Results](#results) · [The 5 Pipelines](#the-5-pipelines) · [Setup](#setup) · [Team](#team)

</div>

---

## The Story in Four Acts

### Act I — The Problem

MovieLens 100K: 943 users rating 1,682 movies — one hundred thousand opinions in a matrix where **93.7% of cells are empty**. Pure collaborative filtering works, but only just. The real question is what happens when you ask the system something it has never quite been asked before: a mood, a vibe, a film that doesn't exist yet in the catalogue.

> *"Something like Inception but sadder"*
> *"A Spielberg film with a strong female lead set during the Cold War"*
> *"Dark psychological thriller with moral ambiguity — something I haven't seen"*

Our goal: a system that refuses to collapse in those moments. One that understands movies the way a knowledgeable friend would, reasons about taste across a thin rating signal, and ultimately feels like a **product** rather than an experiment.

### Act II — The Data

Raw MovieLens metadata is shallow. We rebuilt the entire corpus:

- **TMDB** → overview text, top cast, keywords, runtime, popularity, vote stats
- **OMDB** → title reconciliation, year gap-fill
- **Kaggle corpora** (46K + 126K datasets) → coverage backfill
- **LLM-assisted web retrieval** → semantic augmentation for thin descriptions

Result: `Master_final.csv` — **1,682 movies, 99.5% complete**, 26 unique genres, mean 7.7 keywords/movie, full director coverage.

### Act III — The Models

**EASE over LightFM.** Closed-form solution, no SGD instability, empirically superior MAP@10. Three fused signals: structural similarity (genre/director/year), embedding similarity (all-MiniLM-L6-v2), temporal decay (half-life recency weighting). **200+ Optuna trials on NVIDIA A100** for hyperparameter search.

Semantic similarity doesn't override CF — it *corrects* it, selectively, where CF alone is uncertain.

### Act IV — The System & Product

Model → RAG system → cinematic product UI. Query parser → FAISS dense + BM25 lexical retrieval → Reciprocal Rank Fusion → Mix GPU CF ranking → cross-encoder reranking → web fallback. React + Vite frontend with cinema-noir visual language: near-black surfaces, warm amber accents, horizontally scrollable result rails.

---

## Architecture

```
User query (natural language)
        │
        ▼
  [query_parser.py]  ──── Groq LLM → structured JSON
        │
        ├── genre_filter, director_filter, year_range
        └── semantic_description + hypothetical_ideal (HyDE)
                        │
             ┌──────────┴──────────┐
             │                     │
   [Mix_GPU CF Engine]    [RAG Retrieval Engine]
   EASE + temporal decay   BM25 Okapi + FAISS
   + embedding CB          → Reciprocal Rank Fusion
   (200+ Optuna trials)    + sub-query decomposition
             │                     │
             └──────────┬──────────┘
                        │
              [Z-norm Score Fusion]
              α·z(mix) + β·z(rag)
              α=0.80, β=0.20
                        │
              [Cross-Encoder Reranker]
              ms-marco-MiniLM-L-6-v2
              (applied POST-fusion)
                        │
              [Web Search Fallback]
              Tavily → rescored via Mix_GPU content arm
                        │
              [LLM Synthesis]
              Groq → Gemini → NVIDIA NIM → OpenRouter
                        │
                  Final ranked list + explanation
```

> Deep dive: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## Results

### CF Backbone Evolution — 5-Fold Cross-Validation

| Pipeline | Algorithm | MAP@10 | nDCG@10 | Hit Rate@10 | Diversity | Novelty |
|---|---|---|---|---|---|---|
| Gold Standard | LightFM + hyperparams | 0.2058 | 0.3360 | — | — | — |
| Platinum | LightFM + time decay + negatives | 0.2141 | 0.3458 | — | — | — |
| Diamond | EASE + Optuna fusion | 0.2438 | 0.3783 | — | 0.797 | 2.152 |
| Vibranium | EASE + graph propagation | 0.2330 | 0.3642 | — | 0.706 | 2.255 |
| Adamantium | LightGBM reranker over EASE | 0.1523 | 0.2652 | — | 0.692 | 2.376 |
| **Mix GPU Final** | **EASE + Optuna A100** | **0.2510** | **0.3850** | **0.8695** | **0.771** | **2.145** |

**+8.0% nDCG@10 over Microsoft Recommender benchmark** (0.3565 → 0.3850)

> *Increase exploration and ranking precision pushes back. Push for precision alone and recommendations collapse into an echo chamber. The question is never which metric to win — it's where to sit on the curve.*

### Mix GPU Final — Full Statistical Report

| Metric | Mean | ± Std Dev | What it means |
|---|---|---|---|
| MAP (full) | 0.2801 | ±0.0052 | Average precision over all relevant items |
| MAP@10 | 0.2510 | ±0.0067 | Precision in top-10 ranked list |
| nDCG (full) | 0.5803 | ±0.0045 | Ranking quality, full list |
| **nDCG@10** | **0.3850** | **±0.0079** | **Ranking quality, top-10** |
| **Hit Rate@10** | **0.8695** | **±0.0105** | **87% chance a relevant item is in top-10** |
| Serendipity | 0.2907 | ±0.0082 | Relevant but genuinely surprising |
| Coverage | 22.03% | ±0.37% | Catalogue breadth explored |
| Novelty | 2.145 | ±0.007 | Inverse popularity (higher = less mainstream) |
| Diversity | 0.771 | ±0.002 | Intra-list variety |

> Full per-fold breakdowns: [docs/RESULTS.md](docs/RESULTS.md)

---

## The 5 Pipelines

Each pipeline introduces exactly one architectural change over the previous — disciplined ablation, not ad-hoc iteration.

| | P1 Sequential | P2 Dual | P3 RAG-only | P4 HyDE | P5 HybridDeep |
|---|---|---|---|---|---|
| **Role** | Baseline | Standard | Ablation | Primary | Production |
| **CF Signal** | Only | Yes | **No** | Yes | Yes |
| **Retrieval** | None | FAISS dense | FAISS dense | FAISS (HyDE) | **BM25 + FAISS** |
| **Query Enrichment** | Rule-based | Rule-based | Rule-based | **HyDE** | **HyDE + sub-queries** |
| **Reranker** | None | Before fusion | After CF | Before fusion | **After fusion** |
| **LLM Calls** | 2 | 2 | 2 | 3 | 4 |
| **Candidate Pool** | 20 | ~30 | ~30 | ~30 | **~64** |
| **Keyword Queries** | Weak | Weak | Weak | Weak | **Strong** |

### Score Fusion
```
P1:  final = mix_score
P2:  final = α·z_norm(mix) + β·z_norm(faiss_dense)
P3:  final = faiss_dense_score         [no CF — ablation]
P4:  final = α·z_norm(mix) + β·z_norm(faiss_hyde)
P5:  final = α·z_norm(mix) + β·z_norm(rrf_bm25_faiss)
     α=0.80, β=0.20  (Optuna-tuned — CF is the primary signal)
```

**Key P5 innovation:** Reranker moved **post-fusion** — reranks what was already selected by both retrievers and the personalized scorer, not the raw candidate stream.

> Full pipeline rationale: [docs/PIPELINES.md](docs/PIPELINES.md)

---

## Research Contributions

### HyDE — Hypothetical Document Embeddings
Based on [Gao et al. 2022](https://arxiv.org/abs/2212.10496). Instead of embedding the raw query, generate a *hypothetical ideal film description* and embed that.

```
User:  "I want something like Inception but sadder"

HyDE:  "A cerebral science-fiction drama that uses nested dream architecture
        as a metaphor for grief, blending Nolan-esque visual ambition with
        the quiet devastation of a European character study."
```

Bridges the vocabulary gap between colloquial phrasing and structured cinematic metadata.

### Sub-Query Decomposition
For complex multi-intent queries, the LLM decomposes into 2 focused sub-queries each with an adaptive α (keyword vs semantic weight). Each runs independently through BM25+FAISS, merged via RRF.

### Best Optuna Config (200+ trials, A100)
```python
lambda_ease:      2199.94   # regularization strength
half_life_days:   218       # rating recency half-life
negative_penalty: 0.967     # downweight low-rated items
alpha_struct:     0.100     # structural similarity weight
alpha_embed:      0.037     # semantic embedding weight
```

---

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/cfyourmovie.git
cd cfyourmovie

python -m venv rag_venv && source rag_venv/bin/activate
pip install -r rag_pipeline/requirements.txt

cp .env.example .env   # fill in GROQ_API_KEY + NVIDIA_API_KEY

# Build indexes (one-time)
python -m rag_pipeline.build_index        # FAISS dense index
python -m rag_pipeline.build_bm25_index   # BM25 keyword index
python rag_pipeline/mix_gpu_train.py      # CF model

# Run the UI
cd ui && npm install && bash start.sh
```

> Full setup guide with troubleshooting: [docs/SETUP.md](docs/SETUP.md)

### Quick API Usage

```python
from rag_pipeline.pipeline_5_hybrid_deep import run

result = run(user_id=1, query="dark psychological thriller from the 90s", n=10)
print(result['explanation'])
for m in result['movies']:
    print(f"[{m['fused_score']:+.3f}] {m['title']} ({m['year']})")
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| CF Model | EASE (Steck 2019) — closed-form regularized autoencoder |
| Hyperparameter Tuning | Optuna — 200+ trials on NVIDIA A100 |
| Content Similarity | `sentence-transformers/all-MiniLM-L6-v2` |
| Dense Embeddings | NVIDIA NIM `llama-nemotron-embed-1b-v2` |
| Vector Store | FAISS + LangChain |
| Keyword Retrieval | `rank_bm25` — BM25Okapi |
| Score Fusion | Reciprocal Rank Fusion + Z-norm weighted sum |
| Cross-Encoder Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM — Primary | Groq `llama-3.3-70b-versatile` |
| LLM — Fallback Chain | Gemini 2.0 Flash → NVIDIA NIM nemotron-super-120b → OpenRouter |
| Web Search Fallback | Tavily API |
| Training | NVIDIA A100 GPU (IIIT Delhi) |
| Inference | Mac M2 Pro (CPU) |
| Frontend | React + Vite + TypeScript + Tailwind + Framer Motion |
| Backend API | FastAPI + Uvicorn |

---

## Repository Structure

```
cfyourmovie/
├── README.md
├── .env.example                          # API key template
├── Master_final.csv                      # Enriched 1682-movie dataset
│
├── docs/
│   ├── ARCHITECTURE.md                   # System design deep-dive
│   ├── PIPELINES.md                      # All 5 pipeline designs
│   ├── RESULTS.md                        # Full metrics + per-fold data
│   └── SETUP.md                          # Detailed setup guide
│
├── 01_data_prep.py                       # EDA + data cleaning
├── 02_train_lightfm.py                   # LightFM baseline
├── 03_lightfm_complete_pipeline.py
├── 04_hybrid_cf_content_pipeline.py
├── 05_mix_gpu.py                         # EASE + content model
├── 05_mix_gpu_final.py                   # Final tuned EASE model
├── admantium_feature_prep.py
│
├── 1.2Hybrid_RecSys_Optimised_Pipeline.ipynb
├── 1.3Deep_HyperParam_Tuning.ipynb
└── 2.1Final_Adaptive_RecSys.ipynb
│
└── rag_pipeline/
    ├── config.py                         # LLM factory + API config
    ├── build_index.py                    # FAISS index builder
    ├── build_bm25_index.py               # BM25 index builder
    ├── hybrid_retriever.py               # BM25 + FAISS RRF engine
    ├── query_parser.py                   # LLM → structured JSON
    ├── mix_gpu_train.py                  # CF training
    ├── mix_gpu_infer.py                  # CF inference
    ├── nvidia_reranker.py                # Cross-encoder reranker
    ├── web_search_fallback.py            # Tavily fallback
    ├── pipeline_1_seq.py  →  pipeline_5_hybrid_deep.py
    ├── recommend.py                      # Auto-select pipeline router
    └── compare_all.py                    # Cross-pipeline evaluation
│
└── ui/                                   # React + Vite frontend
    └── start.sh                          # Launches backend + frontend
```

---

## References

| Paper | Authors | Role |
|---|---|---|
| [Embarrassingly Shallow Autoencoders](https://arxiv.org/abs/1905.03375) | Steck, 2019 | EASE — CF backbone |
| [Precise Zero-Shot Dense Retrieval (HyDE)](https://arxiv.org/abs/2212.10496) | Gao et al., 2022 | HyDE query enrichment |
| [Reciprocal Rank Fusion](https://dl.acm.org/doi/10.1145/1571941.1572114) | Cormack et al., 2009 | BM25+FAISS score fusion |
| [MovieLens Datasets](https://dl.acm.org/doi/10.1145/2827872) | Harper & Konstan, 2015 | Base dataset |
| [Microsoft Recommenders](https://github.com/microsoft/recommenders) | Microsoft, 2019 | Primary benchmark |

---

## Built By

**Kush Tokas · Oyouknome_who**


---

<div align="center">

Built at IIIT Delhi · Collaborative Filtering Course · 2025–26

</div>
