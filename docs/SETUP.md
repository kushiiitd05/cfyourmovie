# Setup Guide

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.8+ | 3.14+ recommended |
| pip | Any | Used inside virtualenv |
| git | Any | For cloning |
| GROQ_API_KEY | Required | Free tier works; ~100 req/day limit |
| NVIDIA_API_KEY | Required | Free NVIDIA AI Foundation tier |
| TAVILY_API_KEY | Optional | Web search fallback; free tier available |
| GEMINI_API_KEY | Optional | LLM fallback chain |

---

## Installation

### 1. Clone

```bash
git clone https://github.com/YOUR_USERNAME/cfyourmovie.git
cd cfyourmovie
```

### 2. Virtual Environment

```bash
python -m venv rag_venv
source rag_venv/bin/activate          # macOS/Linux
# rag_venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r rag_pipeline/requirements.txt
```

Core packages installed:
- `langchain>=0.2.16` + `langchain-openai>=0.1.8`
- `faiss-cpu>=1.7.4`
- `sentence-transformers>=2.7.0`
- `numpy`, `pandas`, `scipy`, `scikit-learn`
- `optuna>=3.4.0`
- `openai>=1.40.0` (NVIDIA NIM is OpenAI-compatible)

### 4. Configure API Keys

```bash
cp .env.example .env
```

Edit `.env`:
```bash
GROQ_API_KEY=gsk_...              # Required — query parser + LLM synthesis
NVIDIA_API_KEY=nvapi-...          # Required — embeddings + reranker
TAVILY_API_KEY=tvly-...           # Optional — web fallback
GEMINI_API_KEY=AIza...            # Optional — LLM fallback
OPENROUTER_API_KEY=sk-or-...      # Optional — last-resort LLM
```

Load into shell:
```bash
source .env
# Or use python-dotenv (auto-loaded by config.py)
```

---

## Building Indexes (One-Time)

These artifacts are gitignored (large binaries). Build them once locally.

### Dense FAISS Index

Requires NVIDIA_API_KEY. Embeds all 1682 movies via NVIDIA NIM API.
Approximate time: 5–10 minutes (API rate limits).

```bash
python -m rag_pipeline.build_index
# Output: rag_pipeline/faiss_index/index.faiss + metadata.pkl
```

### BM25 Index

No API key required. Pure Python, fast (< 30 seconds).

```bash
python -m rag_pipeline.build_bm25_index
# Output: rag_pipeline/bm25_index/bm25.pkl
```

### Train CF Model

Loads `Master_final.csv` + `ml-100k/` ratings, trains EASE + content features.
- On A100: ~2 minutes
- On Mac M2 CPU: ~10–15 minutes

```bash
python rag_pipeline/mix_gpu_train.py
# Output: rag_pipeline/matrices/hybrid.npy, Xo.npy
```

---

## Running Pipelines

### Single Query

```python
from rag_pipeline.pipeline_5_hybrid_deep import run

result = run(
    user_id=1,
    query="dark psychological thriller with moral ambiguity from the 90s",
    n=10
)

print(result['explanation'])
for movie in result['movies']:
    print(f"[{movie['fused_score']:+.3f}] {movie['title']} ({movie['year']})")
```

### Using the Auto-Router

```python
from rag_pipeline.recommend import recommend

# Automatically selects best pipeline for the query type
result = recommend(user_id=42, query="Spielberg adventure films", n=10)
```

### Pipeline-Specific Imports

```python
from rag_pipeline.pipeline_1_seq import run    # Pure CF
from rag_pipeline.pipeline_2_dual import run   # CF + FAISS
from rag_pipeline.pipeline_3_rag import run    # RAG-only
from rag_pipeline.pipeline_4_hyde import run   # HyDE
from rag_pipeline.pipeline_5_hybrid_deep import run  # Full system
```

### Compare All Pipelines

```bash
python -m rag_pipeline.compare_all
```

Runs all 5 pipelines on a standard test set and prints a comparison table.

---

## Frontend (React UI)

```bash
cd ui
npm install
npm run dev
# Opens at http://localhost:5173 (or 5174 if port is taken)
```

Ensure the FastAPI backend is running:

```bash
cd rag_pipeline
uvicorn app:app --reload --port 8000
```

---

## GPU Training (A100 — Optional)

For Optuna hyperparameter search or retraining EASE on new data:

```bash
# On NVIDIA A100 server
python rag_pipeline/optuna_tune.py --n-trials 200 --study-name ease_optuna
```

Results saved to `optuna_results/best_params.json`. Copy best params to `mix_gpu_train.py`.

---

## Troubleshooting

**FAISS index not found:**
```
FileNotFoundError: rag_pipeline/faiss_index/index.faiss
```
→ Run `python -m rag_pipeline.build_index` first.

**NVIDIA API rate limit:**
```
RateLimitError: 429
```
→ NVIDIA free tier has rate limits. The pipeline auto-falls-back to Groq for LLM calls. For embeddings, retry after 60s or use a paid key.

**CF matrices not found:**
```
FileNotFoundError: rag_pipeline/matrices/hybrid.npy
```
→ Run `python rag_pipeline/mix_gpu_train.py` first.

**Groq JSON parsing error:**
Occasionally Groq returns malformed JSON. `query_parser.py` has a retry loop with exponential backoff. If it persists, check your GROQ_API_KEY quota.

**Low confidence / web fallback triggering on every query:**
- Check that FAISS index was built with the correct NVIDIA_API_KEY
- Verify `rag_pipeline/matrices/Xo.npy` exists (user-item matrix from training)
- Adjust thresholds in `rag_pipeline/config.py`: `FAISS_CONFIDENCE_THRESHOLD`, `MIX_CONFIDENCE_THRESHOLD`

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | Yes | — | Primary LLM (query parser + synthesis) |
| `NVIDIA_API_KEY` | Yes | — | Embeddings (`llama-nemotron-embed-1b-v2`) + reranker |
| `TAVILY_API_KEY` | No | — | Web search fallback |
| `GEMINI_API_KEY` | No | — | LLM fallback #2 |
| `OPENROUTER_API_KEY` | No | — | LLM fallback #3 |
| `OLLAMA_API_KEY` | No | — | Local Ollama (if running locally) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434/v1` | Ollama endpoint |
