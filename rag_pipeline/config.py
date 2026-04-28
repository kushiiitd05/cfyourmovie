from pathlib import Path
from langchain_core.embeddings import Embeddings as _LCEmbeddings
import os

# Fix for macOS OpenMP duplicate lib conflict (torch + faiss both link libomp)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# ── Project root — auto-detect or override via env ────────────
# Set CF_PROJECT_DIR env var to override (useful on any server/machine).
# Default: directory two levels above this file (rag_pipeline/../)
_ENV_PATH = os.environ.get("CF_PROJECT_DIR")
if _ENV_PATH:
    BASE_DIR = Path(_ENV_PATH)
else:
    BASE_DIR = Path(__file__).parent.parent.resolve()

# Enable CUDA GPU if available (set CF_CUDA_DEVICE env var to override)
_cuda_device = os.environ.get("CF_CUDA_DEVICE")
if _cuda_device:
    os.environ["CUDA_VISIBLE_DEVICES"] = _cuda_device
DATA_DIR     = BASE_DIR / 'ml-100k'
META_PATH    = BASE_DIR / 'Master_final.csv'
RAG_DIR      = BASE_DIR / 'rag_pipeline'
MATRICES_DIR = RAG_DIR  / 'matrices'
INDEX_DIR    = RAG_DIR  / 'faiss_index'

MATRICES_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────
RANDOM_SEED = 42

# NOTE on embedding models — intentional two-model design:
#   FAISS retrieval  → NVIDIA llama-nemotron-embed-1b-v2 (multilingual, 8192 tok)
#   Mix_GPU CB       → all-MiniLM-L6-v2 (384-dim, via sentence-transformers, unchanged)
# These serve SEPARATE purposes and are never compared in raw vector space.
# Fusion operates on Z-normalized scalar scores — no vector space conflict.

# ── Mix_GPU fixed config (from Optuna run in 05_mix_gpu.py) ──
MIX_CONFIG = {
    'lambda_ease':      2199.9395646690464,
    'half_life_days':   218,
    'negative_penalty': 0.9669969952037883,
    'strict_filter':    4,
}

MIX_PARAMS = {
    'alpha_struct': 0.10008381411545013,
    'alpha_embed':  0.03721010352958694,
    'beta':         0.013144045208014172,
    'w_genre':      1.7638679280017653,
    'w_num':        0.7181982665890585,
    'perc':         74,
}

# ── API Keys & Models ─────────────────────────────────────────

# NVIDIA NIM
NVIDIA_API_KEY  = os.environ.get("NVIDIA_API_KEY", "")
os.environ.setdefault("NVIDIA_API_KEY", NVIDIA_API_KEY)
NVIDIA_BASE_URL   = "https://integrate.api.nvidia.com/v1"
# Chat: nemotron-3-super-120b — 1M context, hybrid Mamba-Transformer, best for agentic/complex queries
NVIDIA_CHAT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
# Embedding: llama-nemotron-embed-1b-v2 — 26 languages, 8192 token context, free endpoint
NVIDIA_EMBED_MODEL  = "nvidia/llama-nemotron-embed-1b-v2"
# Reranker: pairs with embed model — model-specific ranking endpoint
NVIDIA_RERANK_MODEL = "nvidia/llama-nemotron-rerank-1b-v2"
# Reranker uses a dedicated URL (different from the OpenAI-compat base)
NVIDIA_RERANK_URL   = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/ranking"

# Ollama Cloud — kimi-k2.5:cloud (best quality, ~Claude Opus tier)
OLLAMA_API_KEY  = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_BASE_URL = "https://ollama.com/v1"
OLLAMA_MODEL    = "kimi-k2.5:cloud"

# Groq
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", GROQ_API_KEY)
GROQ_MODEL = "llama-3.3-70b-versatile"

# OpenRouter (last resort only)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
os.environ.setdefault("OPENROUTER_API_KEY", OPENROUTER_API_KEY)
OPENROUTER_BASE   = "https://openrouter.ai/api/v1"
OPENROUTER_MODELS = [
    "meta/llama-3.3-70b-instruct",
    "google/gemma-3-27b-it:free",
    "meta-llama/llama-3.2-3b-instruct:free",
]

# Natural language answer system prompt
# Use as the base system message for any LLM answering in natural GPT-like style.
ANSWER_SYSTEM_PROMPT = (
    "You are a friendly, knowledgeable movie recommendation assistant. "
    "Respond naturally and conversationally — like a film-savvy friend, not a search engine. "
    "Be concise (3-5 sentences), mention specific details that explain *why* each film matches, "
    "and never use bullet lists unless explicitly asked. "
    "Do not mention models, databases, APIs, or any internal system details."
)

# Web search fallback
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
os.environ.setdefault("TAVILY_API_KEY", TAVILY_API_KEY)

# Google Gemini (OpenAI-compat endpoint) — gemini-2.0-flash: fast, free, generous quota
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
os.environ.setdefault("GEMINI_API_KEY", GEMINI_API_KEY)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_MODEL    = "gemini-2.0-flash-lite"

# ── Fusion defaults (overridden by optuna_tune.py results) ────
DEFAULT_ALPHA = 0.80
DEFAULT_BETA  = 0.20

# ── FAISS document template ───────────────────────────────────
# budget/revenue excluded (~50% missing, lowest signal per analysis)
# Structured format improves semantic encoding vs raw concatenation
FAISS_DOC_TEMPLATE = (
    "{title} ({year}). "
    "Overview: {overview}. "
    "Keywords: {keywords}. "
    "Genres: {genres}. "
    "Director: {director}. "
    "Cast: {cast}. "
    "Rating: {vote_average}/10. "
    "Runtime: {runtime} min."
)


# ── LLM factory ──────────────────────────────────────────────
def get_llm(temperature: float = 0.3, json_mode: bool = False):
    """
    Returns a LangChain ChatOpenAI model with automatic fallbacks.
    Priority: Groq → Gemini 2.0 Flash → NVIDIA NIM → OpenRouter.
    Uses .with_fallbacks() so 429/5xx on primary auto-retries with next provider.
    """
    from langchain_openai import ChatOpenAI

    def _make(base_url, api_key, model):
        kwargs = dict(model=model, api_key=api_key, base_url=base_url, temperature=temperature)
        if json_mode:
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        return ChatOpenAI(**kwargs)

    llms = []

    # 0. Ollama Cloud — kimi-k2.5:cloud, highest quality (~Claude Opus), priority-0
    if OLLAMA_API_KEY:
        llms.append(_make(OLLAMA_BASE_URL, OLLAMA_API_KEY, OLLAMA_MODEL))

    # 1. Groq — fastest, best for structured JSON
    if GROQ_API_KEY:
        llms.append(_make("https://api.groq.com/openai/v1", GROQ_API_KEY, GROQ_MODEL))

    # 2. Gemini 2.0 Flash — fast, generous free quota, second fastest
    if GEMINI_API_KEY:
        llms.append(_make(GEMINI_BASE_URL, GEMINI_API_KEY, GEMINI_MODEL))

    # 3. NVIDIA NIM — 1M ctx fallback (slow but reliable)
    if NVIDIA_API_KEY:
        llms.append(_make(NVIDIA_BASE_URL, NVIDIA_API_KEY, NVIDIA_CHAT_MODEL))

    # 4. OpenRouter models — last resort chain
    if OPENROUTER_API_KEY:
        for m in OPENROUTER_MODELS:
            llms.append(_make(OPENROUTER_BASE, OPENROUTER_API_KEY, m))

    if not llms:
        raise RuntimeError("No LLM API key available. Set GROQ_API_KEY or GEMINI_API_KEY.")

    # Wire up automatic fallback chain: primary.with_fallbacks([f1, f2, ...])
    primary = llms[0]
    return primary.with_fallbacks(llms[1:]) if len(llms) > 1 else primary


# ── NVIDIA NIM Embeddings ─────────────────────────────────────
class NVIDIAEmbeddings(_LCEmbeddings):
    """
    Thin OpenAI-compatible embedder for NVIDIA NIM.
    Inherits LangChain Embeddings ABC — FAISS treats it as a proper embedder,
    no 'embedding_function' deprecation warning.

    input_type: "passage" for document indexing, "query" for search.
    NVIDIA nemotron-embed applies asymmetric representations per input_type.
    """
    def __init__(self, input_type: str = "passage"):
        from openai import OpenAI
        self._client = OpenAI(api_key=NVIDIA_API_KEY, base_url=NVIDIA_BASE_URL)
        self._input_type = input_type

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batch in chunks of 50 to stay within NIM request limits
        results = []
        for i in range(0, len(texts), 50):
            chunk = texts[i:i + 50]
            resp = self._client.embeddings.create(
                model=NVIDIA_EMBED_MODEL,
                input=chunk,
                extra_body={"input_type": self._input_type, "truncate": "END"},
            )
            resp_sorted = sorted(resp.data, key=lambda d: d.index)
            results.extend([d.embedding for d in resp_sorted])
        return results

    def embed_query(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(
            model=NVIDIA_EMBED_MODEL,
            input=[text],
            extra_body={"input_type": "query", "truncate": "END"},
        )
        return resp.data[0].embedding


def get_embedder(input_type: str = "passage") -> NVIDIAEmbeddings:
    """Returns NVIDIA NIM embedder. Use input_type='passage' for build_index, 'query' for search."""
    return NVIDIAEmbeddings(input_type=input_type)
