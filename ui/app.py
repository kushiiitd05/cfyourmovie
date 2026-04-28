"""
app.py — CineMatch UI backend
FastAPI server wrapping the recommend router.
Run: uvicorn app:app --reload --port 8000
"""

import sys
import os
import logging
from pathlib import Path

# Load .env before anything imports config.py
_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

# Add rag_pipeline to path
sys.path.insert(0, str(Path(__file__).parent.parent / "rag_pipeline"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="CineMatch", version="1.0.0")


PIPELINE_META = {
    "P5": {
        "name": "Hybrid Deep",
        "badge": "BEST",
        "description": "BM25 + FAISS + HyDE + sub-query decomposition + post-fusion reranking",
        "color": "violet",
        "requires_user": True,
    },
    "P4": {
        "name": "HyDE Hybrid",
        "badge": "COMPLEX",
        "description": "Hypothetical Document Embeddings — best for stylistic / comparative queries",
        "color": "blue",
        "requires_user": True,
    },
    "P2": {
        "name": "Dual Engine",
        "badge": "BALANCED",
        "description": "Mix_GPU CF + FAISS fusion — personalized, fast, solid baseline",
        "color": "emerald",
        "requires_user": True,
    },
    "P1": {
        "name": "Sequential CF",
        "badge": "FASTEST",
        "description": "Pure Mix_GPU collaborative filtering — lowest latency",
        "color": "amber",
        "requires_user": True,
    },
    "P3": {
        "name": "RAG Cold-Start",
        "badge": "NO LOGIN",
        "description": "Content-only FAISS retrieval — works without a user profile",
        "color": "rose",
        "requires_user": False,
    },
    "auto": {
        "name": "Auto Select",
        "badge": "AUTO",
        "description": "Picks the best pipeline based on query complexity and user context",
        "color": "slate",
        "requires_user": False,
    },
}


class RecommendRequest(BaseModel):
    query: str
    user_id: int | None = None
    pipeline: str = "P5"
    n: int = 8


class RecommendResponse(BaseModel):
    pipeline: str
    query: str
    movies: list[dict]
    explanation: str
    meta: dict = {}


@app.get("/api/pipelines")
def get_pipelines():
    return PIPELINE_META


@app.post("/api/recommend")
async def recommend_endpoint(req: RecommendRequest):
    try:
        from recommend import recommend
        result = recommend(
            query=req.query,
            user_id=req.user_id,
            pipeline=req.pipeline,
            n=req.n,
        )
        # Normalize movie score field for frontend
        for m in result.get("movies", []):
            m["_score"] = (
                m.get("fused_score")
                or m.get("mix_score")
                or m.get("faiss_score_norm")
                or m.get("score")
                or 0.0
            )
        # Hard-enforce n — pipelines may return more internally
        result["movies"] = result.get("movies", [])[:req.n]
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        log.exception("recommend failed")
        raise HTTPException(status_code=500, detail=str(e))


# In dev mode: Vite runs on :5173 and proxies /api here.
# In prod: run `npm run build` → serve ui_dist/ via StaticFiles below.
# static_dir = Path(__file__).parent.parent / "ui_dist"
# if static_dir.exists():
#     app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
