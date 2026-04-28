import logging
import sys
import os
from pathlib import Path
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
log = logging.getLogger("calibrate_fallback")

sys.path.insert(0, str(Path(__file__).parent))
from nvidia_reranker import rerank_movies

# Case: HARD MISMATCH
query = "car racing movies"
movies = [
    {"movie_id": 1, "title": "Vertigo", "overview": "A retired San Francisco detective suffering from acrophobia investigates the strange activities of an old friend's wife."},
    {"movie_id": 2, "title": "Chasing Amy", "overview": "Holden and Banky are comic book artists. Everything's going well for them until they meet Alyssa."},
]

log.info(f"Testing BGE scores for HARD MISMATCH: '{query}'")
results = rerank_movies(query, movies, top_k=5)

for m in results:
    print(f"[{m['nvidia_rank_score']:.4f}] {m['title']}")

# Case: POTENTIAL MATCH (even if weak in MovieLens)
query2 = "detective mystery"
log.info(f"\nTesting BGE scores for POTENTIAL MATCH: '{query2}'")
results2 = rerank_movies(query2, movies, top_k=5)
for m in results2:
    print(f"[{m['nvidia_rank_score']:.4f}] {m['title']}")
