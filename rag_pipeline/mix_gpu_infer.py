"""
mix_gpu_infer.py
----------------
Loads pre-computed matrices from mix_gpu_train.py.
Fast per-user recommendation and scoring. Matrices loaded once at import.

Public API:
    recommend(user_id, n, genre_filter, director_filter, year_min, year_max)
    score_for_items(user_id, item_ids) → dict{movie_id: float}
"""
import os
import torch as _torch
if _torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import ast
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import MATRICES_DIR, META_PATH

log = logging.getLogger(__name__)

# ── Load once at import ───────────────────────────────────────
log.info("[mix_gpu_infer] Loading matrices...")
hybrid = np.load(MATRICES_DIR / 'hybrid.npy')
Xo     = np.load(MATRICES_DIR / 'Xo.npy')

meta = pd.read_csv(META_PATH)
meta.rename(columns={'movie_id': 'item_id'}, inplace=True)
meta = meta.reset_index(drop=True)

n_users, n_items = hybrid.shape
META_DF = meta  # alias for web_movie_scorer.py

# ── Robust Index Mapping ──
# Maps raw movie_id (item_id in df) to matrix row/column index
_movie_id_to_idx = {int(row['item_id']): i for i, row in meta.iterrows()}
log.info(f"[mix_gpu_infer] Ready — {n_users} users × {n_items} items (map size: {len(_movie_id_to_idx)})")

# Popularity-based fallback scores for cold-start users
_pop_score = (meta['vote_average'].fillna(0) *
              np.log1p(meta['vote_count'].fillna(0))).values


# ── Helpers ───────────────────────────────────────────────────
def _parse_list(val):
    if pd.isna(val):
        return []
    try:
        r = ast.literal_eval(str(val))
        return r if isinstance(r, list) else [str(r)]
    except Exception:
        return [str(val)]


def _row_to_dict(idx: int, score: float) -> dict:
    row = meta.iloc[idx]
    return {
        'movie_id':     int(row['item_id']),
        'title':        str(row['title']),
        'year':         int(row['year']) if pd.notna(row.get('year')) else None,
        'genres':       str(row.get('genres', '')),
        'director':     str(row.get('director', '')),
        'vote_average': float(row.get('vote_average', 0.0) or 0.0),
        'overview':     str(row.get('overview', ''))[:300],
        'mix_score':    round(float(score), 6),
    }


# ── Public functions ──────────────────────────────────────────
def recommend(
    user_id: int,
    n: int = 10,
    genre_filter: list = None,
    director_filter: str = None,
    year_min: int = None,
    year_max: int = None,
) -> list:
    """
    Top-N recommendations for a user with optional metadata filters.
    Filters are applied AFTER scoring (mask already-seen + constraint items to -inf).

    Cold-start (user_id out of range): popularity-based fallback.
    If filters are too restrictive and return < n results, returns whatever is available
    without relaxing filters — caller handles the gap in the explainer prompt.
    """
    if user_id < 1 or user_id > n_users:
        log.warning(f"[infer] Cold-start: user_id={user_id} not in training data")
        top_idx = np.argsort(-_pop_score)[:n]
        return [_row_to_dict(i, float(_pop_score[i])) for i in top_idx]

    u      = user_id - 1
    scores = hybrid[u].copy()
    scores[Xo[u] > 0] = -np.inf   # mask seen items

    # ── Apply filters ──
    if genre_filter or director_filter or year_min or year_max:
        gf = {g.strip().lower() for g in genre_filter} if genre_filter else None
        df = director_filter.strip().lower() if director_filter else None

        for i in range(n_items):
            if scores[i] == -np.inf:
                continue
            row = meta.iloc[i]

            if gf:
                movie_genres = {g.lower() for g in _parse_list(row.get('genres', ''))}
                if not gf.intersection(movie_genres):
                    scores[i] = -np.inf
                    continue

            if df:
                if str(row.get('director', '')).strip().lower() != df:
                    scores[i] = -np.inf
                    continue

            if year_min or year_max:
                y = row.get('year')
                if pd.notna(y):
                    y = int(y)
                    if (year_min and y < year_min) or (year_max and y > year_max):
                        scores[i] = -np.inf
                else:
                    scores[i] = -np.inf

    top_idx = np.argsort(-scores)[:n]
    results = [_row_to_dict(int(i), scores[i])
               for i in top_idx if scores[i] > -np.inf]

    if len(results) < n:
        log.info(f"[infer] user={user_id} filters yielded only {len(results)}/{n} results")

    return results


def score_for_items(user_id: int, item_ids: list) -> dict:
    """
    Mix_GPU hybrid scores for specific movie_ids (used by P2/P4 fusion).
    Returns raw (unnormalized) scores — caller does Z-normalization.
    """
    u = user_id - 1 if 1 <= user_id <= n_users else 0
    scores = {}
    for mid in item_ids:
        idx = _movie_id_to_idx.get(int(mid))
        if idx is not None and idx < n_items:
            scores[mid] = float(hybrid[u, idx])
    return scores


# ── Quick test ────────────────────────────────────────────────
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print("\n--- recommend(user_id=1, n=5) ---")
    for m in recommend(1, n=5):
        print(f"  [{m['mix_score']:.4f}]  {m['title']} ({m['year']})")

    print("\n--- recommend(user_id=1, genre_filter=['Action'], n=5) ---")
    for m in recommend(1, n=5, genre_filter=['Action']):
        print(f"  [{m['mix_score']:.4f}]  {m['title']} ({m['year']})")

    print("\n--- score_for_items(user_id=1, [1,2,3]) ---")
    print(score_for_items(1, [1, 2, 3]))
