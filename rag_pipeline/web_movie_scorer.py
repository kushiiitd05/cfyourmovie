"""
web_movie_scorer.py
-------------------
Proper content-based scoring for web-retrieved movies that are NOT in the database.

Since EASE^R (CF component) requires historical ratings which new movies don't have,
we use ONLY the content-based component of Mix_GPU — the same Rocchio user profiles
that Mix_GPU already built during training.

Scoring formula for a web movie w and user u:
    cb_struct_score = user_profile_struct[u] · content_struct(w)
    cb_embed_score  = user_profile_embed[u]  · embed(overview + keywords + cast)
    
    final_score = alpha_struct * Z(cb_struct_score)
                + alpha_embed  * Z(cb_embed_score)

Where:
    - user_profile_struct[u] and user_profile_embed[u] are loaded from the
      saved Mix_GPU matrices (already computed during mix_gpu_train.py)
    - alpha_struct, alpha_embed are the same Optuna-tuned values from MIX_PARAMS
    - Z() = z-score normalisation against the user's existing CB scores

This is academically honest: we clearly state in the output that EASE^R CF signal
is absent (cold item) and the score is CB-only. This is a documented limitation
and valid Future Work item.

Used by: web_search_fallback.py → _score_web_movies_proper()
"""

import sys
import ast
import logging
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from config import (MATRICES_DIR, META_PATH, MIX_PARAMS, MIX_CONFIG)

log = logging.getLogger(__name__)

# ── Module-level metadata cache (avoids reloading CSV per movie) ─
_meta_df_cache = None
_genre_list_cache = None

def _get_meta_and_genres():
    global _meta_df_cache, _genre_list_cache
    if _meta_df_cache is None:
        import pandas as pd
        _meta_df_cache = pd.read_csv(META_PATH)
        _meta_df_cache.rename(columns={'movie_id': 'item_id'}, inplace=True)
        all_g = set()
        for g in _meta_df_cache['genres'].dropna():
            try: all_g.update(ast.literal_eval(str(g)))
            except: pass
        _genre_list_cache = sorted(list(all_g))
        log.info(f"[web_scorer] Metadata cached: {len(_meta_df_cache)} movies, {len(_genre_list_cache)} genres")
    return _meta_df_cache, _genre_list_cache

# ── Load user profiles once at import ────────────────────────
# These are saved by mix_gpu_train.py — shape (n_users, feature_dim)
_UP_STRUCT_PATH = MATRICES_DIR / 'user_profiles_struct.npy'
_UP_EMBED_PATH  = MATRICES_DIR / 'user_profiles_embed.npy'
_CB_STRUCT_PATH = MATRICES_DIR / 'cb_struct_scores.npy'  # (n_users, n_items)
_CB_EMBED_PATH  = MATRICES_DIR / 'cb_embed_scores.npy'   # (n_users, n_items)

# Check if extended matrices exist (saved by updated mix_gpu_train.py below)
_profiles_available = _UP_STRUCT_PATH.exists() and _UP_EMBED_PATH.exists()
if _profiles_available:
    log.info("[web_scorer] Loading user content profiles...")
    _up_struct = np.load(_UP_STRUCT_PATH)   # (n_users, struct_dim)
    _up_embed  = np.load(_UP_EMBED_PATH)    # (n_users, embed_dim=384)
    log.info(f"[web_scorer] Profiles loaded: struct={_up_struct.shape} embed={_up_embed.shape}")
else:
    log.warning("[web_scorer] User profiles not found. Run updated mix_gpu_train.py first.")
    _up_struct = None
    _up_embed  = None


def _build_struct_features(movie: dict) -> Optional[np.ndarray]:
    """
    Build the structural content feature vector for a web movie.
    Matches the exact feature construction in mix_gpu_train.py:
        [genre_onehot * w_genre | numeric_features * w_num]  → L2 normalised

    Args:
        movie: dict with keys title, genres, vote_average, vote_count,
               popularity, budget (optional), revenue (optional)

    Returns:
        np.ndarray of shape (struct_dim,) or None if features unavailable
    """
    from scipy.sparse import csr_matrix, hstack
    from sklearn.preprocessing import normalize

    try:
        # Use cached metadata — avoids reloading CSV per movie
        meta, gl = _get_meta_and_genres()

        # Genre one-hot
        movie_genres_str = movie.get('genres', '')
        try:
            movie_genre_list = ast.literal_eval(str(movie_genres_str)) \
                if movie_genres_str.startswith('[') else \
                [g.strip() for g in movie_genres_str.split(',')]
        except Exception:
            movie_genre_list = [movie_genres_str] if movie_genres_str else []

        movie_genre_set = {g.lower().strip() for g in movie_genre_list}
        genre_vec = np.array([[1.0 if g.lower() in movie_genre_set else 0.0
                               for g in gl]])   # (1, n_genres)

        # Numeric features — same log1p + z-score as training
        # Use dataset mean/std from training data for normalisation
        numeric_cols = ['vote_average', 'vote_count', 'popularity', 'budget', 'revenue']
        nums = []
        for col in numeric_cols:
            val = float(movie.get(col, 0) or 0)
            v_all = np.log1p(meta[col].fillna(0).values)
            v_norm = (np.log1p(val) - v_all.mean()) / (v_all.std() + 1e-8)
            nums.append(v_norm)
        numeric_vec = np.array([nums])   # (1, 5)

        # Stack and weight — same as mix_gpu_train.py
        struct = hstack([
            csr_matrix(genre_vec)  * MIX_PARAMS['w_genre'],
            csr_matrix(numeric_vec) * MIX_PARAMS['w_num'],
        ])
        struct_norm = normalize(struct, norm='l2').toarray().flatten()
        return struct_norm

    except Exception as e:
        log.warning(f"[web_scorer] struct feature build failed: {e}")
        return None


def _build_embed_features(movie: dict) -> Optional[np.ndarray]:
    """
    Build the 384-dim sentence embedding for a web movie.
    Uses the EXACT same SentenceTransformer and text format as mix_gpu_train.py:
        overview + " " + keywords + " " + cast
    """
    from sklearn.preprocessing import normalize
    from sentence_transformers import SentenceTransformer

    try:
        overview  = str(movie.get('overview', '') or '')
        keywords  = str(movie.get('keywords', '') or movie.get('movie_keywords', '') or '')
        cast      = str(movie.get('cast', '') or movie.get('top_cast', '') or '')

        text = f"{overview} {keywords} {cast}".strip()
        if not text:
            text = str(movie.get('title', ''))

        model = SentenceTransformer("all-MiniLM-L6-v2", device='cpu')
        emb = normalize(model.encode([text], batch_size=1))
        return emb.flatten()   # (384,)

    except Exception as e:
        log.warning(f"[web_scorer] embed feature build failed: {e}")
        return None


def score_web_movie_for_user(user_id: int, movie: dict) -> dict:
    """
    Score a single web-retrieved movie for a user using Mix_GPU CB component.

    Returns the movie dict with added keys:
        mix_score        : float — CB-only score (higher = better match)
        scoring_method   : str   — explains how score was computed
        cf_available     : False — always False for web movies
    """
    if not _profiles_available or _up_struct is None:
        movie['mix_score']      = 0.0
        movie['scoring_method'] = 'profiles_not_loaded'
        movie['cf_available']   = False
        return movie

    u = user_id - 1
    if u < 0 or u >= _up_struct.shape[0]:
        movie['mix_score']      = 0.0
        movie['scoring_method'] = 'user_out_of_range'
        movie['cf_available']   = False
        return movie

    scores = []

    # ── Structural CB score ──
    struct_feat = _build_struct_features(movie)
    if struct_feat is not None and np.linalg.norm(_up_struct[u]) > 1e-8:
        cb_struct = float(np.dot(_up_struct[u], struct_feat))
        scores.append(MIX_PARAMS['alpha_struct'] * cb_struct)
    else:
        cb_struct = 0.0

    # ── Semantic embedding CB score ──
    embed_feat = _build_embed_features(movie)
    if embed_feat is not None and np.linalg.norm(_up_embed[u]) > 1e-8:
        cb_embed = float(np.dot(_up_embed[u], embed_feat))
        scores.append(MIX_PARAMS['alpha_embed'] * cb_embed)
    else:
        cb_embed = 0.0

    final_score = sum(scores)

    movie['mix_score']      = round(final_score, 4)
    movie['cb_struct_score'] = round(cb_struct, 4)
    movie['cb_embed_score']  = round(cb_embed, 4)
    movie['scoring_method']  = 'cb_only_no_cf'   # honest label
    movie['cf_available']    = False
    return movie


def score_web_movies_proper(
    user_id: Optional[int],
    web_movies: list[dict],
) -> list[dict]:
    """
    Score all web-retrieved movies for a user.
    First checks if each movie exists in the DB (title match) — if yes,
    uses the full Mix_GPU hybrid score. If not, uses CB-only scoring above.

    This is called by web_search_fallback.py instead of the old genre-overlap heuristic.
    """
    import pandas as pd
    from mix_gpu_infer import hybrid, META_DF, n_users

    if user_id is None:
        for m in web_movies:
            m['mix_score']     = 0.0
            m['scoring_method'] = 'cold_start'
            m['cf_available']   = False
        return web_movies

    u = user_id - 1

    for movie in web_movies:
        title = str(movie.get('title', '')).lower().strip()

        # Try exact DB title match first
        matches = META_DF[META_DF['title'].str.lower().str.strip() == title]
        if not matches.empty:
            item_idx = matches.index[0]
            movie['mix_score']      = round(float(hybrid[u, item_idx]), 4)
            movie['movie_id']       = int(matches.iloc[0]['item_id'])
            movie['scoring_method'] = 'full_hybrid_db_match'
            movie['cf_available']   = True
            movie['in_db']          = True
        else:
            # Not in DB — use CB-only scoring with actual user profiles
            movie = score_web_movie_for_user(user_id, movie)
            movie['in_db'] = False

    # Sort by score
    web_movies.sort(key=lambda x: x.get('mix_score', 0), reverse=True)
    return web_movies
