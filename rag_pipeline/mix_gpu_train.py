#!/usr/bin/env python3
"""
mix_gpu_train.py
----------------
Trains Mix_GPU on ALL 100k ratings (no CV split) and saves matrices.
Run once. Subsequent runs load cached embeddings — saves ~3 min per restart.

Run:  python mix_gpu_train.py
"""

import os
import torch
_DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent))
from config import (DATA_DIR, META_PATH, MATRICES_DIR,
                    MIX_CONFIG, MIX_PARAMS, RANDOM_SEED)

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)s  %(message)s',
    handlers=[
        logging.FileHandler(MATRICES_DIR.parent / 'train.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)

# ── Embedding cache path ──────────────────────────────────────
EMB_CACHE = MATRICES_DIR / 'embeddings_minilm.npy'


def train_and_save():
    log.info("=== mix_gpu_train.py starting ===")

    # ── Load data ─────────────────────────────────────────────
    log.info("Loading ratings and metadata...")
    raw = pd.read_csv(
        DATA_DIR / 'u.data', sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp']
    )
    meta = pd.read_csv(META_PATH)
    meta.rename(columns={'movie_id': 'item_id'}, inplace=True)
    meta = meta.reset_index(drop=True)

    n_users = int(raw['user_id'].max())
    n_items = len(meta)
    log.info(f"Users={n_users}  Items={n_items}  Ratings={len(raw)}")

    # ── Time-decayed ratings ──────────────────────────────────
    max_ts = raw['timestamp'].max()
    days   = (max_ts - raw['timestamp']) / 86400
    decay  = np.power(0.5, days / MIX_CONFIG['half_life_days'])
    raw['wr'] = raw['rating'] * decay

    u_idx = raw['user_id'].values - 1
    i_idx = raw['item_id'].values - 1
    r_w   = raw['wr'].values
    r_o   = raw['rating'].values

    # ── Sentence embeddings — CACHED ─────────────────────────
    # First run: ~3 min. All subsequent runs: instant (loads from disk).
    if EMB_CACHE.exists():
        log.info(f"Loading cached embeddings from {EMB_CACHE}")
        emb = np.load(EMB_CACHE)
        log.info(f"Embeddings loaded: shape={emb.shape}")
    else:
        log.info("Computing sentence embeddings (all-MiniLM-L6-v2) — first run only...")
        sent_model = SentenceTransformer("all-MiniLM-L6-v2", device=_DEVICE)
        text = (meta['overview'].fillna('') + " " +
                meta['movie_keywords'].fillna('').astype(str) + " " +
                meta['top_cast'].fillna('').astype(str)).tolist()
        emb = normalize(sent_model.encode(text, batch_size=64, show_progress_bar=True))
        np.save(EMB_CACHE, emb)
        log.info(f"Embeddings saved to {EMB_CACHE}  shape={emb.shape}")

    # ── Genre one-hot ─────────────────────────────────────────
    log.info("Building genre and numeric feature matrices...")
    all_g = set()
    for g in meta['genres'].dropna():
        try: all_g.update(eval(g))
        except: pass
    gl = sorted(list(all_g))

    genre_rows = []
    for g in meta['genres']:
        try:   gset = set(eval(g)) if pd.notna(g) else set()
        except: gset = set()
        genre_rows.append([1.0 if x in gset else 0.0 for x in gl])
    genres_mat = csr_matrix(np.array(genre_rows))

    # ── Numeric features
    # budget/revenue: imputed to 0 (50% missing) — consistent with original
    # popularity Z-normalized before subtraction to prevent scale mismatch
    nums = []
    for col in ['vote_average', 'vote_count', 'popularity', 'budget', 'revenue']:
        v = np.log1p(meta[col].fillna(0).values)
        v = (v - v.mean()) / (v.std() + 1e-8)
        nums.append(v)
    nums_mat = csr_matrix(np.array(nums).T)

    content_struct = normalize(hstack([
        genres_mat * MIX_PARAMS['w_genre'],
        nums_mat   * MIX_PARAMS['w_num']
    ]))
    content_embed = csr_matrix(emb)

    # ── Full rating matrices ───────────────────────────────────
    log.info("Building EASE^R on full 100k data...")
    X  = coo_matrix((r_w, (u_idx, i_idx)), shape=(n_users, n_items)).toarray()
    Xo = coo_matrix((r_o, (u_idx, i_idx)), shape=(n_users, n_items)).toarray()

    G = X.T @ X
    G[np.diag_indices_from(G)] += MIX_CONFIG['lambda_ease']
    P = np.linalg.inv(G)
    B = P / (-np.diag(P)[:, None])
    np.fill_diagonal(B, 0)
    cf = X @ B
    log.info("EASE^R done")

    # ── Rocchio user profiles ─────────────────────────────────
    log.info("Building user content profiles (Rocchio)...")
    up_struct = np.zeros((n_users, content_struct.shape[1]))
    up_embed  = np.zeros((n_users, content_embed.shape[1]))

    for u in range(n_users):
        liked = np.where(Xo[u] >= MIX_CONFIG['strict_filter'])[0]
        hated = np.where((Xo[u] > 0) & (Xo[u] < 3))[0]
        if len(liked) > 0:
            ps = content_struct[liked].mean(axis=0).A1
            pe = content_embed[liked].mean(axis=0).A1
            if len(hated) > 0:
                ps = np.maximum(ps - MIX_CONFIG['negative_penalty'] * content_struct[hated].mean(axis=0).A1, 0)
                pe = np.maximum(pe - MIX_CONFIG['negative_penalty'] * content_embed[hated].mean(axis=0).A1, 0)
            up_struct[u] = ps / (np.linalg.norm(ps) + 1e-8)
            up_embed[u]  = pe / (np.linalg.norm(pe) + 1e-8)

    cb_struct = up_struct @ content_struct.T.toarray()
    cb_embed  = up_embed  @ content_embed.T.toarray()

    # ── Z-score normalize per user (same as original mix_gpu.py) ─
    cf        = (cf        - cf.mean(1, keepdims=True))        / (cf.std(1, keepdims=True)        + 1e-8)
    cb_struct = (cb_struct  - cb_struct.mean(1, keepdims=True)) / (cb_struct.std(1, keepdims=True)  + 1e-8)
    cb_embed  = (cb_embed   - cb_embed.mean(1, keepdims=True))  / (cb_embed.std(1, keepdims=True)   + 1e-8)

    # ── Popularity penalty ─────────────────────────────────────
    # Popularity = log(item_counts), subtracted with beta weight.
    # Scale is consistent because cf is already Z-normalized and beta (0.013) is small.
    item_counts = (Xo > 0).sum(axis=0)
    popularity  = np.log1p(item_counts)

    # ── Hybrid score matrix ───────────────────────────────────
    # perc=74: CB embeddings only activate for items below 74th percentile of CF score.
    # This prevents CB from overriding strong CF signals — justified by Optuna tuning.
    threshold = np.percentile(cf, MIX_PARAMS['perc'])
    mask      = (cf < threshold)
    hybrid    = (cf
                 + MIX_PARAMS['alpha_struct'] * cb_struct
                 + MIX_PARAMS['alpha_embed']  * (cb_embed * mask)
                 - MIX_PARAMS['beta']         * popularity)

    # ── Save ──────────────────────────────────────────────────
    log.info(f"Saving matrices to {MATRICES_DIR}...")
    np.save(MATRICES_DIR / 'hybrid.npy', hybrid.astype(np.float32))
    np.save(MATRICES_DIR / 'Xo.npy',     Xo.astype(np.float32))
    np.save(MATRICES_DIR / 'B.npy',      B.astype(np.float32))
    np.save(MATRICES_DIR / 'user_profiles_struct.npy', up_struct.astype(np.float32))
    np.save(MATRICES_DIR / 'user_profiles_embed.npy',  up_embed.astype(np.float32))
    log.info(f'  user_profiles_struct.npy: {up_struct.shape}')
    log.info(f'  user_profiles_embed.npy : {up_embed.shape}')
    log.info(f"  hybrid.npy : {hybrid.shape}  {hybrid.nbytes/1e6:.1f} MB")
    log.info(f"  Xo.npy     : {Xo.shape}")
    log.info(f"  B.npy      : {B.shape}")

    # ── Sanity check ─────────────────────────────────────────
    log.info("Sanity check — top 5 for user 1:")
    s = hybrid[0].copy()
    s[Xo[0] > 0] = -np.inf
    top5 = np.argsort(-s)[:5]
    for idx in top5:
        log.info(f"  [{s[idx]:.4f}]  {meta.iloc[idx]['title']} ({meta.iloc[idx].get('year','?')})")

    log.info("=== Done. Run build_index.py next. ===")


if __name__ == '__main__':
    train_and_save()
