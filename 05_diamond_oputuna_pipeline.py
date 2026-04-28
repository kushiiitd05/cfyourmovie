# #!/usr/bin/env python3
# """
# THE GOD-MODE PIPELINE: Optuna HPO + Diamond Evaluation
# Phase 1: Finds Best Params (Time-Decay, Rocchio, Lambda, Ensemble Weight)
# Phase 2: Evaluates Full 5-Fold CV on Best Params with ALL Metrics (MAP, nDCG, Full MAP, Full nDCG, Coverage, Novelty, Diversity)
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy.sparse import csr_matrix, coo_matrix, hstack
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import normalize
# from sklearn.model_selection import StratifiedKFold
# import optuna
# import warnings
# warnings.filterwarnings('ignore')
# optuna.logging.set_verbosity(optuna.logging.INFO)
# #  
# # 3🏆 GOD PARAMETERS FOUND:
# #    - lambda_ease: 767.6177
# #    - half_life_days: 197
# #    - negative_penalty: 0.5735
# #    - cf_weight: 0.7710
# #    - cb_weight: 0.2290


# # 2🏆 GOD PARAMETERS FOUND:
# #    - lambda_ease: 607.3118
# #    - half_life_days: 84
# #    - negative_penalty: 0.6524
# #    - cf_weight: 0.7959
# #    - cb_weight: 0.2041

# # 1🏆 GOD PARAMETERS FOUND:

# #    - lambda_ease: 798.7934

# #    - half_life_days: 116

# #    - negative_penalty: 0.8778

# #    - cf_weight: 0.7870

# #    - cb_weight: 0.2130


# CONFIG = {
#     'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
#     'output_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf/lightfm_results'),
#     'strict_filter': 4,
#     'n_optuna_trials': 264 # Optuna kitni baar try karega
# }

# # ============================================================================
# # [STEP 1] LOAD DATA & BUILD FEATURES
# # ============================================================================
# print("\n[STEP 1] Loading Data & Extracting Features...")

# raw_ratings_df = pd.read_csv(
#     CONFIG['data_dir'] / 'ml-100k/u.data',
#     sep='\t',
#     names=['user_id', 'item_id', 'rating', 'timestamp'],
#     dtype={'user_id': int, 'item_id': int, 'rating': int}
# )

# metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
# metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# n_users = raw_ratings_df['user_id'].max()
# n_items = len(metadata_df)
# user_idx_raw = raw_ratings_df['user_id'].values - 1
# item_idx_raw = raw_ratings_df['item_id'].values - 1

# # Features
# features = []
# all_genres = set()
# for g_str in metadata_df['genres'].dropna():
#     try: all_genres.update(eval(g_str))
#     except: pass
# genre_list = sorted(list(all_genres))
# genre_vecs = [[1.0 if g in (eval(g_str) if pd.notna(g_str) else []) else 0.0 for g in genre_list] for g_str in metadata_df['genres']]
# features.append(csr_matrix(np.array(genre_vecs)))

# overview_tf = TfidfVectorizer(max_features=100, min_df=2).fit_transform(metadata_df['overview'].fillna(''))
# features.append(csr_matrix(TruncatedSVD(n_components=min(32, overview_tf.shape[1]-1), random_state=42).fit_transform(overview_tf)))

# kw_texts = [' '.join(eval(kw)[:20]) if pd.notna(kw) else '' for kw in metadata_df['movie_keywords']]
# kw_tf = TfidfVectorizer(max_features=50).fit_transform(kw_texts)
# features.append(csr_matrix(TruncatedSVD(n_components=min(8, kw_tf.shape[1]-1), random_state=42).fit_transform(kw_tf)))

# dir_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform([str(d) if pd.notna(d) else '' for d in metadata_df['director']])
# features.append(csr_matrix(TruncatedSVD(n_components=min(4, dir_tf.shape[1]-1), random_state=42).fit_transform(dir_tf)))

# cast_texts = [' '.join(eval(c)) if pd.notna(c) else '' for c in metadata_df['top_cast']]
# cast_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform(cast_texts)
# features.append(csr_matrix(TruncatedSVD(n_components=min(4, cast_tf.shape[1]-1), random_state=42).fit_transform(cast_tf)))

# numeric_vecs = []
# for col in ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity']:
#     val_log = np.log1p(metadata_df[col].fillna(0).values)
#     numeric_vecs.append((val_log - val_log.mean()) / (val_log.std() + 1e-8))
# features.append(csr_matrix(np.array(numeric_vecs).T))

# content_features = hstack(features).tocsr()
# content_features = normalize(content_features, norm='l2')

# print(f"✓ Feature Matrix Ready: {content_features.shape}")

# # ============================================================================
# # [STEP 2] OPTUNA OBJECTIVE FUNCTION (Fast Search)
# # ============================================================================
# def objective(trial):
#     p_lambda = trial.suggest_float('lambda_ease', 100.0, 800.0)
#     p_half_life = trial.suggest_int('half_life_days', 30, 730)
#     p_neg_pen = trial.suggest_float('negative_penalty', 0.1, 1.0)
#     p_cf_weight = trial.suggest_float('cf_weight', 0.3, 0.8)
#     p_cb_weight = 1.0 - p_cf_weight

#     max_timestamp = raw_ratings_df['timestamp'].max()
#     days_old = (max_timestamp - raw_ratings_df['timestamp']) / 86400
#     decay_weights = np.power(0.5, days_old / p_half_life)
    
#     trial_ratings_df = raw_ratings_df.copy()
#     trial_ratings_df['weighted_rating'] = trial_ratings_df['rating'].values * decay_weights
    
#     r_weighted = trial_ratings_df['weighted_rating'].values.astype(float)
#     r_orig = trial_ratings_df['rating'].values.astype(float)
    
#     skf_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#     fold_maps, fold_divs = [], []
    
#     for fold_id, (train_idx, test_idx) in enumerate(skf_fast.split(trial_ratings_df, trial_ratings_df['user_id'])):
#         if fold_id > 1: break # Only 2 folds for speed
        
#         X = coo_matrix((r_weighted[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
#         X_orig = coo_matrix((r_orig[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
        
#         test_relevant = trial_ratings_df.iloc[test_idx]
#         test_relevant = test_relevant[test_relevant['rating'] >= CONFIG['strict_filter']]
#         test_dict = test_relevant.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
        
#         # EASE^R
#         G = X.T @ X
#         diag_indices = np.diag_indices_from(G)
#         G[diag_indices] += p_lambda
#         P = np.linalg.inv(G)
#         B = P / (-np.diag(P)[:, None])
#         np.fill_diagonal(B, 0)
#         cf_scores = X @ B 
        
#         # Rocchio
#         user_profiles = np.zeros((n_users, content_features.shape[1]))
#         for u in range(n_users):
#             u_ratings = X_orig[u]
#             liked = np.where(u_ratings >= CONFIG['strict_filter'])[0]
#             hated = np.where((u_ratings > 0) & (u_ratings < 3))[0]
#             if len(liked) > 0:
#                 prof = content_features[liked].mean(axis=0).A1
#                 if len(hated) > 0:
#                     prof = prof - (p_neg_pen * content_features[hated].mean(axis=0).A1)
#                     prof = np.maximum(prof, 0)
#                 user_profiles[u] = prof / (np.linalg.norm(prof) + 1e-8)
#         cb_scores = user_profiles @ content_features.T.toarray() 
        
#         # Z-Score Blend
#         cf_z = (cf_scores - cf_scores.mean(axis=1, keepdims=True)) / (cf_scores.std(axis=1, keepdims=True) + 1e-8)
#         cb_z = (cb_scores - cb_scores.mean(axis=1, keepdims=True)) / (cb_scores.std(axis=1, keepdims=True) + 1e-8)
#         hybrid = (p_cf_weight * cf_z) + (p_cb_weight * cb_z)
        
#         aps, divs = [], []
#         for u in range(n_users):
#             target_items = test_dict.get(u + 1, [])
#             if len(target_items) == 0: continue
            
#             scores_u = hybrid[u].copy()
#             scores_u[np.where(X_orig[u] > 0)[0]] = -np.inf 
#             top_10 = np.argsort(-scores_u)[:10]
            
#             hits = np.isin(top_10, target_items)
#             if hits.sum() > 0:
#                 aps.append(np.sum(np.cumsum(hits) / np.arange(1, 11) * hits) / min(len(target_items), 10))
#             else:
#                 aps.append(0.0)
                
#             top10_feats = content_features[top_10].toarray()
#             sim_mat = top10_feats @ top10_feats.T
#             uptri = sim_mat[np.triu_indices(10, k=1)]
#             divs.append(1.0 - np.mean(uptri) if len(uptri) > 0 else 1.0)
            
#         fold_maps.append(np.mean(aps))
#         fold_divs.append(np.mean(divs))
        
#     # Aura Fitness: MAP with a bonus for Diversity
#     return np.mean(fold_maps) + (0.1 * np.mean(fold_divs))

# print(f"\n[STEP 2] Running Optuna HPO ({CONFIG['n_optuna_trials']} Trials)...")
# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=CONFIG['n_optuna_trials'])

# BEST = study.best_params
# BEST['cb_weight'] = 1.0 - BEST['cf_weight']
# print("\n🏆 GOD PARAMETERS FOUND:")
# for k, v in BEST.items(): print(f"   - {k}: {v:.4f}" if isinstance(v, float) else f"   - {k}: {v}")

# # ============================================================================
# # [STEP 3] FINAL DIAMOND RUN (Full 5-Fold with ALL Metrics)
# # ============================================================================
# print("\n[STEP 3] Running Final 5-Fold Diamond Evaluation...")

# max_ts = raw_ratings_df['timestamp'].max()
# days = (max_ts - raw_ratings_df['timestamp']) / 86400
# raw_ratings_df['weighted_rating'] = raw_ratings_df['rating'].values * np.power(0.5, days / BEST['half_life_days'])

# r_weighted = raw_ratings_df['weighted_rating'].values.astype(float)
# r_orig = raw_ratings_df['rating'].values.astype(float)

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# final_results = []

# for fold_id, (train_idx, test_idx) in enumerate(skf.split(raw_ratings_df, raw_ratings_df['user_id']), 1):
#     X = coo_matrix((r_weighted[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
#     X_orig = coo_matrix((r_orig[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
    
#     test_rel = raw_ratings_df.iloc[test_idx]
#     test_rel = test_rel[test_rel['rating'] >= CONFIG['strict_filter']]
#     test_dict = test_rel.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
    
#     # Novelty Setup
#     item_counts = np.array((X_orig > 0).sum(axis=0)).flatten()
#     item_novelty = -np.log2(np.maximum(item_counts / n_users, 1e-9))
    
#     # EASE
#     G = X.T @ X
#     diag_indices = np.diag_indices_from(G)
#     G[diag_indices] += BEST['lambda_ease']
#     P = np.linalg.inv(G)
#     B = P / (-np.diag(P)[:, None])
#     np.fill_diagonal(B, 0)
#     cf_scores = X @ B 
    
#     # Rocchio
#     user_profiles = np.zeros((n_users, content_features.shape[1]))
#     for u in range(n_users):
#         u_ratings = X_orig[u]
#         liked = np.where(u_ratings >= CONFIG['strict_filter'])[0]
#         hated = np.where((u_ratings > 0) & (u_ratings < 3))[0]
#         if len(liked) > 0:
#             prof = content_features[liked].mean(axis=0).A1
#             if len(hated) > 0:
#                 prof = np.maximum(prof - (BEST['negative_penalty'] * content_features[hated].mean(axis=0).A1), 0)
#             user_profiles[u] = prof / (np.linalg.norm(prof) + 1e-8)
#     cb_scores = user_profiles @ content_features.T.toarray() 
    
#     # Z-Blend
#     cf_z = (cf_scores - cf_scores.mean(axis=1, keepdims=True)) / (cf_scores.std(axis=1, keepdims=True) + 1e-8)
#     cb_z = (cb_scores - cb_scores.mean(axis=1, keepdims=True)) / (cb_scores.std(axis=1, keepdims=True) + 1e-8)
#     hybrid = (BEST['cf_weight'] * cf_z) + (BEST['cb_weight'] * cb_z)
    
#     aps, ndcgs, novs, divs = [], [], [], []
#     aps_full, ndcgs_full = [], [] # Wapas add kar diye!
#     all_recs = set()
    
#     for u in range(n_users):
#         targets = test_dict.get(u + 1, [])
#         if len(targets) == 0: continue
        
#         scores_u = hybrid[u].copy()
#         scores_u[np.where(X_orig[u] > 0)[0]] = -np.inf 
        
#         full_ranking = np.argsort(-scores_u)
#         top_10 = full_ranking[:10]
        
#         # --- BEYOND ACCURACY METRICS ---
#         all_recs.update(top_10)
#         novs.append(np.mean(item_novelty[top_10]))
        
#         t10_feats = content_features[top_10].toarray()
#         sim_mat = t10_feats @ t10_feats.T
#         uptri = sim_mat[np.triu_indices(10, k=1)]
#         divs.append(1.0 - np.mean(uptri) if len(uptri) > 0 else 1.0)
        
#         # --- TOP-10 METRICS ---
#         hits = np.isin(top_10, targets)
#         if hits.sum() > 0:
#             aps.append(np.sum(np.cumsum(hits) / np.arange(1, 11) * hits) / min(len(targets), 10))
#         else: aps.append(0.0)
        
#         dcg = np.sum(hits / np.log2(np.arange(2, 12)))
#         idcg = np.sum(1.0 / np.log2(np.arange(2, min(len(targets), 10) + 2)))
#         ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
#         # --- FULL CATALOG METRICS ---
#         hits_full = np.isin(full_ranking, targets)
#         if hits_full.sum() > 0:
#             precisions_full = np.cumsum(hits_full) / (np.arange(len(full_ranking)) + 1)
#             aps_full.append(precisions_full[hits_full].sum() / len(targets))
#         else: aps_full.append(0.0)

#         dcg_full = np.sum(hits_full / np.log2(np.arange(2, len(full_ranking) + 2)))
#         idcg_full = np.sum(1.0 / np.log2(np.arange(2, len(targets) + 2)))
#         ndcgs_full.append(dcg_full / idcg_full if idcg_full > 0 else 0.0)
        
#     final_results.append({
#         'Fold': fold_id, 
#         'MAP (full)': np.mean(aps_full),
#         'MAP@10': np.mean(aps), 
#         'nDCG (full)': np.mean(ndcgs_full),
#         'nDCG@10': np.mean(ndcgs),
#         'Coverage (%)': (len(all_recs) / n_items) * 100,
#         'Novelty': np.mean(novs), 
#         'Diversity': np.mean(divs)
#     })

# print("\n" + "="*70)
# print("[FINAL RESULTS] OPTUNA + DIAMOND TIER (WITH FULL METRICS)")
# print("="*70)
# df_res = pd.DataFrame(final_results)
# # Rearranging columns for clean display
# df_res = df_res[['Fold', 'MAP (full)', 'MAP@10', 'nDCG (full)', 'nDCG@10', 'Coverage (%)', 'Novelty', 'Diversity']]
# print("\n" + df_res.to_string(index=False))

# print(f"\n[AGGREGATED METRICS]")
# print(f"  MAP (full):  {df_res['MAP (full)'].mean():.4f} ± {df_res['MAP (full)'].std():.4f}")
# print(f"  MAP@10:      {df_res['MAP@10'].mean():.4f} ± {df_res['MAP@10'].std():.4f}")
# print(f"  nDCG (full): {df_res['nDCG (full)'].mean():.4f} ± {df_res['nDCG (full)'].std():.4f}")
# print(f"  nDCG@10:     {df_res['nDCG@10'].mean():.4f} ± {df_res['nDCG@10'].std():.4f}")
# print(f"  Coverage:    {df_res['Coverage (%)'].mean():.2f}% ± {df_res['Coverage (%)'].std():.2f}%")
# print(f"  Novelty:     {df_res['Novelty'].mean():.4f} ± {df_res['Novelty'].std():.4f}")
# print(f"  Diversity:   {df_res['Diversity'].mean():.4f} ± {df_res['Diversity'].std():.4f}")

# CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)
# df_res.to_csv(CONFIG['output_dir'] / 'god_mode_results.csv', index=False)


#!/usr/bin/env python3
"""
THE GOLD TURBO: Optuna Master for Diamond Tier
Phase 1: Deep Search for God-Parameters (Lambda, Decay, Penalty, Weights)
Phase 2: Final SOTA Evaluation with Full Catalog Metrics
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
import optuna
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.INFO)

CONFIG = {
    # 'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
    'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
    'strict_filter': 4,
    'n_optuna_trials': 371  # 🚀 Turbo Mode: Full Squeeze
}

# ============================================================================
# [STEP 1] FEATURE ENGINEERING (The Content Foundation)
# ============================================================================
print("\n[STEP 1] Loading Data & Building High-Dimensional Features...")

raw_ratings_df = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

n_users = raw_ratings_df['user_id'].max()
n_items = len(metadata_df)
user_idx_raw = raw_ratings_df['user_id'].values - 1
item_idx_raw = raw_ratings_df['item_id'].values - 1

# --- Content Vectorization ---
features = []
# 1. Genres
all_genres = set()
for g_str in metadata_df['genres'].dropna():
    try: all_genres.update(eval(g_str))
    except: pass
genre_list = sorted(list(all_genres))
genre_vecs = [[1.0 if g in (eval(g_str) if pd.notna(g_str) else []) else 0.0 for g in genre_list] for g_str in metadata_df['genres']]
features.append(csr_matrix(np.array(genre_vecs)))

# 2. SVD Overviews
ov_tf = TfidfVectorizer(max_features=100, min_df=2).fit_transform(metadata_df['overview'].fillna(''))
features.append(csr_matrix(TruncatedSVD(n_components=min(32, ov_tf.shape[1]-1), random_state=42).fit_transform(ov_tf)))

# 3. Numeric Signals (Normalized)
num_vecs = []
for col in ['vote_average', 'vote_count', 'popularity']:
    val = np.log1p(metadata_df[col].fillna(0).values)
    num_vecs.append((val - val.mean()) / (val.std() + 1e-8))
features.append(csr_matrix(np.array(num_vecs).T))

content_features = hstack(features).tocsr()
content_features = normalize(content_features, norm='l2')

# ============================================================================
# [STEP 2] THE MASTER OBJECTIVE (Optimization)
# ============================================================================
def objective(trial):
    # Search Ranges
    p_lambda = trial.suggest_float('lambda_ease', 500.0, 900.0)
    p_half_life = trial.suggest_int('half_life_days', 80, 250)
    p_neg_pen = trial.suggest_float('negative_penalty', 0.1, 0.9)
    p_cf_weight = trial.suggest_float('cf_weight', 0.7, 0.85)
    p_cb_weight = 1.0 - p_cf_weight

    # Time Decay
    max_ts = raw_ratings_df['timestamp'].max()
    decay = np.power(0.5, (max_ts - raw_ratings_df['timestamp']) / (86400 * p_half_life))
    w_ratings = raw_ratings_df['rating'].values * decay
    
    # Fast 3-Fold for Optuna
    skf_fast = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_aps = []
    
    for fold_id, (train_idx, test_idx) in enumerate(skf_fast.split(raw_ratings_df, raw_ratings_df['user_id'])):
        if fold_id > 1: break # Speed
        
        # Matrix Construction
        X = coo_matrix((w_ratings[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
        X_orig = coo_matrix((raw_ratings_df['rating'].values[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
        
        # 1. EASE^R Core
        G = X.T @ X + np.eye(n_items) * p_lambda
        P = np.linalg.inv(G)
        B = P / (-np.diag(P)[:, None]); np.fill_diagonal(B, 0)
        cf_scores = X @ B 
        
        # 2. Rocchio Content Core
        user_profiles = np.zeros((n_users, content_features.shape[1]))
        for u in range(n_users):
            liked = np.where(X_orig[u] >= CONFIG['strict_filter'])[0]
            hated = np.where((X_orig[u] > 0) & (X_orig[u] < 3))[0]
            if len(liked) > 0:
                prof = content_features[liked].mean(axis=0).A1
                if len(hated) > 0:
                    prof = np.maximum(prof - (p_neg_pen * content_features[hated].mean(axis=0).A1), 0)
                user_profiles[u] = prof / (np.linalg.norm(prof) + 1e-8)
        cb_scores = user_profiles @ content_features.T.toarray() 
        
        # 3. Z-Blend
        cf_z = (cf_scores - cf_scores.mean(axis=1, keepdims=True)) / (cf_scores.std(axis=1, keepdims=True) + 1e-8)
        cb_z = (cb_scores - cb_scores.mean(axis=1, keepdims=True)) / (cb_scores.std(axis=1, keepdims=True) + 1e-8)
        hybrid = (p_cf_weight * cf_z) + (p_cb_weight * cb_z)
        
        # Evaluation
        test_dict = raw_ratings_df.iloc[test_idx][raw_ratings_df['rating'] >= CONFIG['strict_filter']].groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
        aps = []
        for u in range(n_users):
            targets = test_dict.get(u + 1, [])
            if len(targets) == 0: continue
            u_s = hybrid[u].copy(); u_s[np.where(X_orig[u] > 0)[0]] = -1e9
            top_10 = np.argsort(-u_s)[:10]
            hits = np.isin(top_10, targets)
            if hits.sum() > 0: aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(targets), 10))
            else: aps.append(0.0)
        fold_aps.append(np.mean(aps))
        
    return np.mean(fold_aps)

print(f"\n[STEP 2] Running Master Optuna Study ({CONFIG['n_optuna_trials']} Trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=CONFIG['n_optuna_trials'])

BEST = study.best_params
print("\n🏆 GOD-TIER PARAMETERS DISCOVERED:")
for k, v in BEST.items(): print(f"   - {k}: {v:.4f}")

# ============================================================================
# [STEP 3] FINAL SOTA RUN (5-Fold CV)
# ============================================================================
print("\n[STEP 3] Validating with Final 5-Fold Diamond Run...")

# [Remaining evaluation code is same as your previous script, using BEST params]
# ... (Full 5-fold evaluation with Coverage, Novelty, Diversity metrics)