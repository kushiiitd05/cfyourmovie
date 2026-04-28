#!/usr/bin/env python3
"""
THE FINAL DIAMOND PIPELINE: Optimized with God Parameters (101-Trial Optuna)
Features: Time-Decay EASE^R, Rocchio Hard-Negative Content, and Multi-Metric Evaluation
"""
# 264 optuna trials ke baad bhi kuch zyada improvement nahi mila, toh 101 trials pe hi ruk gaye.
# 🏆 GOD PARAMETERS FOUND:
#    - lambda_ease: 799.0287
#    - half_life_days: 119
#    - negative_penalty: 0.1309
#    - cf_weight: 0.7997
#    - cb_weight: 0.2003 

#0 371 trial 🏆 GOD-TIER PARAMETERS DISCOVERED:
#    - lambda_ease: 899.8477
#    - half_life_days: 119.0000
#    - negative_penalty: 0.1839
#    - cf_weight: 0.8497
#    - cb_weight: 0.1503
#  iska best 
# ======================================================================
# [FINAL RESULTS] 0  DIAMOND TIER WITH GOD PARAMETERS
# ======================================================================

#  Fold  MAP (full)   MAP@10  nDCG (full)  nDCG@10  Coverage (%)  Novelty  Diversity
#     1    0.255481 0.223221     0.558573 0.353633     24.375743 2.225198   0.812850
#     2    0.269170 0.242296     0.571672 0.377011     24.732461 2.234403   0.809724
#     3    0.257543 0.228068     0.559961 0.359802     25.029727 2.231831   0.809776
#     4    0.264286 0.233514     0.565320 0.365993     24.019025 2.230375   0.806186
#     5    0.262714 0.231400     0.565392 0.362023     25.267539 2.242723   0.805491

# [AGGREGATED METRICS]
#   MAP (full):  0.2618 ± 0.0055
#   MAP@10:      0.2317 ± 0.0071
#   nDCG (full): 0.5642 ± 0.0052
#   nDCG@10:     0.3637 ± 0.0087
#   Coverage:    24.68%
#   Diversity:   0.8088
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# --- GOD PARAMETERS (LOCKED FROM 101 TRIALS) ---
CONFIG = {
    # 'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
    'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
    # 'output_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf/lightfm_results'),
    'output_dir': Path('/home/vinayak23597/Kush/cf_final_project/lightfm_results'),
    # 'lambda_ease': 899.8477,
    # 'half_life_days': 119,
    # 'negative_penalty': 0.1839,
    # 'cf_weight': 0.8497,
    # 'cb_weight': 0.1503,
    'lambda_ease': 2199.9395646690464,
    'half_life_days': 218, 
    'negative_penalty': 0.9669969952037883, 
    'cf_weight': 0.7917321665358672, 
    'cb_weight': 0.2082678334641328,
    # 'lambda_ease': 4999.465960408743,
    # 'half_life_days': 453, 
    # 'negative_penalty': 0.7500423149701485, 
    # 'cf_weight': 0.852815429126317, 
    # 'cb_weight': 0.14718457087368297,
    'strict_filter': 4
}

print("\n[STEP 1] Loading MovieLens 100K + Enriched Metadata...")

raw_ratings_df = pd.read_csv(
    CONFIG['data_dir'] / 'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    dtype={'user_id': int, 'item_id': int, 'rating': int}
)

metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# --- TIME DECAY LOGIC (God-Tuned: 116 Days Half-Life) ---
max_timestamp = raw_ratings_df['timestamp'].max()
seconds_per_day = 86400
days_old = (max_timestamp - raw_ratings_df['timestamp']) / seconds_per_day
decay_weights = np.power(0.5, days_old / CONFIG['half_life_days'])
raw_ratings_df['weighted_rating'] = raw_ratings_df['rating'].values * decay_weights

print(f"✓ Total Ratings for Training: {len(raw_ratings_df):,}")

print("\n[STEP 2] Building Content Features (79 dims)...")
features = []
# 1. Genres
all_genres = set()
for g_str in metadata_df['genres'].dropna():
    try: all_genres.update(eval(g_str))
    except: pass
genre_list = sorted(list(all_genres))
genre_vecs = [[1.0 if g in (eval(g_str) if pd.notna(g_str) else []) else 0.0 for g in genre_list] for g_str in metadata_df['genres']]
features.append(csr_matrix(np.array(genre_vecs)))

# 2. Textual Features (Overview, Keywords)
overview_tf = TfidfVectorizer(max_features=100, min_df=2).fit_transform(metadata_df['overview'].fillna(''))
features.append(csr_matrix(TruncatedSVD(n_components=min(32, overview_tf.shape[1]-1), random_state=42).fit_transform(overview_tf)))

kw_texts = [' '.join(eval(kw)[:20]) if pd.notna(kw) else '' for kw in metadata_df['movie_keywords']]
kw_tf = TfidfVectorizer(max_features=50).fit_transform(kw_texts)
features.append(csr_matrix(TruncatedSVD(n_components=min(8, kw_tf.shape[1]-1), random_state=42).fit_transform(kw_tf)))

# 3. People Features (Director, Cast)
dir_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform([str(d) if pd.notna(d) else '' for d in metadata_df['director']])
features.append(csr_matrix(TruncatedSVD(n_components=min(4, dir_tf.shape[1]-1), random_state=42).fit_transform(dir_tf)))

cast_texts = [' '.join(eval(c)) if pd.notna(c) else '' for c in metadata_df['top_cast']]
cast_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform(cast_texts)
features.append(csr_matrix(TruncatedSVD(n_components=min(4, cast_tf.shape[1]-1), random_state=42).fit_transform(cast_tf)))

# 4. Numeric
numeric_vecs = []
for col in ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity']:
    val_log = np.log1p(metadata_df[col].fillna(0).values)
    numeric_vecs.append((val_log - val_log.mean()) / (val_log.std() + 1e-8))
features.append(csr_matrix(np.array(numeric_vecs).T))

content_features = hstack(features).tocsr()
content_features = normalize(content_features, norm='l2')

# ============================================================================
# [STEP 3] FINAL 5-FOLD EVALUATION
# ============================================================================
print("\n[STEP 3] Training 5-Fold Diamond Evaluation Pipeline...")
n_users = raw_ratings_df['user_id'].max()
n_items = len(metadata_df)
user_idx_raw = raw_ratings_df['user_id'].values - 1
item_idx_raw = raw_ratings_df['item_id'].values - 1

r_weighted = raw_ratings_df['weighted_rating'].values.astype(float)
r_orig = raw_ratings_df['rating'].values.astype(float)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_results = []

for fold_id, (train_idx, test_idx) in enumerate(skf.split(raw_ratings_df, raw_ratings_df['user_id']), 1):
    print(f"\n[FOLD {fold_id}] Training & Evaluating...")
    
    # Matrices
    X = coo_matrix((r_weighted[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
    X_orig = coo_matrix((r_orig[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
    
    # Ground Truth
    test_rel = raw_ratings_df.iloc[test_idx]
    test_rel = test_rel[test_rel['rating'] >= CONFIG['strict_filter']]
    test_dict = test_rel.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
    
    # Novelty Pre-calc
    item_counts = np.array((X_orig > 0).sum(axis=0)).flatten()
    item_novelty = -np.log2(np.maximum(item_counts / n_users, 1e-9))
    
    # 1. EASE^R Engine
    G = X.T @ X
    diag_indices = np.diag_indices_from(G)
    G[diag_indices] += CONFIG['lambda_ease']
    P = np.linalg.inv(G)
    B = P / (-np.diag(P)[:, None])
    np.fill_diagonal(B, 0)
    cf_scores = X @ B 
    
    # 2. Rocchio Content Engine
    user_profiles = np.zeros((n_users, content_features.shape[1]))
    for u in range(n_users):
        u_ratings = X_orig[u]
        liked = np.where(u_ratings >= CONFIG['strict_filter'])[0]
        hated = np.where((u_ratings > 0) & (u_ratings < 3))[0]
        if len(liked) > 0:
            prof = content_features[liked].mean(axis=0).A1
            if len(hated) > 0:
                prof = np.maximum(prof - (CONFIG['negative_penalty'] * content_features[hated].mean(axis=0).A1), 0)
            user_profiles[u] = prof / (np.linalg.norm(prof) + 1e-8)
    cb_scores = user_profiles @ content_features.T.toarray() 
    
    # 3. Z-Score Ensemble (78.7% CF / 21.3% Content)
    cf_z = (cf_scores - cf_scores.mean(axis=1, keepdims=True)) / (cf_scores.std(axis=1, keepdims=True) + 1e-8)
    cb_z = (cb_scores - cb_scores.mean(axis=1, keepdims=True)) / (cb_scores.std(axis=1, keepdims=True) + 1e-8)
    hybrid = (CONFIG['cf_weight'] * cf_z) + (CONFIG['cb_weight'] * cb_z)
    
    aps, ndcgs, novs, divs = [], [], [], []
    aps_full, ndcgs_full = [], []
    all_recs = set()
    
    for u in range(n_users):
        targets = test_dict.get(u + 1, [])
        if len(targets) == 0: continue
        
        scores_u = hybrid[u].copy()
        scores_u[np.where(X_orig[u] > 0)[0]] = -np.inf 
        
        full_ranking = np.argsort(-scores_u)
        top_10 = full_ranking[:10]
        
        # --- BEYOND ACCURACY ---
        all_recs.update(top_10)
        novs.append(np.mean(item_novelty[top_10]))
        
        t10_feats = content_features[top_10].toarray()
        sim_mat = t10_feats @ t10_feats.T
        uptri = sim_mat[np.triu_indices(10, k=1)]
        divs.append(1.0 - np.mean(uptri) if len(uptri) > 0 else 1.0)
        
        # --- ACCURACY @ 10 ---
        hits = np.isin(top_10, targets)
        if hits.sum() > 0:
            aps.append(np.sum(np.cumsum(hits) / np.arange(1, 11) * hits) / min(len(targets), 10))
        else: aps.append(0.0)
        
        dcg = np.sum(hits / np.log2(np.arange(2, 12)))
        idcg = np.sum(1.0 / np.log2(np.arange(2, min(len(targets), 10) + 2)))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
        
        # --- FULL CATALOG ACCURACY ---
        hits_full = np.isin(full_ranking, targets)
        if hits_full.sum() > 0:
            precisions_full = np.cumsum(hits_full) / (np.arange(len(full_ranking)) + 1)
            aps_full.append(precisions_full[hits_full].sum() / len(targets))
        else: aps_full.append(0.0)
        
        dcg_full = np.sum(hits_full / np.log2(np.arange(2, len(full_ranking) + 2)))
        idcg_full = np.sum(1.0 / np.log2(np.arange(2, len(targets) + 2)))
        ndcgs_full.append(dcg_full / idcg_full if idcg_full > 0 else 0.0)
        
    final_results.append({
        'Fold': fold_id, 
        'MAP (full)': np.mean(aps_full),
        'MAP@10': np.mean(aps), 
        'nDCG (full)': np.mean(ndcgs_full),
        'nDCG@10': np.mean(ndcgs),
        'Coverage (%)': (len(all_recs) / n_items) * 100,
        'Novelty': np.mean(novs), 
        'Diversity': np.mean(divs)
    })

# ============================================================================
# [STEP 4] FINAL REPORT
# ============================================================================
print("\n" + "="*70)
print("[FINAL RESULTS] DIAMOND TIER WITH GOD PARAMETERS")
print("="*70)
df_res = pd.DataFrame(final_results)
df_res = df_res[['Fold', 'MAP (full)', 'MAP@10', 'nDCG (full)', 'nDCG@10', 'Coverage (%)', 'Novelty', 'Diversity']]
print("\n" + df_res.to_string(index=False))

print(f"\n[AGGREGATED METRICS]")
print(f"  MAP (full):  {df_res['MAP (full)'].mean():.4f} ± {df_res['MAP (full)'].std():.4f}")
print(f"  MAP@10:      {df_res['MAP@10'].mean():.4f} ± {df_res['MAP@10'].std():.4f}")
print(f"  nDCG (full): {df_res['nDCG (full)'].mean():.4f} ± {df_res['nDCG (full)'].std():.4f}")
print(f"  nDCG@10:     {df_res['nDCG@10'].mean():.4f} ± {df_res['nDCG@10'].std():.4f}")
print(f"  Coverage:    {df_res['Coverage (%)'].mean():.2f}%")
print(f"  Diversity:   {df_res['Diversity'].mean():.4f}")

CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)
df_res.to_csv(CONFIG['output_dir'] / 'diamond_cv_results.csv', index=False)