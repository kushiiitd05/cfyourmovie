#!/usr/bin/env python3
"""
THE ADAMANTIUM V2 ULTIMATE SUITE: SVD-Refined Two-Stage + Full Diamond Metrics
Fixes: SVD Noise Cancellation, Regularized LGBM, and Full Evaluation Dashboard.
"""

import numpy as np
import pandas as pd
import ast
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

CONFIG = {
    # 'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
    'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
    # 'output_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf/lightfm_results'),
    'output_dir': Path('/home/vinayak23597/Kush/cf_final_project/lightfm_results'),
    'lambda_ease': 798.7934,
    'half_life_days': 116,
    'strict_filter': 4,
    'top_n_candidates': 100,
    'latent_dims': 100  # 🚀 Noise-Free DNA
    
}

# ============================================================================
# [STEP 1] SVD-POWERED FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 1] Forging Refined Adamantium Features (SVD Compression)...")

ratings_df = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
users_df = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.user', sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip'])
for col in ['gender', 'occupation']: users_df[col] = LabelEncoder().fit_transform(users_df[col])

metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

def clean_list(x):
    try: return " ".join([str(i).replace(" ", "_") for i in ast.literal_eval(x)])
    except: return ""

metadata_df['clean_genres'] = metadata_df['genres'].apply(clean_list)
metadata_df['clean_keywords'] = metadata_df['movie_keywords'].apply(clean_list)
metadata_df['clean_cast'] = metadata_df['top_cast'].apply(clean_list)

# Vectorizing
cv_gen = CountVectorizer(binary=True).fit_transform(metadata_df['clean_genres'])
cv_kw = CountVectorizer(binary=True, min_df=3).fit_transform(metadata_df['clean_keywords'])
cv_cast = CountVectorizer(binary=True, min_df=2).fit_transform(metadata_df['clean_cast'])

# 🚀 SVD to kill the 2164 -> 100 noise
full_sparse = hstack([cv_gen, cv_kw, cv_cast]).tocsr()
svd = TruncatedSVD(n_components=CONFIG['latent_dims'], random_state=42)
latent_feats = svd.fit_transform(full_sparse)

# Numerics
num_cols = ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity', 'year']
df_num = metadata_df[num_cols].fillna(0)
for col in ['budget', 'revenue']: df_num[col] = np.log1p(df_num[col])
feat_num = MinMaxScaler().fit_transform(df_num)

# Final Adamantium V2 Matrix
feature_matrix = normalize(np.hstack([latent_feats, feat_num]), norm='l2')
print(f"✓ Feature Matrix Ready: {feature_matrix.shape}")

# ============================================================================
# [STEP 2] THE FULL-DASHBOARD EVALUATION
# ============================================================================
n_users, n_items = ratings_df['user_id'].max(), feature_matrix.shape[0]
user_idx_raw, item_idx_raw = ratings_df['user_id'].values - 1, ratings_df['item_id'].values - 1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
final_results = []

for fold_id, (train_idx, test_idx) in enumerate(skf.split(ratings_df, ratings_df['user_id']), 1):
    print(f"\n[FOLD {fold_id}] Training Stage-1 & Stage-2...")
    
    # --- STAGE 1: RETRIEVAL ---
    max_ts = ratings_df['timestamp'].max()
    decay = np.power(0.5, (max_ts - ratings_df['timestamp']) / (86400 * CONFIG['half_life_days']))
    w_ratings = ratings_df['rating'].values * decay
    X = coo_matrix((w_ratings[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
    X_orig = coo_matrix((ratings_df['rating'].values[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
    
    item_counts = np.array((X_orig > 0).sum(axis=0)).flatten()
    item_novelty = -np.log2(np.maximum(item_counts / n_users, 1e-9))
    
    G = X.T @ X + np.eye(n_items) * CONFIG['lambda_ease']
    B = np.linalg.inv(G); B /= -np.diag(B)[:, None]; np.fill_diagonal(B, 0)
    retrieval_scores = X @ B 

    # --- STAGE 2: LIGHTGBM (Regularized) ---
    train_samples, train_labels = [], []
    for u in range(n_users):
        pos = np.where(X_orig[u] >= 4)[0]
        neg = np.where((X_orig[u] > 0) & (X_orig[u] < 3))[0]
        for i in list(pos) + list(neg):
            feat_vec = np.concatenate([[retrieval_scores[u, i]], [users_df.iloc[u]['age'], users_df.iloc[u]['gender'], users_df.iloc[u]['occupation']], feature_matrix[i]])
            train_samples.append(feat_vec)
            train_labels.append(1 if i in pos else 0)

    ranker = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, reg_alpha=0.1, reg_lambda=0.1, verbose=-1)
    ranker.fit(np.array(train_samples), train_labels)

    # --- EVALUATION DASHBOARD ---
    test_rel = ratings_df.iloc[test_idx]
    test_dict = test_rel[test_rel['rating'] >= 4].groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
    
    metrics = {'aps': [], 'ndcgs': [], 'novs': [], 'divs': [], 'aps_f': [], 'ndcgs_f': []}
    all_recs_fold = set()
    
    for u in range(n_users):
        targets = test_dict.get(u + 1, [])
        if len(targets) == 0: continue
        
        # Retrieval
        u_scores = retrieval_scores[u].copy()
        u_scores[np.where(X_orig[u] > 0)[0]] = -np.inf
        candidates = np.argsort(-u_scores)[:CONFIG['top_n_candidates']]
        
        # Re-ranking
        rerank_feats = [np.concatenate([[retrieval_scores[u, i]], [users_df.iloc[u]['age'], users_df.iloc[u]['gender'], users_df.iloc[u]['occupation']], feature_matrix[i]]) for i in candidates]
        probs = ranker.predict_proba(np.array(rerank_feats))[:, 1]
        top_10 = candidates[np.argsort(-probs)[:10]]
        
        # Full Rank Assembly
        full_ranking = np.concatenate([top_10, np.delete(np.argsort(-u_scores), np.where(np.isin(np.argsort(-u_scores), top_10)))])
        
        # Diamond Metrics
        all_recs_fold.update(top_10)
        metrics['novs'].append(np.mean(item_novelty[top_10]))
        t10_feats = feature_matrix[top_10]
        sim_mat = t10_feats @ t10_feats.T
        uptri = sim_mat[np.triu_indices(10, k=1)]
        metrics['divs'].append(1.0 - np.mean(uptri) if len(uptri) > 0 else 1.0)
        
        # Accuracy Metrics
        hits_10 = np.isin(top_10, targets)
        if hits_10.sum() > 0:
            metrics['aps'].append(np.sum(np.cumsum(hits_10) / np.arange(1, 11) * hits_10) / min(len(targets), 10))
            metrics['ndcgs'].append(np.sum(hits_10 / np.log2(np.arange(2, 12))) / np.sum(1.0 / np.log2(np.arange(2, min(len(targets), 10) + 2))))
        else:
            metrics['aps'].append(0.0); metrics['ndcgs'].append(0.0)
            
        # Full Accuracy
        hits_f = np.isin(full_ranking, targets)
        if hits_f.sum() > 0:
            metrics['aps_f'].append(np.sum(np.cumsum(hits_f) / (np.arange(len(full_ranking)) + 1) * hits_f) / len(targets))
            metrics['ndcgs_f'].append(np.sum(hits_f / np.log2(np.arange(2, len(full_ranking) + 2))) / np.sum(1.0 / np.log2(np.arange(2, len(targets) + 2))))
        else:
            metrics['aps_f'].append(0.0); metrics['ndcgs_f'].append(0.0)

    final_results.append({
        'Fold': fold_id, 'MAP (full)': np.mean(metrics['aps_f']), 'MAP@10': np.mean(metrics['aps']), 
        'nDCG (full)': np.mean(metrics['ndcgs_f']), 'nDCG@10': np.mean(metrics['ndcgs']),
        'Coverage (%)': (len(all_recs_fold) / n_items) * 100,
        'Novelty': np.mean(metrics['novs']), 'Diversity': np.mean(metrics['divs'])
    })
    print(f"   ✓ Fold {fold_id} Refined MAP@10: {np.mean(metrics['aps']):.4f}")

# ============================================================================
# [STEP 3] THE FINAL REPORT
# ============================================================================
print("\n" + "="*80)
print("🏆 ADAMANTIUM V2: THE ULTIMATE SUITE (REFINED)")
print("="*80)
df_res = pd.DataFrame(final_results)
print("\n" + df_res[['Fold', 'MAP (full)', 'MAP@10', 'nDCG (full)', 'nDCG@10', 'Coverage (%)', 'Novelty', 'Diversity']].to_string(index=False))

print(f"\n[AGGREGATED METRICS]")
for m in ['MAP (full)', 'MAP@10', 'nDCG (full)', 'nDCG@10', 'Coverage (%)', 'Novelty', 'Diversity']:
    print(f"  {m:12}: {df_res[m].mean():.4f} ± {df_res[m].std():.4f}")