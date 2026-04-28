#!/usr/bin/env python3
"""
THE PLATINUM PIPELINE: Time-Decayed EASE^R + Rocchio Content Ensemble
Train on full graph -> Evaluate STRICTLY on hits (Rating >= 4)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    # 'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
    'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
    # 'output_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf/lightfm_results'),
    'output_dir': Path('/home/vinayak23597/Kush/cf_final_project/lightfm_results'),
    # 'lambda_ease': 500,     
    # 'half_life_days': 365,  # 1 year half-life for Time-Decay
    # 'negative_penalty': 0.5, # Rocchio penalty for 1-2 star movies
    'lambda_ease': 2199.9395646690464,
    'half_life_days': 218, 
    'negative_penalty': 0.9669969952037883,
    'strict_filter': 4
}
 
    # 'cf_weight': 0.7917321665358672, 
    # 'cb_weight': 0.2082678334641328,

print("\n[STEP 1] Loading MovieLens 100K + Enriched Metadata...")

# Load ALL ratings for dense training graph
ratings_df = pd.read_csv(
    CONFIG['data_dir'] / 'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    dtype={'user_id': int, 'item_id': int, 'rating': int}
)

metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# --- TIME DECAY LOGIC (Inspired by SAR) ---
# Weight recent ratings higher. Half-life means a rating 1 year old is worth 50%
max_timestamp = ratings_df['timestamp'].max()
seconds_per_day = 86400
days_old = (max_timestamp - ratings_df['timestamp']) / seconds_per_day
decay_weights = np.power(0.5, days_old / CONFIG['half_life_days'])
# Weighted Rating = Original Rating * Time Decay Weight
ratings_df['weighted_rating'] = ratings_df['rating'].values * decay_weights

print(f"✓ Total Ratings for Training: {len(ratings_df):,}")

print("\n[STEP 2] Building Content Features (79 dims)...")
features = []
all_genres = set()
for g_str in metadata_df['genres'].dropna():
    try: all_genres.update(eval(g_str))
    except: pass
genre_list = sorted(list(all_genres))
genre_vecs = []
for g_str in metadata_df['genres']:
    try: genres = set(eval(g_str))
    except: genres = set()
    genre_vecs.append([1.0 if g in genres else 0.0 for g in genre_list])
features.append(csr_matrix(np.array(genre_vecs)))

overview_tf = TfidfVectorizer(max_features=100, min_df=2).fit_transform(metadata_df['overview'].fillna(''))
features.append(csr_matrix(TruncatedSVD(n_components=min(32, overview_tf.shape[1]-1), random_state=42).fit_transform(overview_tf)))

kw_texts = [' '.join(eval(kw)[:20]) if not pd.isna(kw) else '' for kw in metadata_df['movie_keywords']]
kw_tf = TfidfVectorizer(max_features=50).fit_transform(kw_texts)
features.append(csr_matrix(TruncatedSVD(n_components=min(8, kw_tf.shape[1]-1), random_state=42).fit_transform(kw_tf)))

dir_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform([str(d) if not pd.isna(d) else '' for d in metadata_df['director']])
features.append(csr_matrix(TruncatedSVD(n_components=min(4, dir_tf.shape[1]-1), random_state=42).fit_transform(dir_tf)))

cast_texts = [' '.join(eval(c)) if not pd.isna(c) else '' for c in metadata_df['top_cast']]
cast_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform(cast_texts)
features.append(csr_matrix(TruncatedSVD(n_components=min(4, cast_tf.shape[1]-1), random_state=42).fit_transform(cast_tf)))

numeric_vecs = []
for col in ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity']:
    val_log = np.log1p(metadata_df[col].fillna(0).values)
    numeric_vecs.append((val_log - val_log.mean()) / (val_log.std() + 1e-8))
features.append(csr_matrix(np.array(numeric_vecs).T))

content_features = hstack(features).tocsr()
content_features = normalize(content_features, norm='l2')

print("\n[STEP 3] Training 5-Fold Cross-Validation Pipeline...")
from sklearn.model_selection import StratifiedKFold

n_users = ratings_df['user_id'].max()
n_items = len(metadata_df)
user_idx = ratings_df['user_id'].values - 1
item_idx = ratings_df['item_id'].values - 1

# Using the new weighted ratings for the CF matrix
ratings_weighted = ratings_df['weighted_rating'].values.astype(float)
# Keeping original ratings array for the Truth Filter & Rocchio logic
ratings_original = ratings_df['rating'].values.astype(float)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(ratings_df, ratings_df['user_id']))

results = []

for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
    print(f"\n[FOLD {fold_id}] Training...")
    
    # EASE^R sees the TIME-DECAYED matrix
    train_interactions = coo_matrix((ratings_weighted[train_idx], (user_idx[train_idx], item_idx[train_idx])), shape=(n_users, n_items)).tocsr()
    # Content Engine logic needs the ORIGINAL ratings to know what was a 1-star vs 5-star
    train_interactions_orig = coo_matrix((ratings_original[train_idx], (user_idx[train_idx], item_idx[train_idx])), shape=(n_users, n_items)).tocsr()
    
    test_fold_df = ratings_df.iloc[test_idx]
    test_relevant_df = test_fold_df[test_fold_df['rating'] >= CONFIG['strict_filter']]
    test_relevant_dict = test_relevant_df.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
    
    X = train_interactions.toarray() 
    X_orig = train_interactions_orig.toarray()
    
    # 1. TIME-DECAYED EASE^R Engine
    G = X.T @ X
    diag_indices = np.diag_indices_from(G)
    G[diag_indices] += CONFIG['lambda_ease']
    P = np.linalg.inv(G)
    B = P / (-np.diag(P)[:, None])
    np.fill_diagonal(B, 0)
    cf_scores = X @ B 
    
    # 2. ROCCHIO CONTENT ENGINE (Hard Negative Mining)
    user_profiles = np.zeros((n_users, content_features.shape[1]))
    for u in range(n_users):
        u_ratings = X_orig[u]
        
        # Positives (Ratings >= 4)
        liked_items = np.where(u_ratings >= CONFIG['strict_filter'])[0]
        # Hard Negatives (Ratings 1 or 2) -> We ignore 3 (Neutral)
        hated_items = np.where((u_ratings > 0) & (u_ratings < 3))[0]
        
        if len(liked_items) > 0:
            prof_liked = content_features[liked_items].mean(axis=0).A1
            
            # If user has hated movies, subtract their features!
            if len(hated_items) > 0:
                prof_hated = content_features[hated_items].mean(axis=0).A1
                # The Rocchio Formula
                prof = prof_liked - (CONFIG['negative_penalty'] * prof_hated)
                # Apply ReLU (Turn any negative values to 0)
                prof = np.maximum(prof, 0)
            else:
                prof = prof_liked
                
            user_profiles[u] = prof / (np.linalg.norm(prof) + 1e-8)
            
    cb_scores = user_profiles @ content_features.T.toarray() 
    
    # 3. Z-Score Scaling & Ensemble
    cf_mean = cf_scores.mean(axis=1, keepdims=True)
    cf_std = cf_scores.std(axis=1, keepdims=True) + 1e-8
    cf_scores_z = (cf_scores - cf_mean) / cf_std
    
    cb_mean = cb_scores.mean(axis=1, keepdims=True)
    cb_std = cb_scores.std(axis=1, keepdims=True) + 1e-8
    cb_scores_z = (cb_scores - cb_mean) / cb_std
    
    hybrid_scores = 0.6 * cf_scores_z + 0.4 * cb_scores_z
    
    aps, aps_full, recalls, ndcgs, ndcgs_full = [], [], [], [], []
    
    # 4. Evaluation Loop
    for u in range(n_users):
        test_items_u = test_relevant_dict.get(u + 1, np.array([]))
        if len(test_items_u) == 0:
            continue
            
        scores_u = hybrid_scores[u].copy()
        rated_items_u = train_interactions_orig[u].nonzero()[1]
        scores_u[rated_items_u] = -np.inf 
        
        full_ranking = np.argsort(-scores_u)
        top_10 = full_ranking[:10]
        
        hits_10 = np.isin(top_10, test_items_u)
        if hits_10.sum() > 0:
            precisions = np.cumsum(hits_10) / (np.arange(10) + 1)
            aps.append(precisions[hits_10].sum() / min(len(test_items_u), 10))
        else:
            aps.append(0.0)
            
        recalls.append(hits_10.sum() / len(test_items_u))
        
        dcg_10 = np.sum(hits_10 / np.log2(np.arange(2, 12)))
        idcg_10 = np.sum(1.0 / np.log2(np.arange(2, min(len(test_items_u), 10) + 2)))
        ndcgs.append(dcg_10 / idcg_10 if idcg_10 > 0 else 0.0)

        hits_full = np.isin(full_ranking, test_items_u)
        if hits_full.sum() > 0:
            precisions_full = np.cumsum(hits_full) / (np.arange(len(full_ranking)) + 1)
            aps_full.append(precisions_full[hits_full].sum() / len(test_items_u))
        else:
            aps_full.append(0.0)

        dcg_full = np.sum(hits_full / np.log2(np.arange(2, len(full_ranking) + 2)))
        idcg_full = np.sum(1.0 / np.log2(np.arange(2, len(test_items_u) + 2)))
        ndcgs_full.append(dcg_full / idcg_full if idcg_full > 0 else 0.0)
    
    fold_results = {
        'Fold': fold_id, 
        'MAP (full)': np.mean(aps_full),
        'MAP@10': np.mean(aps), 
        'Recall@10': np.mean(recalls), 
        'nDCG (full)': np.mean(ndcgs_full),
        'nDCG@10': np.mean(ndcgs)
    }
    
    print(f"  ✓ MAP (full):  {fold_results['MAP (full)']:.4f}")
    print(f"  ✓ MAP@10:      {fold_results['MAP@10']:.4f}")
    print(f"  ✓ Recall@10:   {fold_results['Recall@10']:.4f}")
    print(f"  ✓ nDCG (full): {fold_results['nDCG (full)']:.4f}")
    print(f"  ✓ nDCG@10:     {fold_results['nDCG@10']:.4f}")
    
    results.append(fold_results)

print("\n" + "="*70)
print("[STEP 4] FINAL RESULTS (PLATINUM TIER)")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

print(f"\n[AGGREGATED]")
print(f"  MAP (full):  {results_df['MAP (full)'].mean():.4f} ± {results_df['MAP (full)'].std():.4f}")
print(f"  MAP@10:      {results_df['MAP@10'].mean():.4f} ± {results_df['MAP@10'].std():.4f}")
print(f"  Recall@10:   {results_df['Recall@10'].mean():.4f} ± {results_df['Recall@10'].std():.4f}")
print(f"  nDCG (full): {results_df['nDCG (full)'].mean():.4f} ± {results_df['nDCG (full)'].std():.4f}")
print(f"  nDCG@10:     {results_df['nDCG@10'].mean():.4f} ± {results_df['nDCG@10'].std():.4f}")

CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)
results_df.to_csv(CONFIG['output_dir'] / 'platinum_cv_results.csv', index=False)