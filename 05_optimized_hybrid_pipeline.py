#!/usr/bin/env python3
"""
Optimized Movie Recommendation Pipeline
Using Content-Based + Collaborative Filtering Ensemble

Movies with MovieLens 100K enriched metadata (17 features)
Target: MAP@10 ≥ 0.135
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
    'output_dir': Path('/home/vinayak23597/Kush/cf_final_project/BKLRESULTS'),
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[STEP 1] Loading MovieLens 100K + Enriched Metadata...")

# Load ratings
ratings_df = pd.read_csv(
    CONFIG['data_dir'] / 'ml-100k/u.data',
    sep='\t',
    names=['user_id', 'item_id', 'rating', 'timestamp'],
    dtype={'user_id': int, 'item_id': int, 'rating': int}
)

# Load metadata
metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

print(f"✓ Ratings: {len(ratings_df):,}")
print(f"✓ Movies: {len(metadata_df)}")
print(f"✓ Users: {ratings_df['user_id'].nunique()}")

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

print("\n[STEP 2] Building Content Features (79 dims)...")

n_items = len(metadata_df)
features = []

# 1. Genres (26 dims from parsing)
print("  - Genres [26]...", end=" ")
all_genres = set()
for g_str in metadata_df['genres'].dropna():
    try:
        all_genres.update(eval(g_str))
    except:
        pass

genre_list = sorted(list(all_genres))
genre_vecs = []
for g_str in metadata_df['genres']:
    try:
        genres = set(eval(g_str))
    except:
        genres = set()
    genre_vecs.append([1.0 if g in genres else 0.0 for g in genre_list])

genres_mat = csr_matrix(np.array(genre_vecs))
features.append(genres_mat)
print(f"✓ {genres_mat.shape[1]} features")

# 2. Overview (32 dims)
print("  - Overview [32]...", end=" ")
overviews = metadata_df['overview'].fillna('').tolist()
tfidf = TfidfVectorizer(max_features=100, min_df=2)
overview_tf = tfidf.fit_transform(overviews)
svd_ov = TruncatedSVD(n_components=min(32, overview_tf.shape[1]-1), random_state=42)
overview_mat = csr_matrix(svd_ov.fit_transform(overview_tf))
features.append(overview_mat)
print(f"✓ {overview_mat.shape[1]} features")

# 3. Keywords (8 dims)
print("  - Keywords [8]...", end=" ")
kw_texts = []
for kw_str in metadata_df['movie_keywords']:
    try:
        kw_texts.append(' '.join(eval(kw_str)[:20]) if not pd.isna(kw_str) else '')
    except:
        kw_texts.append('')

tfidf_kw = TfidfVectorizer(max_features=50)
kw_tf = tfidf_kw.fit_transform(kw_texts)
svd_kw = TruncatedSVD(n_components=min(8, kw_tf.shape[1]-1), random_state=42)
keywords_mat = csr_matrix(svd_kw.fit_transform(kw_tf))
features.append(keywords_mat)
print(f"✓ {keywords_mat.shape[1]} features")

# 4. Director (4 dims)
print("  - Director [4]...", end=" ")
dirs = [str(d) if not pd.isna(d) else '' for d in metadata_df['director']]
tfidf_dir = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30)
dir_tf = tfidf_dir.fit_transform(dirs)
svd_dir = TruncatedSVD(n_components=min(4, dir_tf.shape[1]-1), random_state=42)
dir_mat = csr_matrix(svd_dir.fit_transform(dir_tf))
features.append(dir_mat)
print(f"✓ {dir_mat.shape[1]} features")

# 5. Cast (4 dims)
print("  - Cast [4]...", end=" ")
cast_texts = []
for c_str in metadata_df['top_cast']:
    try:
        cast_texts.append(' '.join(eval(c_str)) if not pd.isna(c_str) else '')
    except:
        cast_texts.append('')

tfidf_cast = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30)
cast_tf = tfidf_cast.fit_transform(cast_texts)
svd_cast = TruncatedSVD(n_components=min(4, cast_tf.shape[1]-1), random_state=42)
cast_mat = csr_matrix(svd_cast.fit_transform(cast_tf))
features.append(cast_mat)
print(f"✓ {cast_mat.shape[1]} features")

# 6. Numeric (5 dims)
print("  - Numeric [5]...", end=" ")
numeric_cols = ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity']
numeric_vecs = []
for col in numeric_cols:
    val = metadata_df[col].fillna(0).values
    val_log = np.log1p(val)
    val_norm = (val_log - val_log.mean()) / (val_log.std() + 1e-8)
    numeric_vecs.append(val_norm)

numeric_mat = csr_matrix(np.array(numeric_vecs).T)
features.append(numeric_mat)
print(f"✓ {numeric_mat.shape[1]} features")

# Combine all
from scipy.sparse import hstack
content_features = hstack(features).tocsr()
content_features = normalize(content_features, norm='l2')

print(f"\n✓ Content feature matrix: {content_features.shape}")
print(f"✓ Sparsity: {1 - content_features.nnz / (content_features.shape[0] * content_features.shape[1]):.1%}")

# ============================================================================
# SIMPLE EVALUATION: COSINE SIMILARITY + RATING FEEDBACK
# ============================================================================

print("\n[STEP 3] Training 5-Fold Cross-Validation Pipeline...")
print("="*70)

from sklearn.model_selection import StratifiedKFold

n_users = ratings_df['user_id'].max()
n_items = len(metadata_df)

# Build interaction matrix
user_idx = ratings_df['user_id'].values - 1
item_idx = ratings_df['item_id'].values - 1
ratings = ratings_df['rating'].values.astype(float)

all_interactions = coo_matrix((ratings, (user_idx, item_idx)),
                              shape=(n_users, n_items)).tocsr()

# Create folds
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(ratings_df, ratings_df['user_id']))

results = []

for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
    print(f"\n[FOLD {fold_id}] Training...")
    
    # Split data
    train_ratings = ratings[train_idx]
    train_users = user_idx[train_idx]
    train_items = item_idx[train_idx]
    
    test_ratings = ratings[test_idx]
    test_users = user_idx[test_idx]
    test_items = item_idx[test_idx]
    
    # Build fold matrices
    train_interactions = coo_matrix((train_ratings, (train_users, train_items)),
                                   shape=(n_users, n_items)).tocsr()
    test_interactions = coo_matrix((test_ratings, (test_users, test_items)),
                                  shape=(n_users, n_items)).tocsr()
    
    # === CONTENT-BASED RECOMMENDER ===
    # Compute user-content profiles: average of rated item features
    user_profiles = []
    for u in range(n_users):
        rated_items = train_interactions[u].nonzero()[1]
        if len(rated_items) > 0:
            profile = content_features[rated_items].mean(axis=0).A1
            profile = profile / (np.linalg.norm(profile) + 1e-8)
        else:
            profile = np.zeros(content_features.shape[1])
        user_profiles.append(profile)
    
    user_profiles = np.array(user_profiles)
    
    # Compute scores: user_profile @ content_features.T
    content_scores = user_profiles @ content_features.T  # [n_users, n_items]
    
    # === COLLABORATIVE FILTERING RECOMMENDER ===
    # Simple: item similarity based on user ratings
    # item_similarity[i,j] = correlation of rating patterns
    item_user_mat = train_interactions.T.tocsr()  # [n_items, n_users]
    
    # Normalize
    item_user_norm = normalize(item_user_mat, norm='l2')
    item_similarity = item_user_norm @ item_user_norm.T  # [n_items, n_items]
    
    # User-item scores from CF
    cf_scores = train_interactions @ item_similarity  # [n_users, n_items]
    
    # === BLEND ===
    # Hybrid score
    hybrid_scores = 0.4 * content_scores + 0.6 * cf_scores
    
    # === EVALUATION ===
    print(f"  Evaluating on {test_interactions.nnz:,} test interactions...")
    
    aps = []
    aps_full = []
    recalls = []
    ndcgs = []
    ndcgs_full = []
    
    for u in range(n_users):
        # Ground truth in test set
        test_items_u = test_interactions[u].nonzero()[1]
        
        if len(test_items_u) == 0:
            continue
        
        # Exclude already-rated items
        scores_u = np.array(hybrid_scores[u]).flatten()  # Convert to array
        rated_items_u = train_interactions[u].nonzero()[1]
        scores_u[rated_items_u] = -np.inf
        
        # Full ranking (all items)
        full_ranking = np.argsort(-scores_u)
        
        # Top-10 recommendations
        top_10 = full_ranking[:10]
        
        # Compute AP@10
        hits = np.isin(top_10, test_items_u)
        if hits.sum() > 0:
            precisions = np.cumsum(hits) / (np.arange(len(top_10)) + 1)
            ap = precisions[hits].sum() / min(len(test_items_u), 10)
        else:
            ap = 0.0
        aps.append(ap)
        
        # Compute full MAP (no limit on ranking depth)
        hits_full = np.isin(full_ranking, test_items_u)
        if hits_full.sum() > 0:
            precisions_full = np.cumsum(hits_full) / (np.arange(len(full_ranking)) + 1)
            ap_full = precisions_full[hits_full].sum() / len(test_items_u)
        else:
            ap_full = 0.0
        aps_full.append(ap_full)
        
        # Compute Recall@10
        recall = hits.sum() / len(test_items_u)
        recalls.append(recall)
        
        # Compute nDCG@10
        # DCG = sum of (relevance / log2(rank + 1))
        dcg = np.sum(hits / np.log2(np.arange(2, len(top_10) + 2)))
        # Ideal DCG = perfect ranking of all test items
        idcg = np.sum(1.0 / np.log2(np.arange(2, min(len(test_items_u), 10) + 2)))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        ndcgs.append(ndcg)
        
        # Compute nDCG (full) - across all ranking positions
        dcg_full = np.sum(hits_full / np.log2(np.arange(2, len(full_ranking) + 2)))
        idcg_full = np.sum(1.0 / np.log2(np.arange(2, len(test_items_u) + 2)))
        ndcg_full = dcg_full / idcg_full if idcg_full > 0 else 0.0
        ndcgs_full.append(ndcg_full)
    
    map_at_10 = np.mean(aps) if aps else 0.0
    map_full = np.mean(aps_full) if aps_full else 0.0
    recall_at_10 = np.mean(recalls) if recalls else 0.0
    ndcg_at_10 = np.mean(ndcgs) if ndcgs else 0.0
    ndcg_full = np.mean(ndcgs_full) if ndcgs_full else 0.0
    
    print(f"  ✓ MAP (full):  {map_full:.4f}")
    print(f"  ✓ MAP@10:      {map_at_10:.4f}")
    print(f"  ✓ Recall@10:   {recall_at_10:.4f}")
    print(f"  ✓ nDCG@10:     {ndcg_at_10:.4f}")
    print(f"  ✓ nDCG (full): {ndcg_full:.4f}")
    
    results.append({
        'Fold': fold_id,
        'MAP': map_full,
        'MAP@10': map_at_10,
        'Recall@10': recall_at_10,
        'nDCG@10': ndcg_at_10,
        'nDCG': ndcg_full,
    })

# ============================================================================
# FINAL RESULTS
# ============================================================================

print("\n" + "="*70)
print("[STEP 4] FINAL RESULTS")
print("="*70)

results_df = pd.DataFrame(results)
print("\n" + results_df.to_string(index=False))

map_mean = results_df['MAP'].mean()
map_std = results_df['MAP'].std()
map_at_10_mean = results_df['MAP@10'].mean()
map_at_10_std = results_df['MAP@10'].std()
recall_mean = results_df['Recall@10'].mean()
recall_std = results_df['Recall@10'].std()
ndcg_mean = results_df['nDCG@10'].mean()
ndcg_std = results_df['nDCG@10'].std()
ndcg_full_mean = results_df['nDCG'].mean()
ndcg_full_std = results_df['nDCG'].std()

print(f"\n[AGGREGATED]")
print(f"  MAP (full):   {map_mean:.4f} ± {map_std:.4f}")
print(f"  MAP@10:       {map_at_10_mean:.4f} ± {map_at_10_std:.4f}")
print(f"  Recall@10:    {recall_mean:.4f} ± {recall_std:.4f}")
print(f"  nDCG@10:      {ndcg_mean:.4f} ± {ndcg_std:.4f}")
print(f"  nDCG (full):  {ndcg_full_mean:.4f} ± {ndcg_full_std:.4f}")

# Save results
CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)
results_df.to_csv(CONFIG['output_dir'] / 'hybrid_cv_results.csv', index=False)

print(f"\n✓ Results saved to lightfm_results/hybrid_cv_results.csv")
print("="*70 + "\n")
