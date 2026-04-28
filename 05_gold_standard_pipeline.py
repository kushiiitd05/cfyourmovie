# #!/usr/bin/env python3
# """
# THE GOLD STANDARD PIPELINE: EASE^R + Content Adaptive Router
# Movies with MovieLens 100K enriched metadata (79 features)
# Strict Truth Filter Applied: Only Ratings >= 4 considered.
# """

# import numpy as np
# import pandas as pd
# from pathlib import Path
# from scipy.sparse import csr_matrix, coo_matrix, hstack
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import normalize
# import warnings
# warnings.filterwarnings('ignore')

# CONFIG = {
#     'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
#     'output_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf/lightfm_results'),
#     'lambda_ease': 500,     # Regularization for EASE^R
#     'cold_threshold': 5,    # n >= 5 is warm (EASE^R), n < 5 is cold (Content)
#     'strict_filter': 4      # Only rating >= 4 is a HIT
# }

# # ============================================================================
# # [STEP 1] LOAD & STRICTLY FILTER DATA
# # ============================================================================

# print("\n[STEP 1] Loading MovieLens 100K + Enriched Metadata...")

# # Load ratings
# raw_ratings_df = pd.read_csv(
#     CONFIG['data_dir'] / 'ml-100k/u.data',
#     sep='\t',
#     names=['user_id', 'item_id', 'rating', 'timestamp'],
#     dtype={'user_id': int, 'item_id': int, 'rating': int}
# )

# # THE TRUTH FILTER: Only keep ratings >= 4
# # Yeh ensure karega ki user profiles mein sirf pasand aayi hui movies ka influence ho
# ratings_df = raw_ratings_df[raw_ratings_df['rating'] >= CONFIG['strict_filter']]

# # Load metadata
# metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
# metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# print(f"✓ Total Raw Ratings: {len(raw_ratings_df):,}")
# print(f"✓ STRICT Truth Ratings (>=4): {len(ratings_df):,} ({(len(ratings_df)/len(raw_ratings_df)*100):.1f}%)")
# print(f"✓ Movies: {len(metadata_df)}")
# print(f"✓ Users: {ratings_df['user_id'].nunique()}")

# # ============================================================================
# # [STEP 2] FEATURE ENGINEERING (Exactly 79 Dims as your code)
# # ============================================================================

# print("\n[STEP 2] Building Content Features (79 dims)...")

# features = []

# print("  - Genres [26]...", end=" ")
# all_genres = set()
# for g_str in metadata_df['genres'].dropna():
#     try: all_genres.update(eval(g_str))
#     except: pass
# genre_list = sorted(list(all_genres))
# genre_vecs = []
# for g_str in metadata_df['genres']:
#     try: genres = set(eval(g_str))
#     except: genres = set()
#     genre_vecs.append([1.0 if g in genres else 0.0 for g in genre_list])
# features.append(csr_matrix(np.array(genre_vecs)))

# print("  - Overview [32]...", end=" ")
# overview_tf = TfidfVectorizer(max_features=100, min_df=2).fit_transform(metadata_df['overview'].fillna(''))
# features.append(csr_matrix(TruncatedSVD(n_components=min(32, overview_tf.shape[1]-1), random_state=42).fit_transform(overview_tf)))

# print("  - Keywords [8]...", end=" ")
# kw_texts = [' '.join(eval(kw)[:20]) if not pd.isna(kw) else '' for kw in metadata_df['movie_keywords']]
# kw_tf = TfidfVectorizer(max_features=50).fit_transform(kw_texts)
# features.append(csr_matrix(TruncatedSVD(n_components=min(8, kw_tf.shape[1]-1), random_state=42).fit_transform(kw_tf)))

# print("  - Director [4]...", end=" ")
# dir_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform([str(d) if not pd.isna(d) else '' for d in metadata_df['director']])
# features.append(csr_matrix(TruncatedSVD(n_components=min(4, dir_tf.shape[1]-1), random_state=42).fit_transform(dir_tf)))

# print("  - Cast [4]...", end=" ")
# cast_texts = [' '.join(eval(c)) if not pd.isna(c) else '' for c in metadata_df['top_cast']]
# cast_tf = TfidfVectorizer(analyzer='char', ngram_range=(2,3), max_features=30).fit_transform(cast_texts)
# features.append(csr_matrix(TruncatedSVD(n_components=min(4, cast_tf.shape[1]-1), random_state=42).fit_transform(cast_tf)))

# print("  - Numeric [5]...", end=" ")
# numeric_vecs = []
# for col in ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity']:
#     val_log = np.log1p(metadata_df[col].fillna(0).values)
#     numeric_vecs.append((val_log - val_log.mean()) / (val_log.std() + 1e-8))
# features.append(csr_matrix(np.array(numeric_vecs).T))

# content_features = hstack(features).tocsr()
# content_features = normalize(content_features, norm='l2')
# print(f"\n✓ Content feature matrix: {content_features.shape}")

# # ============================================================================
# # [STEP 3] TRAINING: EASE^R + ADAPTIVE ROUTER
# # ============================================================================

# print("\n[STEP 3] Training 5-Fold Cross-Validation Pipeline...")
# print("="*70)

# from sklearn.model_selection import StratifiedKFold

# # Fixed dimensions based on dataset boundaries
# n_users = raw_ratings_df['user_id'].max()
# n_items = len(metadata_df)

# user_idx = ratings_df['user_id'].values - 1
# item_idx = ratings_df['item_id'].values - 1
# ratings = np.ones(len(ratings_df), dtype=float) # We only care that they interacted positively

# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# folds = list(skf.split(ratings_df, ratings_df['user_id']))

# results = []

# # Normalization function for fair routing
# def min_max_scale(scores_mat):
#     mins = scores_mat.min(axis=1, keepdims=True)
#     maxs = scores_mat.max(axis=1, keepdims=True)
#     ranges = maxs - mins
#     ranges[ranges == 0] = 1.0  # Prevent div by zero
#     return (scores_mat - mins) / ranges

# for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
#     print(f"\n[FOLD {fold_id}] Training...")
    
#     # 1. Build Matrices
#     train_interactions = coo_matrix((ratings[train_idx], (user_idx[train_idx], item_idx[train_idx])), shape=(n_users, n_items)).tocsr()
#     test_interactions = coo_matrix((ratings[test_idx], (user_idx[test_idx], item_idx[test_idx])), shape=(n_users, n_items)).tocsr()
    
#     X = train_interactions.toarray() # Dense is fine for 943x1682
    
#     # ------------------------------------------------------------------------
#     # MODEL A: EASE^R (The Math Upgrade over basic Item-Item CF)
#     # G = X^T X, P = (G + lambda*I)^-1, B = P / -diag(P)
#     # ------------------------------------------------------------------------
#     print("  - Building EASE^R Engine...")
#     G = X.T @ X
#     diag_indices = np.diag_indices_from(G)
#     G[diag_indices] += CONFIG['lambda_ease']
    
#     P = np.linalg.inv(G)
#     B = P / (-np.diag(P)[:, None])
#     np.fill_diagonal(B, 0)
    
#     cf_scores = X @ B # [n_users, n_items]
    
#     # ------------------------------------------------------------------------
#     # MODEL B: Content-Based Profile (Hated Movies Bug Fixed)
#     # Profiles built ONLY from rating >= 4 because we filtered upfront
#     # ------------------------------------------------------------------------
#     print("  - Building Content-Based Engine...")
#     user_profiles = np.zeros((n_users, content_features.shape[1]))
#     for u in range(n_users):
#         rated_items = train_interactions[u].nonzero()[1]
#         if len(rated_items) > 0:
#             prof = content_features[rated_items].mean(axis=0).A1
#             user_profiles[u] = prof / (np.linalg.norm(prof) + 1e-8)
            
#     cb_scores = user_profiles @ content_features.T.toarray() # [n_users, n_items]
    
#     # ------------------------------------------------------------------------
#     # THE ADAPTIVE ROUTER (No more static 40/60 blending)
#     # ------------------------------------------------------------------------
#     print("  - Routing: Warm items -> EASE^R | Cold items -> Content...")
#     # Scale scores per-user [0,1] so routing doesn't favor one engine blindly
#     cf_scores_norm = min_max_scale(cf_scores)
#     cb_scores_norm = min_max_scale(cb_scores)
    
#     # Check popularity in training set
#     item_counts = np.array((train_interactions > 0).sum(axis=0)).flatten()
#     warm_mask = item_counts >= CONFIG['cold_threshold'] # [n_items,]
    
#     # Broadcast routing mask across all users
#     # If item is warm, use normalized EASE^R score. If cold, use normalized Content score.
#     hybrid_scores = np.where(warm_mask, cf_scores_norm, cb_scores_norm)
    
#     # ------------------------------------------------------------------------
#     # EVALUATION
#     # ------------------------------------------------------------------------
#     print(f"  - Evaluating on {test_interactions.nnz:,} Pure Positive (>=4) interactions...")
#     aps, recalls, ndcgs = [], [], []
    
#     for u in range(n_users):
#         test_items_u = test_interactions[u].nonzero()[1]
#         if len(test_items_u) == 0:
#             continue
            
#         scores_u = np.array(hybrid_scores[u]).flatten()
#         rated_items_u = train_interactions[u].nonzero()[1]
#         scores_u[rated_items_u] = -np.inf # Don't recommend what they already watched
        
#         full_ranking = np.argsort(-scores_u)
#         top_10 = full_ranking[:10]
        
#         # Metrics@10
#         hits = np.isin(top_10, test_items_u)
        
#         # MAP@10
#         if hits.sum() > 0:
#             precisions = np.cumsum(hits) / (np.arange(10) + 1)
#             ap = precisions[hits].sum() / min(len(test_items_u), 10)
#         else:
#             ap = 0.0
#         aps.append(ap)
        
#         # Recall@10
#         recalls.append(hits.sum() / len(test_items_u))
        
#         # NDCG@10
#         dcg = np.sum(hits / np.log2(np.arange(2, 12)))
#         idcg = np.sum(1.0 / np.log2(np.arange(2, min(len(test_items_u), 10) + 2)))
#         ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    
#     map_at_10 = np.mean(aps)
#     recall_at_10 = np.mean(recalls)
#     ndcg_at_10 = np.mean(ndcgs)
    
#     print(f"  ✓ MAP@10:    {map_at_10:.4f}")
#     print(f"  ✓ NDCG@10:   {ndcg_at_10:.4f}")
    
#     results.append({'Fold': fold_id, 'MAP@10': map_at_10, 'Recall@10': recall_at_10, 'nDCG@10': ndcg_at_10})

# # ============================================================================
# # FINAL RESULTS
# # ============================================================================

# print("\n" + "="*70)
# print("[STEP 4] FINAL RESULTS (STRICT RATINGS >= 4)")
# print("="*70)

# results_df = pd.DataFrame(results)
# print("\n" + results_df.to_string(index=False))

# print(f"\n[AGGREGATED]")
# print(f"  MAP@10:     {results_df['MAP@10'].mean():.4f} ± {results_df['MAP@10'].std():.4f}")
# print(f"  nDCG@10:    {results_df['nDCG@10'].mean():.4f} ± {results_df['nDCG@10'].std():.4f}")

# CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)
# results_df.to_csv(CONFIG['output_dir'] / 'gold_standard_cv_results.csv', index=False)
# print(f"\n✓ Saved to lightfm_results/gold_standard_cv_results.csv\n")

#!/usr/bin/env python3

#!/usr/bin/env python3
"""
THE REAL GOLD STANDARD PIPELINE: EASE^R + Z-Scaled Content Ensemble
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
    'lambda_ease':2199.9395646690464   ,
    'strict_filter': 4      
}

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
ratings = ratings_df['rating'].values.astype(float) # FULL RATINGS 1-5

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(skf.split(ratings_df, ratings_df['user_id']))

results = []

for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
    print(f"\n[FOLD {fold_id}] Training...")
    
    # Build Train Matrix with FULL DATA
    train_interactions = coo_matrix((ratings[train_idx], (user_idx[train_idx], item_idx[train_idx])), shape=(n_users, n_items)).tocsr()
    
    # ========================================================
    # THE TRUTH FILTER: Extract ONLY >= 4 items for evaluation
    # ========================================================
    test_fold_df = ratings_df.iloc[test_idx]
    test_relevant_df = test_fold_df[test_fold_df['rating'] >= CONFIG['strict_filter']]
    # Map user_id to an array of item_ids they actually LIKED
    test_relevant_dict = test_relevant_df.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
    
    X = train_interactions.toarray() 
    
    # 1. EASE^R Engine
    G = X.T @ X
    diag_indices = np.diag_indices_from(G)
    G[diag_indices] += CONFIG['lambda_ease']
    P = np.linalg.inv(G)
    B = P / (-np.diag(P)[:, None])
    np.fill_diagonal(B, 0)
    cf_scores = X @ B 
    
    # 2. Content Engine (Profile Built ONLY on >= 4 Ratings)
    user_profiles = np.zeros((n_users, content_features.shape[1]))
    for u in range(n_users):
        u_ratings = X[u]
        # Ignore 1,2,3 star movies for content profile
        liked_items = np.where(u_ratings >= CONFIG['strict_filter'])[0]
        if len(liked_items) > 0:
            prof = content_features[liked_items].mean(axis=0).A1
            user_profiles[u] = prof / (np.linalg.norm(prof) + 1e-8)
            
    cb_scores = user_profiles @ content_features.T.toarray() 
    
    # 3. Z-Score Scaling & Ensemble (Mathematically Safe Blend)
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
        rated_items_u = train_interactions[u].nonzero()[1]
        scores_u[rated_items_u] = -np.inf # Exclude what they already watched
        
        full_ranking = np.argsort(-scores_u)
        top_10 = full_ranking[:10]
        
        # --- Metrics @ 10 ---
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

        # --- Full Metrics ---
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
print("[STEP 4] FINAL RESULTS (STRICT RATINGS >= 4)")
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
results_df.to_csv(CONFIG['output_dir'] / 'gold_standard_cv_results.csv', index=False)