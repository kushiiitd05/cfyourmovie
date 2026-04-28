#!/usr/bin/env python3
"""
Matrix Factorization + Content Features Pipeline
MovieLens 100K + Enriched Metadata (17 features)
Alternative to LightFM using pure scipy/sklearn

Author: Senior Recommender Data Analyst
Date: March 2026
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, hstack, coo_matrix
from scipy.sparse.linalg import svds
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
    'ratings_file': 'ml-100k/u.data',
    'metadata_file': 'Master_final.csv',
    'output_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf/lightfm_results'),
    
    # Matrix factorization
    'k': 10,  # Latent dimensions
    'alpha': 40,  # Regularization
    'iterations': 15,
    
    # Feature blending
    'cf_weight': 0.6,  # Weight for CF component
    'content_weight': 0.4,  # Weight for content component
    
    # CV config
    'n_splits': 5,
    'random_state': 42,
}

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

def load_ratings():
    """Load MovieLens 100K ratings from u.data"""
    print("\n[STEP 1.1] Loading ratings from ml-100k/u.data...")
    
    ratings_path = CONFIG['data_dir'] / CONFIG['ratings_file']
    ratings_df = pd.read_csv(
        ratings_path,
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        dtype={'user_id': int, 'item_id': int, 'rating': int, 'timestamp': int}
    )
    
    print(f"   ✓ Loaded {len(ratings_df):,} ratings")
    print(f"   ✓ Users: {ratings_df['user_id'].nunique()} | Items: {ratings_df['item_id'].nunique()}")
    
    return ratings_df

def load_metadata():
    """Load enriched movie metadata"""
    print("\n[STEP 1.2] Loading enriched metadata from Master_final.csv...")
    
    metadata_path = CONFIG['data_dir'] / CONFIG['metadata_file']
    metadata_df = pd.read_csv(metadata_path)
    
    print(f"   ✓ Loaded {len(metadata_df)} movies with {len(metadata_df.columns)} features")
    
    metadata_df = metadata_df.rename(columns={'movie_id': 'item_id'})
    
    return metadata_df

# ============================================================================
# STEP 2: FEATURE ENGINEERING (FAST VERSION - 74 DIMS)
# ============================================================================

def build_item_features(metadata_df):
    """Build content features quickly [74 dims - minimalist]"""
    print("\n[STEP 2] Building Item Feature Matrix (74 dims)...")
    print("="*70)
    
    feature_matrices = []
    feature_names = []
    
    # 1. GENRES [19]
    print("[2.1] Genres [19]...")
    all_genres = set()
    for genres_str in metadata_df['genres'].dropna():
        try:
            genres = eval(genres_str)
            all_genres.update(genres)
        except:
            pass
    
    all_genres = sorted(list(all_genres))
    genre_features = []
    for genres_str in metadata_df['genres']:
        try:
            genres = set(eval(genres_str))
        except:
            genres = set()
        vec = [1.0 if g in genres else 0.0 for g in all_genres]
        genre_features.append(vec)
    
    genre_matrix = csr_matrix(np.array(genre_features))
    feature_matrices.append(genre_matrix)
    feature_names.extend([f'genre:{g}' for g in all_genres])
    print(f"   ✓ Added {genre_matrix.shape[1]} genre features")
    
    # 2. OVERVIEW [32]
    print("[2.2] Overview (TF-IDF + SVD) [32]...")
    overviews = metadata_df['overview'].fillna('').tolist()
    tfidf = TfidfVectorizer(max_features=100, min_df=2)
    overview_tfidf = tfidf.fit_transform(overviews)
    svd = TruncatedSVD(n_components=min(32, overview_tfidf.shape[1]-1), random_state=42)
    overview_svd = svd.fit_transform(overview_tfidf)
    overview_matrix = csr_matrix(overview_svd)
    feature_matrices.append(overview_matrix)
    feature_names.extend([f'overview_dim_{i}' for i in range(overview_matrix.shape[1])])
    print(f"   ✓ Added {overview_matrix.shape[1]} overview features (explained var: {svd.explained_variance_ratio_.sum():.1%})")
    
    # 3. KEYWORDS [8]
    print("[2.3] Keywords (TF-IDF + SVD) [8]...")
    keywords_texts = []
    for kw_str in metadata_df['movie_keywords']:
        try:
            if pd.isna(kw_str):
                keywords_texts.append('')
            else:
                kw_list = eval(kw_str)
                keywords_texts.append(' '.join(kw_list[:20]))
        except:
            keywords_texts.append('')
    
    tfidf_kw = TfidfVectorizer(max_features=50, min_df=1)
    keywords_tfidf = tfidf_kw.fit_transform(keywords_texts)
    svd_kw = TruncatedSVD(n_components=min(8, keywords_tfidf.shape[1]-1), random_state=42)
    keywords_svd = svd_kw.fit_transform(keywords_tfidf)
    keywords_matrix = csr_matrix(keywords_svd)
    feature_matrices.append(keywords_matrix)
    feature_names.extend([f'keyword_dim_{i}' for i in range(keywords_matrix.shape[1])])
    print(f"   ✓ Added {keywords_matrix.shape[1]} keyword features")
    
    # 4. DIRECTOR [4]
    print("[2.4] Director (character n-grams) [4]...")
    directors = [str(d) if not pd.isna(d) else '' for d in metadata_df['director']]
    tfidf_dir = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=30)
    director_tfidf = tfidf_dir.fit_transform(directors)
    svd_dir = TruncatedSVD(n_components=min(4, director_tfidf.shape[1]-1), random_state=42)
    director_svd = svd_dir.fit_transform(director_tfidf)
    director_matrix = csr_matrix(director_svd)
    feature_matrices.append(director_matrix)
    feature_names.extend([f'director_dim_{i}' for i in range(director_matrix.shape[1])])
    print(f"   ✓ Added {director_matrix.shape[1]} director features")
    
    # 5. CAST [4]
    print("[2.5] Cast (character n-grams) [4]...")
    cast_texts = []
    for cast_str in metadata_df['top_cast']:
        try:
            if pd.isna(cast_str):
                cast_texts.append('')
            else:
                cast_list = eval(cast_str)
                cast_texts.append(' '.join(cast_list))
        except:
            cast_texts.append('')
    
    tfidf_cast = TfidfVectorizer(analyzer='char', ngram_range=(2, 3), max_features=30)
    cast_tfidf = tfidf_cast.fit_transform(cast_texts)
    svd_cast = TruncatedSVD(n_components=min(4, cast_tfidf.shape[1]-1), random_state=42)
    cast_svd = svd_cast.fit_transform(cast_tfidf)
    cast_matrix = csr_matrix(cast_svd)
    feature_matrices.append(cast_matrix)
    feature_names.extend([f'cast_dim_{i}' for i in range(cast_matrix.shape[1])])
    print(f"   ✓ Added {cast_matrix.shape[1]} cast features")
    
    # 6. NUMERICS [7]
    print("[2.6] Numeric features (log-normalized) [7]...")
    numeric_features = []
    numeric_cols = ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity']
    for col in numeric_cols:
        if col in metadata_df.columns:
            val = metadata_df[col].fillna(0).values
            # Log transform for skewed distributions
            val_log = np.log1p(val)
            # Normalize
            val_norm = (val_log - val_log.mean()) / (val_log.std() + 1e-8)
            numeric_features.append(val_norm.reshape(-1, 1))
    
    if numeric_features:
        numeric_matrix = csr_matrix(np.hstack(numeric_features))
        feature_matrices.append(numeric_matrix)
        feature_names.extend([f'{col}_norm' for col in numeric_cols if col in metadata_df.columns])
        print(f"   ✓ Added {numeric_matrix.shape[1]} numeric features")
    
    # COMBINE ALL
    item_features = hstack(feature_matrices).tocsr()
    
    print("\n" + "="*70)
    print(f"[STEP 2 COMPLETE] Item Feature Matrix Built:")
    print(f"   ✓ Shape: {item_features.shape}")
    sparsity = 1 - item_features.nnz / (item_features.shape[0] * item_features.shape[1])
    print(f"   ✓ Sparsity: {sparsity:.2%}")
    print(f"   ✓ Total features: {len(feature_names)}")
    print("="*70)
    
    return item_features, feature_names

# ============================================================================
# STEP 3: MATRIX FACTORIZATION + CONTENT FUSION
# ============================================================================

def train_mf_content_model(interactions, item_features, n_users, n_items, k=10):
    """
    Train model: Blend collaborative filtering + content-based
    
    Score = CF_weight * (user_embedding · item_embedding) + 
            Content_weight * (user_embedding · content_embedding)
    """
    
    # === Collaborative Filtering Component ===
    # SVD decomposition: U, Sigma, Vt
    print("\n   Factorizing interaction matrix...")
    
    # Use randomized SVD for efficiency
    try:
        U, s, Vt = svds(interactions, k=k, random_state=42)
        # Sort in descending order
        idx = np.argsort(-s)
        U = U[:, idx]
        s = s[idx]
        Vt = Vt[idx, :]
    except:
        # Fallback: random initialization
        print("      ⚠️  SVD failed, using random initialization")
        U = np.random.randn(n_users, k) * 0.01
        Vt = np.random.randn(k, n_items) * 0.01
        s = np.ones(k)
    
    # Scale
    user_emb_cf = U  # [n_users, k]
    item_emb_cf = (s[:, np.newaxis] * Vt).T  # [n_items, k]
    
    print(f"      User embeddings CF: {user_emb_cf.shape}")
    print(f"      Item embeddings CF: {item_emb_cf.shape}")
    
    # === Content Component ===
    # SVD on content features
    print("   Factorizing content features...")
    
    if item_features.shape[1] > k:
        try:
            svd_content = TruncatedSVD(n_components=k, random_state=42)
            item_emb_content = svd_content.fit_transform(item_features)
            print(f"      Item embeddings Content: {item_emb_content.shape}")
            print(f"      Content explained variance: {svd_content.explained_variance_ratio_.sum():.1%}")
        except:
            item_emb_content = item_features.toarray()[:, :min(k, item_features.shape[1])]
    else:
        item_emb_content = item_features.toarray()
    
    # === Blend ===
    # Create hybrid embedding by weighted combination
    cf_weight = CONFIG['cf_weight']
    content_weight = CONFIG['content_weight']
    
    item_emb_hybrid = (cf_weight * item_emb_cf + 
                      content_weight * item_emb_content / (np.linalg.norm(item_emb_content, axis=1, keepdims=True) + 1e-8))
    
    return user_emb_cf, item_emb_hybrid, item_emb_cf, item_emb_content

def rank_recommendations(user_emb, item_emb, user_id, interactions, n_recs=10):
    """Generate top-N recommendations for a user"""
    
    # Score = user_emb @ item_emb.T
    scores = user_emb[user_id] @ item_emb.T
    
    # Get already-rated items
    rated_items = interactions[user_id].nonzero()[1]
    
    # Mask rated items
    scores[rated_items] = -np.inf
    
    # Top-N
    top_indices = np.argsort(-scores)[:n_recs]
    top_scores = scores[top_indices]
    
    return top_indices, top_scores

def compute_ap_at_k(recommendations, ground_truth, k=10):
    """Compute Average Precision @ k"""
    if len(ground_truth) == 0:
        return 0.0
    
    score = 0.0
    num_hits = 0
    
    for i, item_id in enumerate(recommendations[:k]):
        if item_id in ground_truth:
            num_hits += 1
            score += num_hits / (i + 1)
    
    return score / min(len(ground_truth), k)

def evaluate_fold(user_emb, item_emb, test_interactions, k=10):
    """Evaluate on test set"""
    
    n_users = user_emb.shape[0]
    aps = []
    recalls = []
    
    for user_id in range(n_users):
        ground_truth = test_interactions[user_id].nonzero()[1]
        
        if len(ground_truth) == 0:
            continue
        
        # Get recommendations
        recs, _ = rank_recommendations(user_emb, item_emb, user_id, test_interactions, n_recs=k)
        
        # Compute AP@k
        ap = compute_ap_at_k(recs, ground_truth, k=k)
        aps.append(ap)
        
        # Compute Recall@k
        recall = len(set(recs) & set(ground_truth)) / len(ground_truth)
        recalls.append(recall)
    
    return np.mean(aps) if aps else 0.0, np.mean(recalls) if recalls else 0.0

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main pipeline execution"""
    print("\n" + "="*70)
    print("HYBRID CF + CONTENT PIPELINE - MovieLens 100K")
    print("="*70)
    
    # Create output directory
    CONFIG['output_dir'].mkdir(exist_ok=True, parents=True)
    
    # STEP 1: LOAD DATA
    print("\n[STEP 1] LOADING DATA")
    print("="*70)
    ratings_df = load_ratings()
    metadata_df = load_metadata()
    
    # STEP 2: FEATURE ENGINEERING
    item_features, feature_names = build_item_features(metadata_df)
    
    # Save feature mapping
    with open(CONFIG['output_dir'] / 'feature_mapping.json', 'w') as f:
        json.dump({i: name for i, name in enumerate(feature_names)}, f, indent=2)
    
    # STEP 3: BUILD INTERACTIONS
    print("\n[STEP 3] BUILDING INTERACTION MATRIX")
    print("="*70)
    
    user_idx = ratings_df['user_id'].values - 1
    item_idx = ratings_df['item_id'].values - 1
    ratings = ratings_df['rating'].values.astype(float)
    
    n_users = 943
    n_items = 1682
    
    interactions = coo_matrix((ratings, (user_idx, item_idx)), 
                             shape=(n_users, n_items)).tocsr()
    
    sparsity = 1 - interactions.nnz / (n_users * n_items)
    print(f"   ✓ Interactions: {interactions.shape} | Sparsity: {sparsity:.2%}")
    
    # STEP 4: STRATIFIED FOLDS
    print("\n[STEP 4] CREATING 5-FOLD STRATIFIED SPLIT")
    print("="*70)
    
    skf = StratifiedKFold(n_splits=CONFIG['n_splits'], shuffle=True, 
                         random_state=CONFIG['random_state'])
    folds = list(skf.split(ratings_df, ratings_df['user_id']))
    
    print(f"   ✓ Created {len(folds)} folds")
    
    # STEP 5: TRAIN & EVALUATE
    print("\n[STEP 5] TRAINING & EVALUATING (5-FOLD CV)")
    print("="*70)
    
    fold_results = []
    
    for fold_id, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\n[FOLD {fold_id}]")
        print(f"   Train: {len(train_idx):,} | Test: {len(test_idx):,}")
        
        # Build fold-specific matrices
        train_data = ratings[train_idx]
        train_rows = user_idx[train_idx]
        train_cols = item_idx[train_idx]
        
        test_data = ratings[test_idx]
        test_rows = user_idx[test_idx]
        test_cols = item_idx[test_idx]
        
        train_interactions = coo_matrix((train_data, (train_rows, train_cols)),
                                       shape=(n_users, n_items)).tocsr()
        test_interactions = coo_matrix((test_data, (test_rows, test_cols)),
                                      shape=(n_users, n_items)).tocsr()
        
        # Train
        print(f"   Training hybrid model...")
        user_emb, item_emb, item_emb_cf, item_emb_content = train_mf_content_model(
            train_interactions, item_features, n_users, n_items, k=CONFIG['k']
        )
        
        # Evaluate
        print(f"   Evaluating...")
        map_at_10, recall_at_10 = evaluate_fold(user_emb, item_emb, test_interactions, k=10)
        
        print(f"   ✓ MAP@10: {map_at_10:.4f}")
        print(f"   ✓ Recall@10: {recall_at_10:.4f}")
        
        fold_results.append({
            'fold_id': fold_id,
            'map_at_10': map_at_10,
            'recall_at_10': recall_at_10,
            'user_emb': user_emb,
            'item_emb': item_emb,
        })
    
    # STEP 6: AGGREGATE
    print("\n[STEP 6] AGGREGATING RESULTS")
    print("="*70)
    
    maps = [r['map_at_10'] for r in fold_results]
    recalls = [r['recall_at_10'] for r in fold_results]
    
    print(f"\n[FINAL RESULTS]")
    print(f"   MAP@10:    {np.mean(maps):.4f} ± {np.std(maps):.4f}")
    print(f"   Recall@10: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
    
    # Save results
    results_df = pd.DataFrame([
        {'Fold': r['fold_id'], 'MAP@10': r['map_at_10'], 'Recall@10': r['recall_at_10']}
        for r in fold_results
    ])
    
    results_df.to_csv(CONFIG['output_dir'] / 'hybrid_cv_results.csv', index=False)
    print(f"\n   ✓ Saved results to hybrid_cv_results.csv")
    
    # Save best model
    best_fold = max(fold_results, key=lambda x: x['map_at_10'])
    with open(CONFIG['output_dir'] / 'best_model.pkl', 'wb') as f:
        pickle.dump({
            'user_emb': best_fold['user_emb'],
            'item_emb': best_fold['item_emb'],
        }, f)
    
    print(f"   ✓ Saved best model (Fold {best_fold['fold_id']}) to best_model.pkl")
    
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
