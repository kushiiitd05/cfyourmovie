"""
Step 2: LightFM Training (Days 2–3)
====================================

Trains LightFM (WARP loss, k=10) on 5-fold CV.
Evaluates: MAP@10, NDCG@10, Precision@10, Recall@10

Outputs:
  - lightfm_results_5fold.csv (aggregated metrics)
  - fold_metrics_detailed.csv (per-fold breakdown)
"""

import numpy as np
import pandas as pd
import scipy.sparse
import pickle
from sklearn.metrics import ndcg_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Optional: Install lightfm if not present
try:
    import lightfm
    from lightfm import LightFM
except:
    print("Installing lightfm...")
    import subprocess
    subprocess.run(['pip', 'install', 'lightfm'], check=True)
    from lightfm import LightFM

# ============================================================================
# 2.1 LOAD DATA
# ============================================================================

print("Step 2.1: Loading prepared data...")

# Load features
item_features_csr = scipy.sparse.load_npz(
    '/Users/bournesmartasfuck_kush/Desktop/cf/item_features_74d.npz'
)
print(f"  ✓ item_features_csr: {item_features_csr.shape}")

# Load folds
with open('/Users/bournesmartasfuck_kush/Desktop/cf/fold_splits.pkl', 'rb') as f:
    folds = pickle.load(f)
print(f"  ✓ Folds: {len(folds)}")

# ============================================================================
# 2.2 UTILITY FUNCTIONS
# ============================================================================

def build_interactions_csr(interactions_df, n_users=943, n_items=1682):
    """
    Convert interactions dataframe to sparse CSR matrix.
    Shape: (n_users, n_items)
    """
    interactions_csr = scipy.sparse.lil_matrix((n_users, n_items))
    for _, row in interactions_df.iterrows():
        user_idx = int(row['user_id']) - 1  # 0-indexed
        item_idx = int(row['item_id']) - 1  # 0-indexed
        rating = float(row['rating'])
        # Implicit feedback: 1 if rating >= 3.5, else 0
        interactions_csr[user_idx, item_idx] = 1.0 if rating >= 3.5 else 0.0
    return interactions_csr.tocsr()

def mean_average_precision(true_labels):
    """
    Compute Mean Average Precision.
    true_labels: binary array [1, 0, 1, ...] for top-k items
    """
    if not np.any(true_labels):
        return 0.0
    
    tp = np.cumsum(true_labels)
    k = np.arange(1, len(true_labels) + 1)
    precisions = tp / k
    return np.sum(precisions * true_labels) / np.sum(true_labels)

def evaluate_fold(model, test_interactions, item_features_csr, k=10):
    """
    Evaluate LightFM on test set.
    
    Args:
        model: trained LightFM model
        test_interactions: test set dataframe
        item_features_csr: sparse feature matrix
        k: cutoff for ranking metrics
    
    Returns:
        dict with MAP@10, NDCG@10, Precision@10, Recall@10
    """
    # Group test interactions by user
    user_items_true = {}
    for _, row in test_interactions.iterrows():
        user_id = int(row['user_id']) - 1  # 0-indexed
        item_id = int(row['item_id']) - 1  # 0-indexed
        if user_id not in user_items_true:
            user_items_true[user_id] = set()
        user_items_true[user_id].add(item_id)
    
    # Per-user evaluation
    map_scores = []
    ndcg_scores = []
    prec_scores = []
    recall_scores = []
    
    for user_id, true_items in user_items_true.items():
        if len(true_items) == 0:
            continue
        
        # Predict scores for all items
        all_items = np.arange(1682)
        scores = model.predict(user_id, all_items, 
                               item_features=item_features_csr, num_threads=1)
        
        # Get top-k items
        top_k_indices = np.argsort(-scores)[:k]
        top_k_items = set(top_k_indices)
        
        # Compute metrics
        true_labels = np.array([1 if item in true_items else 0 for item in top_k_indices])
        pred_scores = scores[top_k_indices]
        
        # MAP@10
        map_at_k = mean_average_precision(true_labels)
        map_scores.append(map_at_k)
        
        # NDCG@10
        if np.any(true_labels):
            ndcg = ndcg_score([true_labels], [pred_scores])
            ndcg_scores.append(ndcg)
        
        # Precision@10
        prec_at_k = np.sum(true_labels) / k
        prec_scores.append(prec_at_k)
        
        # Recall@10
        recall_at_k = np.sum(true_labels) / len(true_items) if len(true_items) > 0 else 0.0
        recall_scores.append(recall_at_k)
    
    return {
        'MAP@10': np.mean(map_scores) if map_scores else 0.0,
        'NDCG@10': np.mean(ndcg_scores) if ndcg_scores else 0.0,
        'Precision@10': np.mean(prec_scores) if prec_scores else 0.0,
        'Recall@10': np.mean(recall_scores) if recall_scores else 0.0,
        'N_users_tested': len(user_items_true)
    }

# ============================================================================
# 2.3 TRAIN LIGHTFM ON 5-FOLD CV
# ============================================================================

print("\nStep 2.3: Training LightFM on 5-fold CV...")
print("  Architecture: WARP loss, k=10, item_features=74-dim sparse")

fold_results = []

for fold_num in range(5):
    print(f"\n  ┌─ FOLD {fold_num} ─────────────────────────────────")
    
    fold_train, fold_test = folds[fold_num]
    
    # Build CSR matrices
    print(f"    Building interaction matrices...")
    train_csr = build_interactions_csr(fold_train)
    print(f"      train_csr: {train_csr.shape}, nnz={train_csr.nnz}")
    print(f"      test samples: {len(fold_test)}")
    
    # Train LightFM
    print(f"    Training LightFM...")
    model = LightFM(
        loss='warp',
        k=10,
        learning_rate=0.05,
        item_alpha=1e-6,
        user_alpha=1e-6,
        random_state=42
    )
    
    model.fit(
        interactions=train_csr,
        item_features=item_features_csr,
        epochs=15,
        num_threads=4,
        verbose=False
    )
    print(f"      ✓ Trained 15 epochs")
    
    # Evaluate
    print(f"    Evaluating on test set...")
    metrics = evaluate_fold(model, fold_test, item_features_csr, k=10)
    
    fold_results.append({
        'Fold': fold_num,
        'MAP@10': metrics['MAP@10'],
        'NDCG@10': metrics['NDCG@10'],
        'Precision@10': metrics['Precision@10'],
        'Recall@10': metrics['Recall@10'],
        'N_users': metrics['N_users_tested']
    })
    
    print(f"    Results:")
    print(f"      MAP@10       = {metrics['MAP@10']:.4f}")
    print(f"      NDCG@10      = {metrics['NDCG@10']:.4f}")
    print(f"      Precision@10 = {metrics['Precision@10']:.4f}")
    print(f"      Recall@10    = {metrics['Recall@10']:.4f}")
    print(f"    └─ FOLD {fold_num} COMPLETE")

# ============================================================================
# 2.4 AGGREGATE RESULTS
# ============================================================================

print("\n" + "=" * 70)
print("FOLD AGGREGATION")
print("=" * 70)

fold_results_df = pd.DataFrame(fold_results)
fold_results_df.to_csv(
    '/Users/bournesmartasfuck_kush/Desktop/cf/fold_metrics_detailed.csv',
    index=False
)

# Compute mean ± std
metrics_summary = {
    'Model': 'LightFM + φ*(i) [74d]',
    'MAP@10_mean': fold_results_df['MAP@10'].mean(),
    'MAP@10_std': fold_results_df['MAP@10'].std(),
    'NDCG@10_mean': fold_results_df['NDCG@10'].mean(),
    'NDCG@10_std': fold_results_df['NDCG@10'].std(),
    'Precision@10_mean': fold_results_df['Precision@10'].mean(),
    'Precision@10_std': fold_results_df['Precision@10'].std(),
    'Recall@10_mean': fold_results_df['Recall@10'].mean(),
    'Recall@10_std': fold_results_df['Recall@10'].std()
}

summary_df = pd.DataFrame([metrics_summary])
summary_df.to_csv(
    '/Users/bournesmartasfuck_kush/Desktop/cf/lightfm_results_5fold.csv',
    index=False
)

print("\nPer-Fold Results:")
print(fold_results_df.to_string(index=False))

print("\n" + "─" * 70)
print("AGGREGATE RESULTS (mean ± std)")
print("─" * 70)
print(f"MAP@10:        {metrics_summary['MAP@10_mean']:.4f} ± {metrics_summary['MAP@10_std']:.4f}")
print(f"NDCG@10:       {metrics_summary['NDCG@10_mean']:.4f} ± {metrics_summary['NDCG@10_std']:.4f}")
print(f"Precision@10:  {metrics_summary['Precision@10_mean']:.4f} ± {metrics_summary['Precision@10_std']:.4f}")
print(f"Recall@10:     {metrics_summary['Recall@10_mean']:.4f} ± {metrics_summary['Recall@10_std']:.4f}")

print("\n" + "=" * 70)
print("STEP 2 COMPLETE: LightFM training finished")
print("=" * 70)
print(f"\nExpected: MAP@10 ≈ 0.1355 (+20.8% vs. LightGCN baseline 0.1121)")
print(f"Actual:   MAP@10 = {metrics_summary['MAP@10_mean']:.4f}")
print(f"Improvement: {100*(metrics_summary['MAP@10_mean'] - 0.1121) / 0.1121:.1f}%")
