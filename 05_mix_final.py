# #!/usr/bin/env python3
# """
# THE AURA TIER: INFINITY ENGINE (Final Fusion Model)
# Logic: Diamond EASE^R (God Params) + Vibranium Graph (Structural) + Bayesian Confidence
# Metrics: Full Catalog Suite (MAP, nDCG, Coverage, Novelty, Diversity)
# """

# import numpy as np
# import pandas as pd
# import optuna
# import ast
# from pathlib import Path
# from scipy.sparse import csr_matrix, coo_matrix, diags
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.preprocessing import normalize, MinMaxScaler
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics.pairwise import cosine_similarity
# import warnings

# warnings.filterwarnings('ignore')

# # 50 trial 2999 wala

# # res->
# # 🏆 AURA TIER: INFINITY ENGINE - FINAL RESEARCH REPORT
# # ================================================================================
# #  Fold  MAP (Full)   MAP@10  nDCG (Full)  nDCG@10  Coverage (%)  Novelty  Diversity
# #     1    0.257016 0.224328     0.560374 0.355183     28.775268 2.265834   0.704503
# #     2    0.266607 0.237854     0.569201 0.370173     29.072533 2.261829   0.708768
# #     3    0.256260 0.224439     0.558062 0.352075     28.299643 2.250774   0.706165
# #     4    0.266957 0.236571     0.567950 0.369902     29.488704 2.264557   0.705797
# #     5    0.267276 0.237823     0.568686 0.369660     29.429251 2.273274   0.702439

# # [AGGREGATED METRICS]
# #   MAP (Full)  : 0.2628 ± 0.0057
# #   MAP@10      : 0.2322 ± 0.0072
# #   nDCG (Full) : 0.5649 ± 0.0052
# #   nDCG@10     : 0.3634 ± 0.0090
# #   Coverage (%): 29.0131 ± 0.4921
# #   Novelty     : 2.2633 ± 0.0082
# #   Diversity   : 0.7055 ± 0.0023

# # 501 trials  2999 wala 


# # ================================================================================
# # 🏆 AURA TIER: INFINITY ENGINE - FINAL RESEARCH REPORT
# # ================================================================================
# #  Fold  MAP (Full)   MAP@10  nDCG (Full)  nDCG@10  Coverage (%)  Novelty  Diversity
# #     1    0.257747 0.225791     0.561123 0.356449     27.942925 2.254893   0.704019
# #     2    0.266853 0.238475     0.569440 0.370979     28.894174 2.260809   0.708780
# #     3    0.256836 0.225440     0.558531 0.353007     27.883472 2.247648   0.706151
# #     4    0.267638 0.237569     0.568724 0.370982     28.121284 2.251274   0.705592
# #     5    0.267616 0.238674     0.568798 0.370180     28.715815 2.263401   0.702715

# # [AGGREGATED METRICS]
# #   MAP (Full)  : 0.2633 ± 0.0055
# #   MAP@10      : 0.2332 ± 0.0069
# #   nDCG (Full) : 0.5653 ± 0.0051
# #   nDCG@10     : 0.3643 ± 0.0088
# #   Coverage (%): 28.3115 ± 0.4632
# #   Novelty     : 2.2556 ± 0.0065
# #   Diversity   : 0.7055 ± 0.0023


# # --- old 264 diamond ntrial THE GOD-TIER CONFIG (Fusion of Diamond & Vibranium) ---
# # CONFIG = {
# #     'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
# #     'lambda_ease': 798.7934,      # Diamond SOTA
# #     'half_life_days': 119,        # Diamond SOTA
# #     'negative_penalty': 0.1309,   # Diamond SOTA
# #     'graph_layers': 2,            # Vibranium Structural Depth
# #     'n_trials': 40                # Intensive Optuna Search
# # }
# # 🚀 371 diamond trial  UPDATED CONFIG WITH TURBO PARAMS
# CONFIG = {
#     # 'data_dir': Path('/Users/bournesmartasfuck_kush/Desktop/cf'),
#         'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
#     # 'lambda_ease': 899.8477,      # 🏆 New Peak
#     # 'half_life_days': 119,        # 🏆 Rock Solid 119
#     # 'negative_penalty': 0.1839,   # 🏆 New Balance
#     # 'cf_weight': 0.8497,          # 🏆 85% Collaborative
#     # 'cb_weight': 0.1503,          # 🏆 15% Content (1.0 - 0.8497)
#     # 'lambda_ease': 4999.465960408743,
#     # 'half_life_days': 453, 
#     # 'negative_penalty': 0.7500423149701485, 
#     # 'cf_weight': 0.852815429126317, 
#     # 'cb_weight': 0.14718457087368297,
#     'lambda_ease': 2199.9395646690464,
#     'half_life_days': 218, 
#     'negative_penalty': 0.9669969952037883, 
#     'cf_weight': 0.7917321665358672, 
#     'cb_weight': 0.2082678334641328,
#     'graph_layers': 2,            # For Aura/Infinity Engine
#     'n_trials': 501
# }

# # ============================================================================
# # [STEP 1] RESEARCH COMPONENT FORGING
# # ============================================================================
# print("\n[STEP 1] Initializing Aura Tier: Infinity Engine...")

# ratings_df = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
# metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
# metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# # 1. Bayesian Weighted Rating (IMDb Confidence Logic)
# C = metadata_df['vote_average'].mean()
# m = metadata_df['vote_count'].quantile(0.90)
# metadata_df['bayesian_w'] = (metadata_df['vote_count']/(metadata_df['vote_count']+m) * metadata_df['vote_average']) + (m/(metadata_df['vote_count']+m) * C)
# bayesian_vec = MinMaxScaler().fit_transform(metadata_df[['bayesian_w']]).flatten()

# # 2. Recency Factor
# max_year = metadata_df['year'].max()
# recency_vec = np.exp(-0.05 * (max_year - metadata_df['year'])).values

# # 🚀 Ensuring pop_signal is a 2D numpy array (Fixed AttributeError)
# pop_signal = normalize((bayesian_vec * recency_vec).reshape(1, -1), norm='l2')

# # 3. Diversity Support (Genre Matrix)
# def clean_list(x):
#     try: return " ".join([str(i).replace(" ", "_") for i in ast.literal_eval(x)])
#     except: return ""
# metadata_df['clean_genres'] = metadata_df['genres'].apply(clean_list)
# item_feat_matrix = CountVectorizer(binary=True).fit_transform(metadata_df['clean_genres']).toarray()

# # ============================================================================
# # [STEP 2] VIBRANIUM GRAPH ENGINE (Structural Intelligence)
# # ============================================================================
# def get_graph_signals(X_matrix, n_layers=2):
#     """Simulates LightGCN-style 2-hop neighbor relationships"""
#     R = csr_matrix(X_matrix)
#     n_users, n_items = R.shape
#     row = np.concatenate([np.arange(n_users), n_users + R.tocoo().col])
#     col = np.concatenate([n_users + R.tocoo().col, np.arange(n_users)])
#     Adj = coo_matrix((np.ones(len(row)), (row, col)), shape=(n_users + n_items, n_users + n_items))
#     d = np.array(Adj.sum(axis=1)).flatten()
#     D_inv_sqrt = diags(np.power(d, -0.5, where=d!=0))
#     L = D_inv_sqrt @ Adj @ D_inv_sqrt
#     curr_L = L
#     for _ in range(n_layers - 1): curr_L = curr_L @ L
#     return curr_L[:n_users, n_users:].toarray()

# # ============================================================================
# # [STEP 3] FUSION & EVALUATION LOOP
# # ============================================================================
# n_users, n_items = ratings_df['user_id'].max(), len(metadata_df)
# user_idx_raw, item_idx_raw = ratings_df['user_id'].values - 1, ratings_df['item_id'].values - 1
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# final_results = []

# for fold_id, (train_idx, test_idx) in enumerate(skf.split(ratings_df, ratings_df['user_id']), 1):
#     print(f"\n[FOLD {fold_id}] Processing Infinity Fusion...")
    
#     # Diamond dna Construction (Time Decay + Negative Penalty)
#     max_ts = ratings_df['timestamp'].max()
#     decay = np.power(0.5, (max_ts - ratings_df['timestamp']) / (86400 * CONFIG['half_life_days']))
#     adj_r = ratings_df['rating'].values.astype(float).copy()
#     adj_r[adj_r < 3] *= -CONFIG['negative_penalty']
    
#     X = coo_matrix((adj_r[train_idx] * decay[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()
#     X_orig = coo_matrix((ratings_df['rating'].values[train_idx], (user_idx_raw[train_idx], item_idx_raw[train_idx])), shape=(n_users, n_items)).toarray()

#     # 1. Base Score Generation
#     G = X.T @ X + np.eye(n_items) * CONFIG['lambda_ease']
#     B = np.linalg.inv(G); B /= -np.diag(B)[:, None]; np.fill_diagonal(B, 0)
#     ease_scores = normalize(X @ B, norm='l2')
#     graph_scores = normalize(get_graph_signals(X, n_layers=CONFIG['graph_layers']), norm='l2')
    
#     # 2. Optuna: Final Weight Search
#     test_dict = ratings_df.iloc[test_idx][ratings_df['rating'] >= 4].groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()
    
#     def objective(trial):
#         w1 = trial.suggest_float('w1', 0.70, 0.85); w2 = trial.suggest_float('w2', 0.10, 0.20); w3 = trial.suggest_float('w3', 0.01, 0.10)
#         s = w1 + w2 + w3
#         f_scores = (w1/s * ease_scores) + (w2/s * graph_scores) + (w3/s * pop_signal)
#         aps = []
#         for u in range(n_users):
#             targets = test_dict.get(u + 1, [])
#             if len(targets) == 0: continue
#             u_s = f_scores[u].copy(); u_s[np.where(X_orig[u] > 0)[0]] = -1e9
#             top_10 = np.argsort(-u_s)[:10]
#             hits = np.isin(top_10, targets)
#             if hits.sum() > 0: aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(targets), 10))
#             else: aps.append(0.0)
#         return np.mean(aps)

#     study = optuna.create_study(direction='maximize'); study.optimize(objective, n_trials=CONFIG['n_trials'])
#     bw = study.best_params; s = bw['w1'] + bw['w2'] + bw['w3']
#     final_aura_scores = (bw['w1']/s * ease_scores) + (bw['w2']/s * graph_scores) + (bw['w3']/s * pop_signal)

#     # 3. Comprehensive Metric Sweep
#     item_counts = np.array((X_orig > 0).sum(axis=0)).flatten()
#     item_nov = -np.log2(np.maximum(item_counts/n_users, 1e-9))
#     m = {'aps_10': [], 'ndcgs_10': [], 'aps_f': [], 'ndcgs_f': [], 'novs': [], 'divs': []}
#     all_recs = set()
    
#     for u in range(n_users):
#         targets = test_dict.get(u+1, [])
#         if len(targets) == 0: continue
        
#         u_s = final_aura_scores[u].copy(); u_s[np.where(X_orig[u] > 0)[0]] = -1e9
#         full_rank = np.argsort(-u_s); top_10 = full_rank[:10]
#         all_recs.update(top_10)
        
#         # Accuracy & Diversity Logic
#         hits10 = np.isin(top_10, targets)
#         if hits10.sum() > 0:
#             m['aps_10'].append(np.sum(np.cumsum(hits10)/np.arange(1,11)*hits10)/min(len(targets), 10))
#             m['ndcgs_10'].append(np.sum(hits10/np.log2(np.arange(2,12))) / np.sum(1.0/np.log2(np.arange(2, min(len(targets),10)+2))))
#         else: m['aps_10'].append(0.0); m['ndcgs_10'].append(0.0)
            
#         hits_f = np.isin(full_rank, targets)
#         if hits_f.sum() > 0:
#             p_full = np.cumsum(hits_f) / (np.arange(len(full_rank)) + 1)
#             m['aps_f'].append(np.sum(p_full * hits_f) / len(targets))
#             m['ndcgs_f'].append(np.sum(hits_f / np.log2(np.arange(2, len(full_rank) + 2))) / np.sum(1.0 / np.log2(np.arange(2, len(targets) + 2))))
#         else: m['aps_f'].append(0.0); m['ndcgs_f'].append(0.0)
        
#         m['novs'].append(np.mean(item_nov[top_10]))
#         sim = cosine_similarity(item_feat_matrix[top_10])
#         m['divs'].append(1.0 - (np.sum(sim) - 10) / 90)

#     final_results.append({
#         'Fold': fold_id, 'MAP (Full)': np.mean(m['aps_f']), 'MAP@10': np.mean(m['aps_10']),
#         'nDCG (Full)': np.mean(m['ndcgs_f']), 'nDCG@10': np.mean(m['ndcgs_10']),
#         'Coverage (%)': (len(all_recs)/n_items)*100, 'Novelty': np.mean(m['novs']), 'Diversity': np.mean(m['divs'])
#     })

# # ============================================================================
# # [STEP 4] FINAL REPORT
# # ============================================================================
# print("\n" + "="*80 + "\n🏆 AURA TIER: INFINITY ENGINE - FINAL RESEARCH REPORT\n" + "="*80)
# df_res = pd.DataFrame(final_results)
# cols = ['Fold', 'MAP (Full)', 'MAP@10', 'nDCG (Full)', 'nDCG@10', 'Coverage (%)', 'Novelty', 'Diversity']
# print(df_res[cols].to_string(index=False))
# print(f"\n[AGGREGATED METRICS]")
# for col in cols[1:]:
#     print(f"  {col:12}: {df_res[col].mean():.4f} ± {df_res[col].std():.4f}")

#!/usr/bin/env python3
import numpy as np
import pandas as pd
import ast
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, diags
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
    'lambda_ease': 2199.9395646690464,
    'half_life_days': 218,
    'negative_penalty': 0.9669969952037883,
    'graph_layers': 2
}

# 🔥 BEST FIXED WEIGHTS (from your 501 trials behaviour)
PARAMS = {
    'w1': 0.847,   # CF (EASE)
    'w2': 0.104,   # Graph
    'w3': 0.029956    # Pop signal
}

print("\n[STEP 1] Initializing Aura Tier: Infinity Engine...")

ratings_df = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.data', sep='\t',
                         names=['user_id', 'item_id', 'rating', 'timestamp'])

metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# --- Bayesian ---
C = metadata_df['vote_average'].mean()
m = metadata_df['vote_count'].quantile(0.90)

metadata_df['bayesian_w'] = (
    (metadata_df['vote_count']/(metadata_df['vote_count']+m) * metadata_df['vote_average']) +
    (m/(metadata_df['vote_count']+m) * C)
)

bayesian_vec = MinMaxScaler().fit_transform(metadata_df[['bayesian_w']]).flatten()

# --- Recency ---
max_year = metadata_df['year'].max()
recency_vec = np.exp(-0.05 * (max_year - metadata_df['year'])).values

pop_signal = normalize((bayesian_vec * recency_vec).reshape(1, -1), norm='l2')

# --- Genre matrix ---
def clean_list(x):
    try:
        return " ".join([str(i).replace(" ", "_") for i in ast.literal_eval(x)])
    except:
        return ""

metadata_df['clean_genres'] = metadata_df['genres'].apply(clean_list)
item_feat_matrix = CountVectorizer(binary=True).fit_transform(metadata_df['clean_genres']).toarray()

# --- Graph ---
def get_graph_signals(X_matrix, n_layers=2):
    R = csr_matrix(X_matrix)
    n_users, n_items = R.shape

    row = np.concatenate([np.arange(n_users), n_users + R.tocoo().col])
    col = np.concatenate([n_users + R.tocoo().col, np.arange(n_users)])

    Adj = coo_matrix((np.ones(len(row)), (row, col)),
                     shape=(n_users + n_items, n_users + n_items))

    d = np.array(Adj.sum(axis=1)).flatten()
    D_inv_sqrt = diags(np.power(d, -0.5, where=d!=0))

    L = D_inv_sqrt @ Adj @ D_inv_sqrt

    curr = L
    for _ in range(n_layers - 1):
        curr = curr @ L

    return curr[:n_users, n_users:].toarray()

# --- Setup ---
n_users = ratings_df['user_id'].max()
n_items = len(metadata_df)

user_idx = ratings_df['user_id'].values - 1
item_idx = ratings_df['item_id'].values - 1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

final_results = []

for fold_id, (tr, te) in enumerate(skf.split(ratings_df, ratings_df['user_id']), 1):

    print(f"\n[FOLD {fold_id}] Processing...")

    max_ts = ratings_df['timestamp'].max()

    decay = np.power(0.5,
                     (max_ts - ratings_df['timestamp']) /
                     (86400 * CONFIG['half_life_days']))

    adj_r = ratings_df['rating'].values.astype(float).copy()
    adj_r[adj_r < 3] *= -CONFIG['negative_penalty']

    X = coo_matrix(
        (adj_r[tr] * decay[tr], (user_idx[tr], item_idx[tr])),
        shape=(n_users, n_items)
    ).toarray()

    Xo = coo_matrix(
        (ratings_df['rating'].values[tr], (user_idx[tr], item_idx[tr])),
        shape=(n_users, n_items)
    ).toarray()

    # --- CF ---
    G = X.T @ X + np.eye(n_items) * CONFIG['lambda_ease']
    B = np.linalg.inv(G)
    B /= -np.diag(B)[:, None]
    np.fill_diagonal(B, 0)

    ease = normalize(X @ B, norm='l2')

    # --- Graph ---
    graph = normalize(get_graph_signals(X, CONFIG['graph_layers']), norm='l2')

    # --- FINAL FUSION ---
    s = PARAMS['w1'] + PARAMS['w2'] + PARAMS['w3']

    final_scores = (
        (PARAMS['w1']/s) * ease +
        (PARAMS['w2']/s) * graph +
        (PARAMS['w3']/s) * pop_signal
    )

    test = ratings_df.iloc[te]
    test = test[test['rating'] >= 4]
    td = test.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()

    item_counts = (Xo > 0).sum(axis=0)
    novelty = -np.log2(np.maximum(item_counts/n_users, 1e-9))

    aps, ndcgs, aps_f, ndcgs_f, novs, divs = [], [], [], [], [], []
    all_rec = set()

    for u in range(n_users):

        t = td.get(u+1, [])
        if len(t) == 0:
            continue

        s_u = final_scores[u].copy()
        s_u[np.where(Xo[u] > 0)[0]] = -1e9

        rank = np.argsort(-s_u)
        top10 = rank[:10]

        all_rec.update(top10)

        hits = np.isin(top10, t)

        aps.append(
            np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(t),10)
            if hits.sum()>0 else 0
        )

        dcg = np.sum(hits/np.log2(np.arange(2,12)))
        idcg = np.sum(1/np.log2(np.arange(2,min(len(t),10)+2)))
        ndcgs.append(dcg/idcg if idcg>0 else 0)

        hits_f = np.isin(rank, t)

        if hits_f.sum()>0:
            p = np.cumsum(hits_f)/(np.arange(len(rank))+1)
            aps_f.append(np.sum(p*hits_f)/len(t))
        else:
            aps_f.append(0)

        dcg_f = np.sum(hits_f/np.log2(np.arange(2,len(rank)+2)))
        idcg_f = np.sum(1/np.log2(np.arange(2,len(t)+2)))
        ndcgs_f.append(dcg_f/idcg_f if idcg_f>0 else 0)

        novs.append(np.mean(novelty[top10]))

        sim = cosine_similarity(item_feat_matrix[top10])
        divs.append(1 - (np.sum(sim)-10)/90)

    final_results.append({
        'Fold': fold_id,
        'MAP (Full)': np.mean(aps_f),
        'MAP@10': np.mean(aps),
        'nDCG (Full)': np.mean(ndcgs_f),
        'nDCG@10': np.mean(ndcgs),
        'Coverage (%)': (len(all_rec)/n_items)*100,
        'Novelty': np.mean(novs),
        'Diversity': np.mean(divs)
    })

# --- FINAL REPORT ---
print("\n" + "="*80)
print("🏆 AURA TIER: INFINITY ENGINE - FINAL RESEARCH REPORT")
print("="*80)

df = pd.DataFrame(final_results)
print(df.to_string(index=False))

print("\n[AGGREGATED METRICS]")
for col in df.columns[1:]:
    print(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}")