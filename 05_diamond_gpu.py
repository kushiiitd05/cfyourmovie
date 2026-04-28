# #!/usr/bin/env python3
# """
# GPU-ACCELERATED RECOMMENDER (A100 READY 🚀)
# - Uses CUDA if available
# - Falls back to CPU
# - Accelerates EASE core using PyTorch
# """

# import numpy as np
# import pandas as pd
# import torch
# from pathlib import Path
# from scipy.sparse import coo_matrix, csr_matrix, hstack
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.decomposition import TruncatedSVD
# from sklearn.preprocessing import normalize
# from sklearn.model_selection import StratifiedKFold
# import optuna
# import warnings

# warnings.filterwarnings("ignore")

# # ===========================
# # 🔥 DEVICE SETUP
# # ===========================
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"\n🚀 Using Device: {DEVICE}")

# def to_torch(x):
#     return torch.tensor(x, dtype=torch.float32, device=DEVICE)

# def to_numpy(x):
#     return x.detach().cpu().numpy()

# # ===========================
# # CONFIG
# # ===========================
# CONFIG = {
#     'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
#     'strict_filter': 4,
#     'n_optuna_trials': 50  # reduce first, then increase
# }

# # ===========================
# # LOAD DATA
# # ===========================
# print("\n[STEP 1] Loading Data...")

# raw_ratings_df = pd.read_csv(
#     CONFIG['data_dir'] / 'ml-100k/u.data',
#     sep='\t',
#     names=['user_id', 'item_id', 'rating', 'timestamp']
# )

# metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
# metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# n_users = raw_ratings_df['user_id'].max()
# n_items = len(metadata_df)

# user_idx = raw_ratings_df['user_id'].values - 1
# item_idx = raw_ratings_df['item_id'].values - 1

# # ===========================
# # FEATURE ENGINEERING
# # ===========================
# print("[STEP 2] Building Features...")

# features = []

# # Genres
# all_genres = set()
# for g in metadata_df['genres'].dropna():
#     try: all_genres.update(eval(g))
#     except: pass

# genre_list = sorted(list(all_genres))
# genre_vec = [[1.0 if g in (eval(x) if pd.notna(x) else []) else 0.0 for g in genre_list] for x in metadata_df['genres']]
# features.append(csr_matrix(np.array(genre_vec)))

# # Overview
# tf = TfidfVectorizer(max_features=100)
# ov = tf.fit_transform(metadata_df['overview'].fillna(''))
# features.append(csr_matrix(TruncatedSVD(n_components=32).fit_transform(ov)))

# content_features = normalize(hstack(features)).tocsr()

# # ===========================
# # 🔥 GPU EASE FUNCTION
# # ===========================
# def compute_ease(X, lambda_):
#     X_t = to_torch(X)

#     G = X_t.T @ X_t
#     G += torch.eye(G.shape[0], device=DEVICE) * lambda_

#     P = torch.linalg.inv(G)

#     diag = torch.diag(P)
#     B = P / (-diag.unsqueeze(1))
#     B.fill_diagonal_(0)

#     scores = X_t @ B
#     return to_numpy(scores)

# # ===========================
# # OBJECTIVE
# # ===========================
# def objective(trial):
#     lambda_ = trial.suggest_float('lambda', 500, 900)

#     skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

#     scores_list = []

#     for i, (train_idx, test_idx) in enumerate(skf.split(raw_ratings_df, raw_ratings_df['user_id'])):
#         if i > 1: break

#         X = coo_matrix(
#             (raw_ratings_df['rating'].values[train_idx],
#              (user_idx[train_idx], item_idx[train_idx])),
#             shape=(n_users, n_items)
#         ).toarray()

#         ease_scores = compute_ease(X, lambda_)

#         test_df = raw_ratings_df.iloc[test_idx]
#         test_dict = test_df.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()

#         aps = []

#         for u in range(n_users):
#             targets = test_dict.get(u + 1, [])
#             if len(targets) == 0:
#                 continue

#             scores = ease_scores[u].copy()
#             scores[np.where(X[u] > 0)[0]] = -1e9

#             top10 = np.argsort(-scores)[:10]
#             hits = np.isin(top10, targets)

#             if hits.sum() > 0:
#                 aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(targets),10))
#             else:
#                 aps.append(0.0)

#         scores_list.append(np.mean(aps))

#     return np.mean(scores_list)

# # ===========================
# # RUN OPTUNA
# # ===========================
# print("\n[STEP 3] Running Optuna...")

# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=CONFIG['n_optuna_trials'])

# print("\n🏆 BEST PARAMS:")
# print(study.best_params)

# oll 50 trial


# 🔥 NEW CONFIG 1 oll - 50 oputuna : {'lambda_ease': 999.7711623449915, 'half_life_days': 134, 'negative_penalty': 0.1984458183484561, 'cf_weight': 0.8797279231086061, 'cb_weight': 0.12027207689139385}
# joit 200 oputuna oll config ->  🔥 BEST CONFIG: {'lambda_ease': 1099.9397418907427, 'half_life_days': 129, 'negative_penalty': 0.8615283172455033, 'cf_weight': 0.8010379128658882, 'cb_weight': 0.1989620871341118}
#  joint coint 200 var ranges .1 # 🔥 BEST CONFIG: {'lambda_ease': 1449.8978513478594, 'half_life_days': 150, 'negative_penalty': 0.8376761947937954, 'cf_weight': 0.8099609108112742, 'cb_weight': 0.1900390891887258}

# 🔥 FINAL COMPARISON

# OLD: 0.2776 ± 0.0040
# NEW: 0.2885 ± 0.0044


#  joint config inc trial .2 #

# 🔥 BEST CONFIG: {'lambda_ease': 2199.9395646690464, 'half_life_days': 218, 'negative_penalty': 0.9669969952037883, 'cf_weight': 0.7917321665358672, 'cb_weight': 0.2082678334641328}

# 🔥 FINAL COMPARISON

# OLD: 0.2776 ± 0.0040
# NEW: 0.2954 ± 0.0040


#  joint .3 trial 
# 🔥 BEST CONFIG: {'lambda_ease': 4999.465960408743, 'half_life_days': 453, 'negative_penalty': 0.7500423149701485, 'cf_weight': 0.852815429126317, 'cb_weight': 0.14718457087368297}

# 🔥 FINAL COMPARISON

# OLD: 0.2776 ± 0.0040
# NEW: 0.3013 ± 0.0035


#!/usr/bin/env python3
"""
🔥 FINAL JOINT OPTUNA (CORRECT VERSION)
- ALL params tuned together
- GPU safe
- 3-fold (fast) during search
- 5-fold final evaluation
"""

import numpy as np
import pandas as pd
import torch
import optuna
from pathlib import Path
from scipy.sparse import coo_matrix
from sklearn.model_selection import StratifiedKFold

# ===========================
# DEVICE
# ===========================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🚀 Device: {DEVICE}")

# ===========================
# BASE CONFIG (OLD)
# ===========================
BASE = {
    'lambda_ease': 899.8477,
    'half_life_days': 119,
    'negative_penalty': 0.1839,
    'cf_weight': 0.8497
}
BASE['cb_weight'] = 1 - BASE['cf_weight']

# ===========================
# LOAD DATA
# ===========================
DATA_DIR = Path('/home/vinayak23597/Kush/cf_final_project')

df = pd.read_csv(DATA_DIR / 'ml-100k/u.data', sep='\t',
                 names=['user_id','item_id','rating','timestamp'])

n_users = df['user_id'].max()
n_items = df['item_id'].max()

user_idx = df['user_id'].values - 1
item_idx = df['item_id'].values - 1
ratings = df['rating'].values
timestamps = df['timestamp'].values

# ===========================
# GPU EASE
# ===========================
def ease_gpu(X, lambda_):
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)

    G = X_t.T @ X_t + torch.eye(X.shape[1], device=DEVICE) * lambda_
    P = torch.linalg.inv(G)

    diag = torch.diag(P)
    B = P / (-diag.unsqueeze(1))
    B.fill_diagonal_(0)

    out = (X_t @ B).cpu().numpy()

    del X_t, G, P, B
    torch.cuda.empty_cache()

    return out

# ===========================
# FAST PIPELINE (3-FOLD)
# ===========================
def evaluate(config):

    max_ts = timestamps.max()
    decay = np.power(0.5, (max_ts - timestamps) / (86400 * config['half_life_days']))
    weighted = ratings * decay

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in skf.split(df, df['user_id']):

        X = coo_matrix((weighted[train_idx],
                       (user_idx[train_idx], item_idx[train_idx])),
                       shape=(n_users, n_items)).toarray()

        pred = ease_gpu(X, config['lambda_ease'])

        test_df = df.iloc[test_idx]
        test_dict = test_df.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()

        aps = []

        for u in range(n_users):
            targets = test_dict.get(u+1, [])
            if len(targets)==0: continue

            s = pred[u].copy()
            s[X[u] > 0] = -1e9

            top10 = np.argsort(-s)[:10]
            hits = np.isin(top10, targets)

            if hits.sum()>0:
                aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(targets),10))
            else:
                aps.append(0)

        scores.append(np.mean(aps))

    return np.mean(scores)

# ===========================
# FULL EVAL (5-FOLD)
# ===========================
def full_eval(config):

    max_ts = timestamps.max()
    decay = np.power(0.5, (max_ts - timestamps) / (86400 * config['half_life_days']))
    weighted = ratings * decay

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in skf.split(df, df['user_id']):

        X = coo_matrix((weighted[train_idx],
                       (user_idx[train_idx], item_idx[train_idx])),
                       shape=(n_users, n_items)).toarray()

        pred = ease_gpu(X, config['lambda_ease'])

        test_df = df.iloc[test_idx]
        test_dict = test_df.groupby('user_id')['item_id'].apply(lambda x: x.values - 1).to_dict()

        aps = []

        for u in range(n_users):
            targets = test_dict.get(u+1, [])
            if len(targets)==0: continue

            s = pred[u].copy()
            s[X[u] > 0] = -1e9

            top10 = np.argsort(-s)[:10]
            hits = np.isin(top10, targets)

            if hits.sum()>0:
                aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(targets),10))
            else:
                aps.append(0)

        scores.append(np.mean(aps))

    return np.mean(scores), np.std(scores)

# ===========================
# JOINT OPTUNA
# ===========================
def objective(trial):

    config = {
        'lambda_ease': trial.suggest_float("lambda_ease", 2000, 5000),
        'half_life_days': trial.suggest_int("half_life_days", 210, 500),
        'negative_penalty': trial.suggest_float("negative_penalty", 0.75, 1),
        'cf_weight': trial.suggest_float("cf_weight", 0.75, 0.9)
    }
    config['cb_weight'] = 1 - config['cf_weight']

    return evaluate(config)

# ===========================
# RUN OPTUNA
# ===========================
print("\n🔥 RUNNING JOINT OPTUNA")

study = optuna.create_study(direction="maximize")

study.optimize(
    objective,
    n_trials=1000,   # 🔥 YOU CAN SET 150–250 HERE
    n_jobs=1
)

BEST = study.best_params
BEST['cb_weight'] = 1 - BEST['cf_weight']

print("\n🔥 BEST CONFIG:", BEST)

# ===========================
# FINAL COMPARISON
# ===========================
print("\n🔥 FINAL COMPARISON")

old_mean, old_std = full_eval(BASE)
new_mean, new_std = full_eval(BEST)

print(f"\nOLD: {old_mean:.4f} ± {old_std:.4f}")
print(f"NEW: {new_mean:.4f} ± {new_std:.4f}")

if new_mean > old_mean:
    print("\n🏆 WINNER: OPTUNA CONFIG")
else:
    print("\n🏆 WINNER: BASE CONFIG")