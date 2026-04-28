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

# 🔥 FINAL FIXED WEIGHTS
PARAMS = {
    'w1': 0.85,   # CF (dominant)
    'w2': 0.10,   # Graph
    'w3': 0.05    # Pop (small)
}

print("\n[STEP 1] Vibranium Engine (No Optuna)...")

ratings_df = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.data',
                         sep='\t',
                         names=['user_id','item_id','rating','timestamp'])

metadata_df = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
metadata_df.rename(columns={'movie_id':'item_id'}, inplace=True)

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

# --- Genre ---
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
                     shape=(n_users+n_items, n_users+n_items))

    d = np.array(Adj.sum(axis=1)).flatten()
    D_inv = diags(np.power(d, -0.5, where=d!=0))

    L = D_inv @ Adj @ D_inv

    curr = L
    for _ in range(n_layers-1):
        curr = curr @ L

    return curr[:n_users, n_users:].toarray()

# --- Setup ---
n_users = ratings_df['user_id'].max()
n_items = len(metadata_df)

u_idx = ratings_df['user_id'].values - 1
i_idx = ratings_df['item_id'].values - 1

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

for fold,(tr,te) in enumerate(skf.split(ratings_df, ratings_df['user_id']),1):

    print(f"\n[FOLD {fold}] Running...")

    max_ts = ratings_df['timestamp'].max()
    decay = np.power(0.5,
                     (max_ts - ratings_df['timestamp']) /
                     (86400 * CONFIG['half_life_days']))

    adj_r = ratings_df['rating'].values.astype(float).copy()
    adj_r[adj_r < 3] *= -CONFIG['negative_penalty']

    X = coo_matrix((adj_r[tr]*decay[tr], (u_idx[tr], i_idx[tr])),
                   shape=(n_users,n_items)).toarray()

    Xo = coo_matrix((ratings_df['rating'].values[tr], (u_idx[tr], i_idx[tr])),
                    shape=(n_users,n_items)).toarray()

    # CF
    G = X.T @ X + np.eye(n_items)*CONFIG['lambda_ease']
    B = np.linalg.inv(G)
    B /= -np.diag(B)[:,None]
    np.fill_diagonal(B,0)

    cf = normalize(X @ B, norm='l2')

    # Graph
    graph = normalize(get_graph_signals(X), norm='l2')

    # Fusion
    s = PARAMS['w1'] + PARAMS['w2'] + PARAMS['w3']

    final_scores = (
        (PARAMS['w1']/s)*cf +
        (PARAMS['w2']/s)*graph +
        (PARAMS['w3']/s)*pop_signal
    )

    test = ratings_df.iloc[te]
    test = test[test['rating']>=4]
    td = test.groupby('user_id')['item_id'].apply(lambda x: x.values-1).to_dict()

    item_counts = (Xo>0).sum(axis=0)
    novelty = -np.log2(np.maximum(item_counts/n_users,1e-9))

    aps, ndcgs, aps_f, ndcgs_f, novs, divs = [],[],[],[],[],[]
    all_rec = set()

    for u in range(n_users):
        t = td.get(u+1,[])
        if len(t)==0: continue

        s_u = final_scores[u].copy()
        s_u[Xo[u]>0] = -1e9

        rank = np.argsort(-s_u)
        top10 = rank[:10]

        all_rec.update(top10)

        hits = np.isin(top10,t)

        aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(t),10) if hits.sum()>0 else 0)

        dcg = np.sum(hits/np.log2(np.arange(2,12)))
        idcg = np.sum(1/np.log2(np.arange(2,min(len(t),10)+2)))
        ndcgs.append(dcg/idcg if idcg>0 else 0)

        hits_f = np.isin(rank,t)
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

    results.append({
        'Fold': fold,
        'MAP (Full)': np.mean(aps_f),
        'MAP@10': np.mean(aps),
        'nDCG (Full)': np.mean(ndcgs_f),
        'nDCG@10': np.mean(ndcgs),
        'Coverage (%)': (len(all_rec)/n_items)*100,
        'Novelty': np.mean(novs),
        'Diversity': np.mean(divs)
    })

# --- Final ---
print("\n" + "="*80)
print("🏆 VIBRANIUM FINAL (NO OPTUNA)")
print("="*80)

df = pd.DataFrame(results)
print(df.to_string(index=False))

print("\n[AGGREGATED]")
for col in df.columns[1:]:
    print(f"{col}: {df[col].mean():.4f} ± {df[col].std():.4f}")