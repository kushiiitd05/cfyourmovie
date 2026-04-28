# #!/usr/bin/env python3
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# import numpy as np
# import pandas as pd
# import optuna
# from pathlib import Path
# from scipy.sparse import csr_matrix, coo_matrix, hstack
# from sklearn.preprocessing import normalize
# from sklearn.model_selection import StratifiedKFold
# from sentence_transformers import SentenceTransformer

# CONFIG = {
#     'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
#     'lambda_ease': 2199.9395646690464,
#     'half_life_days': 218,
#     'negative_penalty': 0.9669969952037883,
#     'strict_filter': 4,
#     'optuna_trials': 296
# }

# raw = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.data', sep='\t',
#                   names=['user_id','item_id','rating','timestamp'])
# meta = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
# meta.rename(columns={'movie_id':'item_id'}, inplace=True)

# max_ts = raw['timestamp'].max()
# days = (max_ts - raw['timestamp']) / 86400
# decay = np.power(0.5, days / CONFIG['half_life_days'])
# raw['wr'] = raw['rating'] * decay

# n_users = raw['user_id'].max()
# n_items = len(meta)

# u_idx = raw['user_id'].values - 1
# i_idx = raw['item_id'].values - 1

# r_w = raw['wr'].values
# r_o = raw['rating'].values

# model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')

# emb = normalize(model.encode(
#     (meta['overview'].fillna('') + " " +
#      meta['movie_keywords'].fillna('').astype(str) + " " +
#      meta['top_cast'].fillna('').astype(str)).tolist(),
#     batch_size=64
# ))

# genres = []
# all_g = set()
# for g in meta['genres'].dropna():
#     try: all_g.update(eval(g))
#     except: pass
# gl = sorted(list(all_g))
# for g in meta['genres']:
#     try: gset = set(eval(g)) if pd.notna(g) else set()
#     except: gset = set()
#     genres.append([1.0 if x in gset else 0.0 for x in gl])
# genres = csr_matrix(np.array(genres))

# nums = []
# for col in ['budget','revenue','vote_average','vote_count','popularity']:
#     v = np.log1p(meta[col].fillna(0).values)
#     v = (v - v.mean())/(v.std()+1e-8)
#     nums.append(v)
# nums = csr_matrix(np.array(nums).T)

# def run(alpha_struct, alpha_embed, beta, w_genre, w_num, perc):

#     content_struct = normalize(hstack([
#         genres * w_genre,
#         nums * w_num
#     ]))

#     content_embed = csr_matrix(emb)

#     skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#     results = []

#     for fold,(tr,te) in enumerate(skf.split(raw, raw['user_id']),1):

#         X = coo_matrix((r_w[tr], (u_idx[tr], i_idx[tr])), shape=(n_users,n_items)).toarray()
#         Xo = coo_matrix((r_o[tr], (u_idx[tr], i_idx[tr])), shape=(n_users,n_items)).toarray()

#         test = raw.iloc[te]
#         test = test[test['rating']>=CONFIG['strict_filter']]
#         td = test.groupby('user_id')['item_id'].apply(lambda x: x.values-1).to_dict()

#         item_counts = (Xo>0).sum(axis=0)
#         popularity = np.log1p(item_counts)
#         novelty = -np.log2(np.maximum(item_counts/n_users,1e-9))

#         G = X.T @ X
#         G[np.diag_indices_from(G)] += CONFIG['lambda_ease']
#         P = np.linalg.inv(G)
#         B = P / (-np.diag(P)[:,None])
#         np.fill_diagonal(B,0)
#         cf = X @ B

#         up_struct = np.zeros((n_users, content_struct.shape[1]))
#         up_embed = np.zeros((n_users, content_embed.shape[1]))

#         for u in range(n_users):
#             liked = np.where(Xo[u]>=CONFIG['strict_filter'])[0]
#             hated = np.where((Xo[u]>0)&(Xo[u]<3))[0]

#             if len(liked)>0:
#                 ps = content_struct[liked].mean(axis=0).A1
#                 pe = content_embed[liked].mean(axis=0).A1

#                 if len(hated)>0:
#                     ps = np.maximum(ps - CONFIG['negative_penalty']*content_struct[hated].mean(axis=0).A1,0)
#                     pe = np.maximum(pe - CONFIG['negative_penalty']*content_embed[hated].mean(axis=0).A1,0)

#                 up_struct[u] = ps/(np.linalg.norm(ps)+1e-8)
#                 up_embed[u] = pe/(np.linalg.norm(pe)+1e-8)

#         cb_struct = up_struct @ content_struct.T.toarray()
#         cb_embed = up_embed @ content_embed.T.toarray()

#         cf = (cf - cf.mean(axis=1,keepdims=True))/(cf.std(axis=1,keepdims=True)+1e-8)
#         cb_struct = (cb_struct - cb_struct.mean(axis=1,keepdims=True))/(cb_struct.std(axis=1,keepdims=True)+1e-8)
#         cb_embed = (cb_embed - cb_embed.mean(axis=1,keepdims=True))/(cb_embed.std(axis=1,keepdims=True)+1e-8)

#         threshold = np.percentile(cf, perc)
#         mask = (cf < threshold)

#         hybrid = cf + alpha_struct*cb_struct + alpha_embed*(cb_embed * mask)
#         hybrid = hybrid - beta * popularity

#         aps, ndcgs, novs, divs = [], [], [], []
#         aps_full, ndcgs_full = [], []
#         all_rec = set()

#         for u in range(n_users):
#             t = td.get(u+1,[])
#             if len(t)==0: continue

#             s = hybrid[u].copy()
#             s[Xo[u]>0] = -np.inf

#             rank = np.argsort(-s)
#             top10 = rank[:10]

#             all_rec.update(top10)
#             novs.append(np.mean(novelty[top10]))

#             feats = content_struct[top10].toarray()
#             sim = feats @ feats.T
#             divs.append(1 - np.mean(sim[np.triu_indices(10,1)]))

#             hits = np.isin(top10,t)
#             aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(t),10) if hits.sum()>0 else 0)

#             dcg = np.sum(hits/np.log2(np.arange(2,12)))
#             idcg = np.sum(1/np.log2(np.arange(2,min(len(t),10)+2)))
#             ndcgs.append(dcg/idcg if idcg>0 else 0)

#             hits_full = np.isin(rank,t)
#             if hits_full.sum()>0:
#                 prec = np.cumsum(hits_full)/(np.arange(len(rank))+1)
#                 aps_full.append(prec[hits_full].sum()/len(t))
#             else:
#                 aps_full.append(0)

#             dcg_f = np.sum(hits_full/np.log2(np.arange(2,len(rank)+2)))
#             idcg_f = np.sum(1/np.log2(np.arange(2,len(t)+2)))
#             ndcgs_full.append(dcg_f/idcg_f if idcg_f>0 else 0)

#         results.append({
#             'MAP (Full)': np.mean(aps_full),
#             'MAP@10': np.mean(aps),
#             'nDCG (Full)': np.mean(ndcgs_full),
#             'nDCG@10': np.mean(ndcgs),
#             'Coverage (%)': (len(all_rec)/n_items)*100,
#             'Novelty': np.mean(novs),
#             'Diversity': np.mean(divs)
#         })

#     df = pd.DataFrame(results)
#     return df['MAP@10'].mean(), df

# def objective(trial):
#     alpha_struct = trial.suggest_float("alpha_struct", 0.10, 0.41)
#     alpha_embed  = trial.suggest_float("alpha_embed", 0.01, 0.06)
#     beta         = trial.suggest_float("beta", 0.005, 0.06)
#     w_genre      = trial.suggest_float("w_genre", 1.0, 2.2)
#     w_num        = trial.suggest_float("w_num", 0.7, 1.7)
#     perc         = trial.suggest_int("perc", 60, 85)

#     score,_ = run(alpha_struct, alpha_embed, beta, w_genre, w_num, perc)
#     return score

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=CONFIG['optuna_trials'])

# print("\nBEST:", study.best_params)

# best = study.best_params
# score, df = run(best['alpha_struct'], best['alpha_embed'], best['beta'],
#                 best['w_genre'], best['w_num'], best['perc'])

# print("\nFINAL RESULTS")
# print(df.to_string(index=False))

# print("\nAGGREGATED")
# print("MAP (Full):", df['MAP (Full)'].mean())
# print("MAP@10:", df['MAP@10'].mean())
# print("nDCG (Full):", df['nDCG (Full)'].mean())
# print("nDCG@10:", df['nDCG@10'].mean())
# print("Coverage:", df['Coverage (%)'].mean())
# print("Novelty:", df['Novelty'].mean())
# print("Diversity:", df['Diversity'].mean())

#!/usr/bin/env python3
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix, coo_matrix, hstack
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer

CONFIG = {
    'data_dir': Path('/home/vinayak23597/Kush/cf_final_project'),
    'lambda_ease': 2199.9395646690464,
    'half_life_days': 218,
    'negative_penalty': 0.9669969952037883,
    'strict_filter': 4
}

PARAMS = {
    'alpha_struct': 0.10008381411545013,
    'alpha_embed': 0.03721010352958694,
    'beta': 0.013144045208014172,
    'w_genre': 1.7638679280017653,
    'w_num': 0.7181982665890585,
    'perc': 74
}

raw = pd.read_csv(CONFIG['data_dir'] / 'ml-100k/u.data', sep='\t',
                  names=['user_id','item_id','rating','timestamp'])
meta = pd.read_csv(CONFIG['data_dir'] / 'Master_final.csv')
meta.rename(columns={'movie_id':'item_id'}, inplace=True)

max_ts = raw['timestamp'].max()
days = (max_ts - raw['timestamp']) / 86400
decay = np.power(0.5, days / CONFIG['half_life_days'])
raw['wr'] = raw['rating'] * decay

n_users = raw['user_id'].max()
n_items = len(meta)

u_idx = raw['user_id'].values - 1
i_idx = raw['item_id'].values - 1

r_w = raw['wr'].values
r_o = raw['rating'].values

model = SentenceTransformer("all-MiniLM-L6-v2", device='cuda')

emb = normalize(model.encode(
    (meta['overview'].fillna('') + " " +
     meta['movie_keywords'].fillna('').astype(str) + " " +
     meta['top_cast'].fillna('').astype(str)).tolist(),
    batch_size=64
))

genres = []
all_g = set()
for g in meta['genres'].dropna():
    try: all_g.update(eval(g))
    except: pass
gl = sorted(list(all_g))
for g in meta['genres']:
    try: gset = set(eval(g)) if pd.notna(g) else set()
    except: gset = set()
    genres.append([1.0 if x in gset else 0.0 for x in gl])
genres = csr_matrix(np.array(genres))

nums = []
for col in ['budget','revenue','vote_average','vote_count','popularity']:
    v = np.log1p(meta[col].fillna(0).values)
    v = (v - v.mean())/(v.std()+1e-8)
    nums.append(v)
nums = csr_matrix(np.array(nums).T)

def run():

    content_struct = normalize(hstack([
        genres * PARAMS['w_genre'],
        nums * PARAMS['w_num']
    ]))

    content_embed = csr_matrix(emb)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold,(tr,te) in enumerate(skf.split(raw, raw['user_id']),1):

        X = coo_matrix((r_w[tr], (u_idx[tr], i_idx[tr])), shape=(n_users,n_items)).toarray()
        Xo = coo_matrix((r_o[tr], (u_idx[tr], i_idx[tr])), shape=(n_users,n_items)).toarray()

        test = raw.iloc[te]
        test = test[test['rating']>=CONFIG['strict_filter']]
        td = test.groupby('user_id')['item_id'].apply(lambda x: x.values-1).to_dict()

        item_counts = (Xo>0).sum(axis=0)
        popularity = np.log1p(item_counts)
        novelty = -np.log2(np.maximum(item_counts/n_users,1e-9))

        G = X.T @ X
        G[np.diag_indices_from(G)] += CONFIG['lambda_ease']
        P = np.linalg.inv(G)
        B = P / (-np.diag(P)[:,None])
        np.fill_diagonal(B,0)
        cf = X @ B

        up_struct = np.zeros((n_users, content_struct.shape[1]))
        up_embed = np.zeros((n_users, content_embed.shape[1]))

        for u in range(n_users):
            liked = np.where(Xo[u]>=CONFIG['strict_filter'])[0]
            hated = np.where((Xo[u]>0)&(Xo[u]<3))[0]

            if len(liked)>0:
                ps = content_struct[liked].mean(axis=0).A1
                pe = content_embed[liked].mean(axis=0).A1

                if len(hated)>0:
                    ps = np.maximum(ps - CONFIG['negative_penalty']*content_struct[hated].mean(axis=0).A1,0)
                    pe = np.maximum(pe - CONFIG['negative_penalty']*content_embed[hated].mean(axis=0).A1,0)

                up_struct[u] = ps/(np.linalg.norm(ps)+1e-8)
                up_embed[u] = pe/(np.linalg.norm(pe)+1e-8)

        cb_struct = up_struct @ content_struct.T.toarray()
        cb_embed = up_embed @ content_embed.T.toarray()

        cf = (cf - cf.mean(axis=1,keepdims=True))/(cf.std(axis=1,keepdims=True)+1e-8)
        cb_struct = (cb_struct - cb_struct.mean(axis=1,keepdims=True))/(cb_struct.std(axis=1,keepdims=True)+1e-8)
        cb_embed = (cb_embed - cb_embed.mean(axis=1,keepdims=True))/(cb_embed.std(axis=1,keepdims=True)+1e-8)

        threshold = np.percentile(cf, PARAMS['perc'])
        mask = (cf < threshold)

        hybrid = cf + PARAMS['alpha_struct']*cb_struct + PARAMS['alpha_embed']*(cb_embed * mask)
        hybrid = hybrid - PARAMS['beta'] * popularity

        aps, ndcgs, novs, divs = [], [], [], []
        aps_full, ndcgs_full = [], []
        all_rec = set()

        for u in range(n_users):
            t = td.get(u+1,[])
            if len(t)==0: continue

            s = hybrid[u].copy()
            s[Xo[u]>0] = -np.inf

            rank = np.argsort(-s)
            top10 = rank[:10]

            all_rec.update(top10)
            novs.append(np.mean(novelty[top10]))

            feats = content_struct[top10].toarray()
            sim = feats @ feats.T
            divs.append(1 - np.mean(sim[np.triu_indices(10,1)]))

            hits = np.isin(top10,t)
            aps.append(np.sum(np.cumsum(hits)/np.arange(1,11)*hits)/min(len(t),10) if hits.sum()>0 else 0)

            dcg = np.sum(hits/np.log2(np.arange(2,12)))
            idcg = np.sum(1/np.log2(np.arange(2,min(len(t),10)+2)))
            ndcgs.append(dcg/idcg if idcg>0 else 0)

            hits_full = np.isin(rank,t)
            if hits_full.sum()>0:
                prec = np.cumsum(hits_full)/(np.arange(len(rank))+1)
                aps_full.append(prec[hits_full].sum()/len(t))
            else:
                aps_full.append(0)

            dcg_f = np.sum(hits_full/np.log2(np.arange(2,len(rank)+2)))
            idcg_f = np.sum(1/np.log2(np.arange(2,len(t)+2)))
            ndcgs_full.append(dcg_f/idcg_f if idcg_f>0 else 0)

        results.append({
            'MAP (Full)': np.mean(aps_full),
            'MAP@10': np.mean(aps),
            'nDCG (Full)': np.mean(ndcgs_full),
            'nDCG@10': np.mean(ndcgs),
            'Coverage (%)': (len(all_rec)/n_items)*100,
            'Novelty': np.mean(novs),
            'Diversity': np.mean(divs)
        })

    df = pd.DataFrame(results)

    print("\nFINAL RESULTS")
    print(df.to_string(index=False))

    print("\nAGGREGATED")
    print("MAP (Full):", df['MAP (Full)'].mean())
    print("MAP@10:", df['MAP@10'].mean())
    print("nDCG (Full):", df['nDCG (Full)'].mean())
    print("nDCG@10:", df['nDCG@10'].mean())
    print("Coverage:", df['Coverage (%)'].mean())
    print("Novelty:", df['Novelty'].mean())
    print("Diversity:", df['Diversity'].mean())

run()