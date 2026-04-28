import pandas as pd
import numpy as np
import ast
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

def build_adamantium_features(df):
    """
    Input: Master_final metadata (df)
    Output: Compressed High-Dimensional Feature Matrix
    """
    print(f"[PREP] Processing {len(df)} movies...")

    # 1. Helper Function to Clean String Lists
    def clean_list(x):
        try:
            return " ".join([str(i).replace(" ", "_") for i in ast.literal_eval(x)])
        except:
            return ""

    # 2. Extract Textual Features (The "Big 4")
    print("   - Parsing Lists (Genres, Keywords, Cast, Companies)...")
    df['clean_genres'] = df['genres'].apply(clean_list)
    df['clean_keywords'] = df['movie_keywords'].apply(clean_list)
    df['clean_cast'] = df['top_cast'].apply(clean_list)
    df['clean_prod'] = df['production_companies'].apply(clean_list)

    # 3. Vectorization with Frequency Filtering (Min_DF)
    # Isse 2200+ uniques filter ho jayenge!
    print("   - Vectorizing with Frequency Filtering...")
    
    # Genres are strict, so we take all
    cv_gen = CountVectorizer(binary=True)
    feat_gen = cv_gen.fit_transform(df['clean_genres'])
    
    # Keywords/Cast: Only keep those that appear in at least 3 movies
    cv_kw = CountVectorizer(binary=True, min_df=3) 
    feat_kw = cv_kw.fit_transform(df['clean_keywords'])
    
    cv_cast = CountVectorizer(binary=True, min_df=2)
    feat_cast = cv_cast.fit_transform(df['clean_cast'])
    
    cv_prod = CountVectorizer(binary=True, min_df=2)
    feat_prod = cv_prod.fit_transform(df['clean_prod'])

    print(f"     ✓ Genres: {feat_gen.shape[1]}")
    print(f"     ✓ Keywords: {feat_kw.shape[1]} (Filtered from thousands)")
    print(f"     ✓ Cast: {feat_cast.shape[1]}")

    # 4. Numeric Features (Log Scaling)
    print("   - Scaling Numeric Features (Budget, Popularity, etc.)...")
    num_cols = ['budget', 'revenue', 'vote_average', 'vote_count', 'popularity', 'year']
    df_num = df[num_cols].fillna(0)
    
    # Log transform for huge numbers like budget/revenue
    for col in ['budget', 'revenue']:
        df_num[col] = np.log1p(df_num[col])
        
    scaler = MinMaxScaler()
    feat_num = csr_matrix(scaler.fit_transform(df_num))

    # 5. The Adamantium Merge
    full_matrix = hstack([feat_gen, feat_kw, feat_cast, feat_prod, feat_num]).tocsr()
    
    # Normalize so that long lists don't dominate the math
    full_matrix = normalize(full_matrix, norm='l2')

    print(f"\n[SUCCESS] Final Feature Matrix Shape: {full_matrix.shape}")
    print(f"[INFO] Total Unique 'Meaningful' Features: {full_matrix.shape[1]}")
    
    return full_matrix, df['item_id'].values

# --- MAIN EXECUTION ---
# metadata_df = pd.read_csv('Master_final.csv') # Load your actual file here
# feature_matrix, item_ids = build_adamantium_features(metadata_df)
# Load metadata
metadata_df = pd.read_csv('/Users/bournesmartasfuck_kush/Desktop/cf/Master_final.csv')

# 🔥 IS LINE KO ADD KAR: Rename movie_id to item_id
metadata_df.rename(columns={'movie_id': 'item_id'}, inplace=True)

# Ab function call kar
feature_matrix, item_ids = build_adamantium_features(metadata_df)