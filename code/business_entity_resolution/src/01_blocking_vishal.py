import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import os

def clean_name(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(corp|corporation|inc|llc|pvt|private|ltd|limited)\b', '', text)
    return text.strip()

def tfidf_blocking(s1, s2, s3, threshold=0.3):
    print("Starting TF-IDF Sparse Blocking...")
    
    s1['clean_name'] = s1['business_name'].apply(clean_name)
    s23 = pd.concat([s2, s3], ignore_index=True)
    s23['clean_name'] = s23['business_name'].apply(clean_name)
    
    candidates_list = []
    countries = s1['country'].dropna().unique()
    
    for country in countries:
        print(f"Processing Country: {country}")
        
        s1_c = s1[s1['country'] == country].reset_index(drop=True)
        s23_c = s23[s23['country'] == country].reset_index(drop=True)
        
        if s1_c.empty or s23_c.empty: continue
        
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), min_df=2)
        all_names = pd.concat([s1_c['clean_name'], s23_c['clean_name']])
        vectorizer.fit(all_names)
        
        tf_s1 = vectorizer.transform(s1_c['clean_name'])
        tf_s23 = vectorizer.transform(s23_c['clean_name'])
        
        # --- THE RAM FIX (Sparse Dot Product) ---
        # Instead of dense matrices, we use native sparse matrix multiplication
        # This keeps the memory footprint tiny (megabytes instead of gigabytes!)
        sparse_sim = tf_s1.dot(tf_s23.T)
        
        # Convert to COOrdinate format to extract indices and values instantly
        coo = sparse_sim.tocoo()
        
        # Filter matches above our threshold
        mask = coo.data > threshold
        s1_idx = coo.row[mask]
        s23_idx = coo.col[mask]
        scores = coo.data[mask]
        
        if len(s1_idx) > 0:
            chunk_candidates = pd.DataFrame({
                'source1_entity_id': s1_c.loc[s1_idx, 'entity_id'].values,
                'candidate_entity_id': s23_c.loc[s23_idx, 'entity_id'].values,
                'similarity_score': scores
            })
            candidates_list.append(chunk_candidates)
                
    all_candidates = pd.concat(candidates_list, ignore_index=True)
    all_candidates = all_candidates.sort_values(by=['source1_entity_id', 'similarity_score'], ascending=[True, False])
    
    print(f"Total candidate pairs generated: {len(all_candidates)}")
    return all_candidates

if __name__ == "__main__":
    print("Loading datasets...")
    s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t")
    
    train_candidates = tfidf_blocking(s1, s2, s3, threshold=0.3)
    
    train_candidates.to_parquet("train_candidates.parquet", index=False)
    print("Saved to train_candidates.parquet.")
