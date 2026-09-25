import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

def clean_name(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\b(corp|corporation|inc|llc|pvt|private|ltd|limited)\b', '', text)
    return text.strip()

def tfidf_blocking(s1, s2, s3, threshold=0.3):
    print("Starting TF-IDF Blocking...")
    
    # Clean names
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
        
        chunk_size = 1000
        for i in range(0, tf_s1.shape[0], chunk_size):
            sim_matrix = cosine_similarity(tf_s1[i:i+chunk_size], tf_s23)
            s1_idx, s23_idx = np.where(sim_matrix > threshold)
            
            if len(s1_idx) > 0:
                chunk_candidates = pd.DataFrame({
                    'source1_entity_id': s1_c.loc[i + s1_idx, 'entity_id'].values,
                    'candidate_entity_id': s23_c.loc[s23_idx, 'entity_id'].values,
                    'similarity_score': sim_matrix[s1_idx, s23_idx]
                })
                candidates_list.append(chunk_candidates)
                
    all_candidates = pd.concat(candidates_list, ignore_index=True)
    all_candidates = all_candidates.sort_values(by=['source1_entity_id', 'similarity_score'], ascending=[True, False])
    
    print(f"Total candidate pairs generated: {len(all_candidates)}")
    return all_candidates

if __name__ == "__main__":
    print("Loading datasets...")
    # NOTE FOR VISHAL: Make sure these paths point to where dataset/ is stored!
    s1 = pd.read_csv("../../dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("../../dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("../../dataset/train/train_source3.tsv", sep="\t")
    
    train_candidates = tfidf_blocking(s1, s2, s3, threshold=0.3)
    
    # Save to parquet
    train_candidates.to_parquet("train_candidates.parquet", index=False)
    print("Saved to train_candidates.parquet. Hand this file over to Aravint and Navadeep!")
