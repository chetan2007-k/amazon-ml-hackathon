import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import paired_cosine_distances
import torch

if __name__ == "__main__":
    print("Loading Candidates and Raw Data...")
    candidates = pd.read_parquet("train_candidates.parquet")
    
    s1 = pd.read_csv("student_resource/dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("student_resource/dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("student_resource/dataset/train/train_source3.tsv", sep="\t")
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    print("Merging Text Data...")
    df = candidates.merge(s1[['entity_id', 'business_name']], left_on='source1_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name_s1'}).drop(columns=['entity_id'])
    
    df = df.merge(s23[['entity_id', 'business_name']], left_on='candidate_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name_s23'}).drop(columns=['entity_id'])
    
    # Fill NAs
    df['name_s1'] = df['name_s1'].fillna("")
    df['name_s23'] = df['name_s23'].fillna("")
    
    print("Loading HuggingFace MiniLM Model...")
    # Use GPU if available, else CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    
    print("Generating Embeddings (This uses heavy compute)...")
    # Encode unique names first to save massive time
    unique_s1 = df['name_s1'].unique()
    unique_s23 = df['name_s23'].unique()
    
    dict_s1 = {name: emb for name, emb in zip(unique_s1, model.encode(unique_s1, batch_size=256, show_progress_bar=True))}
    dict_s23 = {name: emb for name, emb in zip(unique_s23, model.encode(unique_s23, batch_size=256, show_progress_bar=True))}
    
    print("Mapping Embeddings to Candidates and Computing Cosine Distance...")
    emb1 = np.vstack(df['name_s1'].map(dict_s1).values)
    emb2 = np.vstack(df['name_s23'].map(dict_s23).values)
    
    # Cosine Similarity is 1 - Cosine Distance
    df['semantic_name_sim'] = 1 - paired_cosine_distances(emb1, emb2)
    
    final_features = df[['source1_entity_id', 'candidate_entity_id', 'semantic_name_sim']]
    
    print("Saving to semantic_features.parquet...")
    final_features.to_parquet("semantic_features.parquet", index=False)
    print("Navadeep's job is done!")
