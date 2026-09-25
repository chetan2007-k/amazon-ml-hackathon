import pandas as pd
import numpy as np
from rapidfuzz import fuzz

def calculate_string_features(name1, name2, addr1, addr2):
    n1, n2 = str(name1).lower(), str(name2).lower()
    a1, a2 = str(addr1).lower(), str(addr2).lower()
    
    return [
        fuzz.ratio(n1, n2),
        fuzz.token_sort_ratio(n1, n2),
        fuzz.partial_ratio(n1, n2),
        fuzz.ratio(a1, a2),
        fuzz.token_set_ratio(a1, a2),
        abs(len(n1) - len(n2))
    ]

if __name__ == "__main__":
    print("Loading Candidates and Raw Data...")
    candidates = pd.read_parquet("train_candidates.parquet")
    
    s1 = pd.read_csv("../../dataset/train/train_source1.tsv", sep="\t")
    s2 = pd.read_csv("../../dataset/train/train_source2.tsv", sep="\t")
    s3 = pd.read_csv("../../dataset/train/train_source3.tsv", sep="\t")
    s23 = pd.concat([s2, s3], ignore_index=True)
    
    print("Merging Text Data...")
    df = candidates.merge(s1[['entity_id', 'business_name', 'business_address']], left_on='source1_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name_s1', 'business_address': 'addr_s1'}).drop(columns=['entity_id'])
    
    df = df.merge(s23[['entity_id', 'business_name', 'business_address']], left_on='candidate_entity_id', right_on='entity_id', how='left')
    df = df.rename(columns={'business_name': 'name_s23', 'business_address': 'addr_s23'}).drop(columns=['entity_id'])
    
    print("Calculating RapidFuzz Features (This might take a minute)...")
    # Using a vectorized approach for speed
    features = df.apply(lambda x: calculate_string_features(x['name_s1'], x['name_s23'], x['addr_s1'], x['addr_s23']), axis=1, result_type='expand')
    features.columns = ['name_ratio', 'name_token_sort', 'name_partial', 'addr_ratio', 'addr_token_set', 'name_len_diff']
    
    # Combine back
    final_features = pd.concat([df[['source1_entity_id', 'candidate_entity_id', 'similarity_score']], features], axis=1)
    
    print("Saving to string_features.parquet...")
    final_features.to_parquet("string_features.parquet", index=False)
    print("Aravint's job is done!")
