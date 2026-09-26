import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import fbeta_score
import os

def compute_f05(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=0.5, zero_division=1.0)

if __name__ == "__main__":
    print("--- CHETAN'S MASTER MLOps SCRIPT ---")
    
    print("1. Loading Features from Aravint and Navadeep...")
    str_feats = pd.read_parquet("string_features.parquet")
    sem_feats = pd.read_parquet("semantic_features.parquet")
    
    # Merge them together
    df = str_feats.merge(sem_feats, on=['source1_entity_id', 'candidate_entity_id'])
    
    print("2. Loading Ground Truth and creating labels...")
    gt = pd.read_csv("student_resource/dataset/train/train_ground_truth.tsv", sep="\t")
    
    # Convert Ground Truth to a fast lookup set of (S1, S23) pairs
    gt_pairs = set()
    for _, row in gt.iterrows():
        s1_id = row['source1_entity_id']
        if pd.notna(row['matched_entity_ids']):
            for s23_id in str(row['matched_entity_ids']).split(','):
                gt_pairs.add((s1_id, s23_id))
                
    df['is_match'] = df.apply(lambda x: 1 if (x['source1_entity_id'], x['candidate_entity_id']) in gt_pairs else 0, axis=1)
    
    print(f"Total Candidate Pairs: {len(df)}")
    print(f"Match Rate in Candidates: {df['is_match'].mean():.4f}")
    
    print("3. Training LightGBM Model with GroupKFold...")
    features = [
        'similarity_score', # From Vishal's TF-IDF
        'name_ratio', 'name_token_sort', 'name_partial', # From Aravint
        'addr_ratio', 'addr_token_set', 'name_len_diff', # From Aravint
        'semantic_name_sim' # From Navadeep
    ]
    
    X = df[features]
    y = df['is_match']
    groups = df['source1_entity_id'] # Ensure S1 entities stay in the same fold
    
    gkf = GroupKFold(n_splits=5)
    
    # We will just train on Fold 1 and evaluate on Fold 2 for speed, 
    # but you can loop this for full Cross Validation.
    train_idx, val_idx = next(gkf.split(X, y, groups))
    
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    
    model = lgb.LGBMClassifier(n_estimators=200, random_state=42, learning_rate=0.05)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_metric='logloss')
    
    print("4. Tuning Threshold for F0.5 Score...")
    val_probs = model.predict_proba(X_val)[:, 1]
    
    best_thresh, best_f05 = 0.5, 0.0
    for thresh in np.arange(0.3, 0.95, 0.02):
        val_preds = (val_probs >= thresh).astype(int)
        score = compute_f05(y_val, val_preds)
        if score > best_f05:
            best_f05, best_thresh = score, thresh
            
    print(f"★ BEST VALIDATION F0.5 SCORE: {best_f05:.4f} (at threshold {best_thresh:.2f}) ★")
    
    print("5. Generating Output Files for Submission...")
    # For now, this just generates the output on the VAL set to prove it works.
    # To run on TEST set, simply point Vishal, Aravint, and Navadeep's scripts to dataset/test/
    
    val_df = df.iloc[val_idx].copy()
    val_df['match_prob'] = val_probs
    val_df['prediction'] = (val_df['match_prob'] >= best_thresh).astype(int)
    
    # Only keep predicted matches
    final_matches = val_df[val_df['prediction'] == 1]
    
    # Format matching_results.tsv
    match_out = final_matches.groupby('source1_entity_id')['candidate_entity_id'].apply(list).reset_index()
    match_out.columns = ['source1_entity_id', 'matched_entity_ids']
    match_out['matched_entity_ids'] = match_out['matched_entity_ids'].apply(lambda x: ",".join(x))
    
    # Ensure all S1 entities from the validation set are present
    all_val_s1 = pd.DataFrame({'source1_entity_id': val_df['source1_entity_id'].unique()})
    match_final = all_val_s1.merge(match_out, on='source1_entity_id', how='left').fillna("")
    
    os.makedirs("output", exist_ok=True)
    match_final.to_csv("output/matching_results.tsv", sep="\t", index=False)
    
    # Format candidate_pairs.tsv (just what Vishal produced, aggregated)
    cand_out = val_df.groupby('source1_entity_id')['candidate_entity_id'].apply(list).reset_index()
    cand_out.columns = ['source1_entity_id', 'candidate_entity_ids']
    cand_out['candidate_entity_ids'] = cand_out['candidate_entity_ids'].apply(lambda x: ",".join(x))
    cand_final = all_val_s1.merge(cand_out, on='source1_entity_id', how='left').fillna("")
    
    cand_final.to_csv("output/candidate_pairs.tsv", sep="\t", index=False)
    
    print("Done! Files generated in the output/ folder.")
    print("You are ready to rule the leaderboard!")
