import argparse
import os
import csv
import pandas as pd
import numpy as np
import random
import xgboost as xgb
import pickle
import math
import re
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import fbeta_score

def load_ground_truth(file_path):
    print("Loading ground truth...")
    truth_df = pd.read_csv(file_path, sep='\t')
    positive_pairs = set()
    
    for _, row in truth_df.iterrows():
        s1_id = row['source1_entity_id']
        matched = str(row['matched_entity_ids']).split(',')
        for m in matched:
            if m and m.strip() != 'nan':
                positive_pairs.add((s1_id, m.strip()))
                
    return positive_pairs

def get_entity_dfs(split, data_dir):
    print(f"Loading entity texts into dataframes for {split}...")
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    
    s1_df = pd.read_csv(os.path.join(data_dir, f"{split}_source1.tsv"), sep='\t', usecols=cols, dtype='string').fillna('')
    s2_df = pd.read_csv(os.path.join(data_dir, f"{split}_source2.tsv"), sep='\t', usecols=cols, dtype='string').fillna('')
    s3_df = pd.read_csv(os.path.join(data_dir, f"{split}_source3.tsv"), sep='\t', usecols=cols, dtype='string').fillna('')
    
    # We keep S1 separate, and combine S2/S3
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    return s1_df, s23_df

def compute_features(df_pairs, s1_df, s23_df):
    print("Merging candidate pairs with their raw text strings...")
    # Merge S1 texts
    merged = df_pairs.merge(s1_df, left_on='source1_entity_id', right_on='entity_id', how='inner')
    merged = merged.rename(columns={'business_name': 'name1', 'business_address': 'addr1', 'country': 'country1'})
    merged.drop('entity_id', axis=1, inplace=True)
    
    # Merge Candidate texts
    merged = merged.merge(s23_df, left_on='candidate_entity_id', right_on='entity_id', how='inner')
    merged = merged.rename(columns={'business_name': 'name2', 'business_address': 'addr2', 'country': 'country2'})
    merged.drop('entity_id', axis=1, inplace=True)
    
    print("Computing Kilo-optimized string similarity features...")
    features = []
    
    # We iterate over the merged dataframe which only contains the pairs we actually care about
    for _, row in merged.iterrows():
        name1, name2 = str(row['name1']).lower(), str(row['name2']).lower()
        addr1, addr2 = str(row['addr1']).lower(), str(row['addr2']).lower()
        
        # Jaro-Winkler
        name_jw = JaroWinkler.normalized_similarity(name1, name2)
        addr_jw = JaroWinkler.normalized_similarity(addr1, addr2)
        
        # Token Jaccard with Abbreviations
        name_jaccard = token_jaccard(name1, name2)
        addr_jaccard = token_jaccard(addr1, addr2)
        
        # RapidFuzz Partial Ratio & Token Sort Ratio
        name_partial = fuzz.partial_token_set_ratio(name1, name2) / 100.0
        addr_partial = fuzz.partial_token_set_ratio(addr1, addr2) / 100.0
        
        name_sort = fuzz.token_sort_ratio(name1, name2) / 100.0
        addr_sort = fuzz.token_sort_ratio(addr1, addr2) / 100.0
        
        country_match = 1.0 if row['country1'] == row['country2'] and row['country1'] != '' else 0.0
        
        features.append({
            'name_jw': name_jw,
            'addr_jw': addr_jw,
            'name_jaccard': name_jaccard,
            'addr_jaccard': addr_jaccard,
            'name_partial': name_partial,
            'addr_partial': addr_partial,
            'name_sort': name_sort,
            'addr_sort': addr_sort,
            'country_match': country_match
        })
        
    feat_df = pd.DataFrame(features)
    return feat_df

def build_training_dataset(candidates_file, positive_pairs):
    print("Loading candidate pairs from blocking stage...")
    candidates = pd.read_csv(candidates_file, sep="\t")
    
    positive_list = []
    negative_list = []
    
    # Flatten candidates
    for _, row in candidates.iterrows():
        s1_id = row['source1_entity_id']
        cands = str(row['candidate_entity_ids']).split(',')
        for c in cands:
            if c and c.strip() != 'nan':
                is_pos = (s1_id, c.strip()) in positive_pairs
                if is_pos:
                    positive_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': c.strip(), 'label': 1})
                else:
                    if random.random() < 0.05:  # Keep 5% of negatives on the fly to avoid OOM
                        negative_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': c.strip(), 'label': 0})
                
    # Inject missed positives
    candidate_set = set((x['source1_entity_id'], x['candidate_entity_id']) for x in positive_list)
    missed_positives = positive_pairs - candidate_set
    for s1_id, p_id in missed_positives:
        positive_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': p_id, 'label': 1})
        
    # Final clamp to 5:1 ratio
    target_negatives = len(positive_list) * 5
    if len(negative_list) > target_negatives:
        print(f"Subsampling {len(negative_list)} negatives down to {target_negatives}...")
        negative_list = random.sample(negative_list, target_negatives)
        
    pair_list = positive_list + negative_list
    random.shuffle(pair_list)
    df_pairs = pd.DataFrame(pair_list)
    print(f"Total training pairs: {len(df_pairs)} (Positives: {df_pairs['label'].sum()})")
    
    return df_pairs

_ABBREV = {
    r'\bpvt\b': 'private',
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bllc\b': 'limited liability company',
    r'\bltd\b': 'limited',
    r'\bco\b': 'company',
    r'\bintl\b': 'international',
    r'\bstr\b': 'street',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\brd\b': 'road',
    r'\bln\b': 'lane',
    r'\bapt\b': 'apartment',
    r'\bste\b': 'suite',
    r'\bfl\b': 'floor',
    r'\bsoln\.?\b': 'solution',
    r'\bma\b': 'massachusetts'
}

def expand_abbrev(text: str) -> str:
    for pat, repl in _ABBREV.items():
        text = re.sub(pat, repl, text)
    return re.sub(r'\s+', ' ', text).strip()

def token_jaccard(a: str, b: str) -> float:
    A = set(expand_abbrev(a).split())
    B = set(expand_abbrev(b).split())
    if not A and not B: return 1.0
    return len(A & B) / len(A | B)

def optimize_threshold(probs, y_true):
    print("Calibrating decision threshold to maximize F0.5 Macro score...")
    best_t = 0.5
    best_f05 = 0.0
    
    for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        preds = (probs > t).astype(int)
        score = fbeta_score(y_true, preds, beta=0.5, average='macro', zero_division=0)
        if score > best_f05:
            best_f05 = score
            best_t = t
            
    print(f"Optimal Threshold: {best_t} (Estimated Training F0.5: {best_f05:.4f})")
    return best_t

def train_model(X, y):
    print("Training XGBoost with Kilo's precision-first hyperparameters...")
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_weight = min(neg_count / pos_count, 200.0) if pos_count > 0 else 5.0
    
    clf = xgb.XGBClassifier(
        tree_method='hist',
        device='cuda',
        objective='binary:logistic',
        scale_pos_weight=scale_weight,
        max_depth=5,
        gamma=0.5,
        min_child_weight=5,
        learning_rate=0.03,
        max_delta_step=5,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=300,
        eval_metric='aucpr',
        n_jobs=-1
    )
    clf.fit(X, y)
    
    # Train-set threshold calibration
    probs = clf.predict_proba(X)[:, 1]
    best_t = optimize_threshold(probs, y)
    
    return clf, best_t

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset/train")
    parser.add_argument("--candidates_file", type=str, default="output/train_candidates.tsv")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"])
    parser.add_argument("--model_path", type=str, default="output/xgb_model.pkl")
    parser.add_argument("--out_file", type=str, default="output/matching_results.tsv")
    args = parser.parse_args()
    
    if args.mode == "train":
        gt_file = os.path.join(args.data_dir, "train_ground_truth.tsv")
        positive_pairs = load_ground_truth(gt_file)
        
        df_pairs = build_training_dataset(args.candidates_file, positive_pairs)
        s1_df, s23_df = get_entity_dfs("train", args.data_dir)
        
        X = compute_features(df_pairs, s1_df, s23_df)
        y = df_pairs['label']
        
        model, best_t = train_model(X, y)
        with open(args.model_path, 'wb') as f:
            pickle.dump({'model': model, 'threshold': best_t}, f)
            
    else:
        print("Loading test candidate pairs...")
        test_candidates = pd.read_csv(args.candidates_file, sep="\t")
        s1_df, s23_df = get_entity_dfs("test", args.data_dir)
        
        pair_list = []
        for _, row in test_candidates.iterrows():
            s1_id = row['source1_entity_id']
            cands = str(row['candidate_entity_ids']).split(',')
            for c in cands:
                if c and c.strip() != 'nan':
                    pair_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': c.strip()})
                    
        df_pairs = pd.DataFrame(pair_list)
        X = compute_features(df_pairs, s1_df, s23_df)
        
        with open(args.model_path, 'rb') as f:
            data = pickle.load(f)
            model = data['model']
            threshold = data['threshold']
            
        print(f"Predicting matches with Calibrated F0.5 threshold ({threshold})...")
        probs = model.predict_proba(X)[:, 1]
        
        df_pairs['is_match'] = (probs > threshold).astype(int)
        
        matches = df_pairs[df_pairs['is_match'] == 1].groupby('source1_entity_id')['candidate_entity_id'].apply(list).reset_index()
        match_dict = dict(zip(matches['source1_entity_id'], matches['candidate_entity_id']))
        
        print(f"Writing final submission to {args.out_file}...")
        with open(args.out_file, 'w') as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for _, row in test_candidates.iterrows():
                s1_id = row['source1_entity_id']
                m = match_dict.get(s1_id, [])
                f.write(f"{s1_id}\t{','.join(m)}\n")

if __name__ == "__main__":
    main()
