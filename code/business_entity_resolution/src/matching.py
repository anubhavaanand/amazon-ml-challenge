import argparse
import os
import csv
import pandas as pd
import numpy as np
import random
import xgboost as xgb
import pickle
import re
import heapq
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
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
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    return s1_df, s23_df

_ABBREV = {
    r'\bpvt\b': 'private', r'\bcorp\b': 'corporation', r'\binc\b': 'incorporated',
    r'\bllc\b': 'limited liability company', r'\bltd\b': 'limited', r'\bco\b': 'company',
    r'\bintl\b': 'international', r'\bstr\b': 'street', r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard', r'\bdr\b': 'drive', r'\brd\b': 'road', r'\bln\b': 'lane',
    r'\bapt\b': 'apartment', r'\bste\b': 'suite', r'\bfl\b': 'floor', r'\bsoln\.?\b': 'solution'
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

def compute_features(df_pairs, s1_df, s23_df):
    print("Merging candidate pairs with text...")
    merged = df_pairs.merge(s1_df, left_on='source1_entity_id', right_on='entity_id', how='inner')
    merged = merged.rename(columns={'business_name': 'name1', 'business_address': 'addr1', 'country': 'country1'})
    merged.drop('entity_id', axis=1, inplace=True)
    merged = merged.merge(s23_df, left_on='candidate_entity_id', right_on='entity_id', how='inner')
    merged = merged.rename(columns={'business_name': 'name2', 'business_address': 'addr2', 'country': 'country2'})
    merged.drop('entity_id', axis=1, inplace=True)
    
    print("Computing features using hyper-optimized vectorization...")
    n1_list = merged['name1'].str.lower().tolist()
    n2_list = merged['name2'].str.lower().tolist()
    a1_list = merged['addr1'].str.lower().tolist()
    a2_list = merged['addr2'].str.lower().tolist()
    c1_list = merged['country1'].tolist()
    c2_list = merged['country2'].tolist()

    features = {
        'name_jw': [JaroWinkler.normalized_similarity(n1, n2) for n1, n2 in zip(n1_list, n2_list)],
        'addr_jw': [JaroWinkler.normalized_similarity(a1, a2) for a1, a2 in zip(a1_list, a2_list)],
        'name_jaccard': [token_jaccard(n1, n2) for n1, n2 in zip(n1_list, n2_list)],
        'addr_jaccard': [token_jaccard(a1, a2) for a1, a2 in zip(a1_list, a2_list)],
        'name_partial': [fuzz.partial_token_set_ratio(n1, n2) / 100.0 for n1, n2 in zip(n1_list, n2_list)],
        'addr_partial': [fuzz.partial_token_set_ratio(a1, a2) / 100.0 for a1, a2 in zip(a1_list, a2_list)],
        'name_sort': [fuzz.token_sort_ratio(n1, n2) / 100.0 for n1, n2 in zip(n1_list, n2_list)],
        'addr_sort': [fuzz.token_sort_ratio(a1, a2) / 100.0 for a1, a2 in zip(a1_list, a2_list)],
        'country_match': [1.0 if c1 == c2 and c1 != '' else 0.0 for c1, c2 in zip(c1_list, c2_list)]
    }
    return pd.DataFrame(features)

def build_training_dataset_hard_mining(candidates_file, positive_pairs, s1_df, s23_df):
    print("Converting DataFrames to lookup dicts for Hard Negative Mining...")
    s1_dict = s1_df.set_index('entity_id').to_dict('index')
    s23_dict = s23_df.set_index('entity_id').to_dict('index')
    
    print("Streaming candidate pairs for Hard Negative Mining...")
    positive_list = []
    heap = []
    tiebreak = 0
    max_heap_size = 3_000_000
    
    candidates = pd.read_csv(candidates_file, sep="\t")
    for _, row in candidates.iterrows():
        s1_id = row['source1_entity_id']
        cands = str(row['candidate_entity_ids']).split(',')
        for cand_id in cands:
            cand_id = cand_id.strip()
            if not cand_id or cand_id == 'nan':
                continue
                
            if (s1_id, cand_id) in positive_pairs:
                positive_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id, 'label': 1})
                continue
            
            s1 = s1_dict.get(s1_id, {})
            s23 = s23_dict.get(cand_id, {})
            name1 = str(s1.get('business_name', '')).lower()
            name2 = str(s23.get('business_name', '')).lower()
            
            A = set(name1.split())
            B = set(name2.split())
            diff = len(A & B) / len(A | B) if (A or B) else 0.0
            
            if diff > 0.05:
                if len(heap) >= max_heap_size:
                    heapq.heappushpop(heap, (diff, tiebreak, {'source1_entity_id': s1_id, 'candidate_entity_id': cand_id, 'label': 0}))
                else:
                    heapq.heappush(heap, (diff, tiebreak, {'source1_entity_id': s1_id, 'candidate_entity_id': cand_id, 'label': 0}))
                tiebreak += 1

    target_neg = len(positive_list) * 5
    heap.sort(reverse=True)
    hard_negatives = [record for _, _, record in heap[:target_neg]]
    
    candidate_set = set((x['source1_entity_id'], x['candidate_entity_id']) for x in positive_list)
    missed_positives = positive_pairs - candidate_set
    for s1_id, p_id in missed_positives:
        positive_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': p_id, 'label': 1})
        
    pair_list = positive_list + hard_negatives
    random.shuffle(pair_list)
    df_pairs = pd.DataFrame(pair_list)
    print(f"Total training pairs: {len(df_pairs)} (Positives: {df_pairs['label'].sum()})")
    return df_pairs

def apply_heuristic_overrides(df_pairs, s1_df, s23_df, model_probs, threshold=0.5):
    print("Applying deterministic business logic overrides...")
    s1_small = s1_df[['entity_id', 'business_name', 'country']]
    s23_small = s23_df[['entity_id', 'business_name', 'country']]
    
    merged = df_pairs.merge(s1_small, left_on='source1_entity_id', right_on='entity_id', how='left')
    merged = merged.merge(s23_small, left_on='candidate_entity_id', right_on='entity_id', how='left')
    merged = merged.rename(columns={
        'business_name_x': 'name1', 'country_x': 'country1',
        'business_name_y': 'name2', 'country_y': 'country2'
    })
    
    pred = (model_probs >= threshold).astype(int)
    
    name1_lower = merged['name1'].str.lower().fillna('')
    name2_lower = merged['name2'].str.lower().fillna('')
    c1 = merged['country1'].str.lower().fillna('')
    c2 = merged['country2'].str.lower().fillna('')
    
    mask1 = (name1_lower == name2_lower) & (name1_lower != '') & (c1 == c2) & (c1 != '')
    pred = np.where(mask1, 1, pred)
    mask2 = (c1 != c2) & (c1 != '') & (c2 != '')
    pred = np.where(mask2, 0, pred)
    mask3 = (name1_lower.str.len() - name2_lower.str.len()).abs() > 50
    pred = np.where(mask3, 0, pred)
    
    return pred

def optimize_threshold(probs, y_true):
    print("Calibrating decision threshold to maximize F0.5 Macro score...")
    best_t = 0.5
    best_f05 = 0.0
    for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        preds = (probs > t).astype(int)
        score = fbeta_score(y_true, preds, beta=0.5, average='macro', zero_division=0)
        if score > best_f05:
            best_f05 = score; best_t = t
    print(f"Optimal Threshold: {best_t} (Training F0.5: {best_f05:.4f})")
    return best_t

def train_model(X, y):
    print("Training XGBoost...")
    pos_count = y.sum()
    neg_count = len(y) - pos_count
    scale_weight = min(neg_count / pos_count, 200.0) if pos_count > 0 else 5.0
    clf = xgb.XGBClassifier(
        tree_method='hist', device='cuda', objective='binary:logistic',
        scale_pos_weight=scale_weight, max_depth=5, gamma=0.5, min_child_weight=5,
        learning_rate=0.03, max_delta_step=5, subsample=0.8, colsample_bytree=0.8,
        n_estimators=300, eval_metric='aucpr', n_jobs=-1
    )
    clf.fit(X, y)
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
        s1_df, s23_df = get_entity_dfs("train", args.data_dir)
        df_pairs = build_training_dataset_hard_mining(args.candidates_file, positive_pairs, s1_df, s23_df)
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
            
        print(f"Predicting matches...")
        probs = model.predict_proba(X)[:, 1]
        
        # Apply Post-Processing Heuristics!
        final_pred = apply_heuristic_overrides(df_pairs, s1_df, s23_df, probs, threshold)
        df_pairs['is_match'] = final_pred
        
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
