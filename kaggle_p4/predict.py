
import os
os.system("pip install rapidfuzz")

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import re
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
import time

KAGGLE_DATA_DIR = "/kaggle/input/datasets/anubhavaanand/amazon-ml-challenge/cp/dataset/test"
CANDIDATES_FILE = "/kaggle/input/datasets/anubhavaanand/amazon-ml-challenge/cp/output/test_candidates.tsv"
MODEL_PATH = "/kaggle/input/datasets/anubhavaanand/amazon-ml-xgb-model-file/xgb_model.pkl"
OUT_FILE = "matching_results.tsv"

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
    merged = df_pairs.merge(s1_df, left_on='source1_entity_id', right_on='entity_id', how='inner')
    merged = merged.rename(columns={'business_name': 'name1', 'business_address': 'addr1', 'country': 'country1'})
    merged.drop('entity_id', axis=1, inplace=True)
    merged = merged.merge(s23_df, left_on='candidate_entity_id', right_on='entity_id', how='inner')
    merged = merged.rename(columns={'business_name': 'name2', 'business_address': 'addr2', 'country': 'country2'})
    merged.drop('entity_id', axis=1, inplace=True)
    
    n1_list = merged['name1'].str.lower().fillna('').tolist()
    n2_list = merged['name2'].str.lower().fillna('').tolist()
    a1_list = merged['addr1'].str.lower().fillna('').tolist()
    a2_list = merged['addr2'].str.lower().fillna('').tolist()
    c1_list = merged['country1'].fillna('').tolist()
    c2_list = merged['country2'].fillna('').tolist()

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

def apply_heuristic_overrides(df_pairs, s1_df, s23_df, model_probs, threshold=0.5):
    s1_small = s1_df[['entity_id', 'business_name', 'country']]
    s23_small = s23_df[['entity_id', 'business_name', 'country']]
    merged = df_pairs.merge(s1_small, left_on='source1_entity_id', right_on='entity_id', how='left')
    merged = merged.merge(s23_small, left_on='candidate_entity_id', right_on='entity_id', how='left')
    
    pred = (model_probs >= threshold).astype(int)
    name1_lower = merged['business_name_x'].str.lower().fillna('')
    name2_lower = merged['business_name_y'].str.lower().fillna('')
    c1 = merged['country_x'].str.lower().fillna('')
    c2 = merged['country_y'].str.lower().fillna('')
    
    mask1 = (name1_lower == name2_lower) & (name1_lower != '') & (c1 == c2) & (c1 != '')
    pred = np.where(mask1, 1, pred)
    mask2 = (c1 != c2) & (c1 != '') & (c2 != '')
    pred = np.where(mask2, 0, pred)
    mask3 = (name1_lower.str.len() - name2_lower.str.len()).abs() > 50
    pred = np.where(mask3, 0, pred)
    return pred

try:
    with open(MODEL_PATH, 'rb') as f:
        data = pickle.load(f)
        model = data['model']
        threshold = data['threshold']
    model.set_params(device="cpu")
except Exception as e:
    print("Could not load model.", e)
    import sys; sys.exit(1)

print("Loading test dataset slices...")
t0 = time.time()
if 60000000 == 0:
    test_candidates = pd.read_csv(CANDIDATES_FILE, sep="\t", nrows=20000000)
else:
    test_candidates = pd.read_csv(CANDIDATES_FILE, sep="\t", skiprows=range(1, 60000000+1), nrows=20000000, header=0)
    # The header is automatically parsed, so skipping range(1, START+1) perfectly keeps the header and skips START rows.
    # Wait, read_csv with header=0 means row 0 is header. skiprows skips rows in the original file.
    # We should just pass names=cols if we skip the header, but skiprows=range(1, 2) keeps row 0 (header) and skips row 1!
    # Yes, skiprows=range(1, START+1) skips data rows 1 to START.

print(f"Loaded {len(test_candidates)} candidates in {time.time()-t0:.1f}s")
cols = ['entity_id', 'business_name', 'business_address', 'country']
s1_df = pd.read_csv(os.path.join(KAGGLE_DATA_DIR, "test_source1.tsv"), sep='\t', usecols=cols, dtype='string').fillna('')
s2_df = pd.read_csv(os.path.join(KAGGLE_DATA_DIR, "test_source2.tsv"), sep='\t', usecols=cols, dtype='string').fillna('')
s3_df = pd.read_csv(os.path.join(KAGGLE_DATA_DIR, "test_source3.tsv"), sep='\t', usecols=cols, dtype='string').fillna('')
s23_df = pd.concat([s2_df, s3_df], ignore_index=True)

match_dict = {}
pair_list = []
chunk_size = 5_000_000
total_processed = 0

def process_chunk(pairs):
    df_pairs = pd.DataFrame(pairs)
    X = compute_features(df_pairs, s1_df, s23_df)
    probs = model.predict_proba(X)[:, 1]
    final_pred = apply_heuristic_overrides(df_pairs, s1_df, s23_df, probs, threshold)
    df_pairs['is_match'] = final_pred
    matches = df_pairs[df_pairs['is_match'] == 1]
    for _, row in matches.iterrows():
        s1 = row['source1_entity_id']
        cand = row['candidate_entity_id']
        if s1 not in match_dict:
            match_dict[s1] = []
        match_dict[s1].append(cand)

for _, row in test_candidates.iterrows():
    s1_id = row['source1_entity_id']
    cands = str(row['candidate_entity_ids']).split(',')
    for c in cands:
        if c and c.strip() != 'nan':
            pair_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': c.strip()})
            
    if len(pair_list) >= chunk_size:
        total_processed += len(pair_list)
        print(f"Processed {total_processed} pairs...")
        process_chunk(pair_list)
        pair_list = []
        
if len(pair_list) > 0:
    total_processed += len(pair_list)
    print(f"Processed {total_processed} pairs...")
    process_chunk(pair_list)

with open(OUT_FILE, 'w') as f:
    # Do not write header for parts 2, 3, 4 so we can just concatenate them easily!
    if 60000000 == 0:
        f.write("source1_entity_id\tmatched_entity_ids\n")
    for _, row in test_candidates.iterrows():
        s1_id = row['source1_entity_id']
        m = match_dict.get(s1_id, [])
        f.write(f"{s1_id}\t{','.join(m)}\n")
print("Finished!")
