import argparse
import os
import csv
import re
from collections import defaultdict
from tqdm import tqdm

csv.field_size_limit(131072 * 10)

def normalize(text):
    if not text: return ""
    text = str(text).lower()
    # Remove all non-alphanumeric
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def extract_keys(row):
    """Generate multiple blocking keys to ensure high recall despite typos."""
    name = normalize(row.get('business_name', ''))
    addr = normalize(row.get('business_address', ''))
    country = normalize(row.get('country', ''))
    
    keys = []
    
    # Key 1: Country + First 6 chars of Name
    if name:
        keys.append(f"{country}_n6_{name[:6]}")
        
    # Key 2: Country + First 6 chars of Address
    if addr:
        keys.append(f"{country}_a6_{addr[:6]}")
        
    # Key 3: Country + First 3 of Name + First 3 of Address
    if name and addr:
        keys.append(f"{country}_n3a3_{name[:3]}_{addr[:3]}")
        
    return keys

def build_inverted_index(filepaths):
    print(f"Building blocking index from {len(filepaths)} files...")
    index = defaultdict(list)
    
    for path in filepaths:
        print(f"  Indexing {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eid = row.get('entity_id', '')
                if not eid: continue
                
                keys = extract_keys(row)
                for k in keys:
                    index[k].append(eid)
                    
    print(f"Total unique blocking keys created: {len(index)}")
    return index

def generate_candidates(data_dir, out_file, split="train", max_candidates=50):
    s1_path = os.path.join(data_dir, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{split}_source3.tsv")
    
    # 1. Build Index on Source 2 and 3
    s23_index = build_inverted_index([s2_path, s3_path])
    
    print("Generating candidate pairs for Source 1...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with open(s1_path, 'r', encoding='utf-8') as fin, \
         open(out_file, 'w', encoding='utf-8') as fout:
         
        reader = csv.DictReader(fin, delimiter='\t')
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for row in tqdm(reader):
            s1_id = row.get('entity_id', '')
            if not s1_id: continue
            
            keys = extract_keys(row)
            candidates = set()
            
            for k in keys:
                # Add all S2/S3 entities that share this key
                candidates.update(s23_index.get(k, []))
                
            # Convert to list
            candidates = list(candidates)
            
            # If a block is wildly generic (e.g. thousands of matches), we truncate to max_candidates 
            # to prevent the ML model from choking later. (Sorted by default order)
            if len(candidates) > max_candidates:
                candidates = candidates[:max_candidates]
                
            cand_str = ",".join(candidates)
            fout.write(f"{s1_id}\t{cand_str}\n")
            
    print(f"Complete! Candidates saved to {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Key Blocking")
    parser.add_argument("--data_dir", type=str, default="../../dataset/train", help="Path to TSVs")
    parser.add_argument("--out_file", type=str, default="../../output/candidate_pairs.tsv", help="Output path")
    parser.add_argument("--max_candidates", type=int, default=50, help="Max candidates per S1 entity")
    
    args = parser.parse_args()
    split = "test" if "test" in args.data_dir else "train"
    generate_candidates(args.data_dir, args.out_file, split=split, max_candidates=args.max_candidates)
