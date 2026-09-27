# EntityResolve-ML: Ultimate V2 Kaggle Pipeline Plan

## Goal
Design a production-ready Kaggle notebook that maximizes macro F0.5 using Dual T4 GPUs (32GB VRAM) and 30GB RAM, with ensemble models and deeper text features.

## Hardware Constraints
- **GPU:** 2x T4 (16GB VRAM each)
- **RAM:** 30GB
- **Storage:** Kaggle default (~100GB available)
- **Dataset:** ~14.7M total rows across train/test

---

## V2 Architecture Overview

### Stage 1: Blocking (V2 major enhancement)
- Multi-key inverted index with similarity-based truncation
- Output: `candidate_pairs.tsv` with top-K candidates per S1
- **V2 improvements:** 
  - Similarity-based truncation instead of arbitrary first-N
  - Adaptive K based on block size
  - Two-stage blocking: exact keys -> optional TF-IDF re-ranking
  - Disk caching of inverted index
  - Optional recall validation on ground truth sample
  - Memory-bounded streaming for large blocks
  - Include address prefix keys for better recall

### Stage 2: Feature Engineering (V2 major enhancement)
**Current features:**
- Jaro-Winkler similarity
- Partial token set ratio
- Token sort ratio
- Token Jaccard with abbreviation expansion
- Country match

**New V2 features:**
1. **TF-IDF Cosine Similarity** (char 3-grams, name+address combined)
2. **Character Bigram Dice Coefficient** (name+address)
3. **Character Trigram Jaccard** (name+address)
4. **Word-level TF-IDF Cosine** (name+address)
5. **Length Ratio Features** (name len ratio, address len ratio)
6. **Token Overlap Count** (raw intersection size)
7. **Prefix Match Features** (first 4 chars, first 8 chars)

### Stage 3: Hard Negative Mining (V2 major enhancement)
- Use 30GB RAM to keep **more** hard negatives
- Increase heap size from 3M to **10M**
- Lower difficulty threshold to keep more borderline cases
- Use **two-pass** mining:
  - Pass 1: Quick Jaccard-based difficulty score
  - Pass 2: Re-score top 1M candidates with TF-IDF cosine for finer ranking
- **V2 improvement:** Stream candidates, bounded heap with early skipping, adaptive keep_ratio based on positive rate, missed positive injection before mining, shuffle before splitting

### Stage 4: Model Training (V2 Ensemble)
**Models:**
1. **XGBoost** (GPU-accelerated, T4 #1)
   - `device='cuda:0'`, `tree_method='hist'`
   - Precision-first hyperparameters
   - Early stopping on validation set

2. **LightGBM** (GPU-accelerated, T4 #2)
   - `device='gpu'`, `gpu_platform_id=0`, `gpu_device_id=1`
   - Mirror XGBoost hyperparameters where possible
   - Early stopping on validation set

**Training Strategy:**
- 80/20 train/validation split, stratified by label
- Both models trained on **identical** features and splits
- Parallel training using `ThreadPoolExecutor` (one thread per GPU)
- Validation set used for:
  - Early stopping
  - Threshold calibration
  - Ensemble weight optimization

### Stage 5: Ensemble & Calibration (V2 major enhancement)
**Ensemble Methods:**
1. **Simple Average:** `ensemble_prob = (xgb_prob + lgb_prob) / 2`
2. **Weighted Average:** Optimize weights on validation set
   - `ensemble_prob = w_xgb * xgb_prob + w_lgb * lgb_prob`
   - Where `w_xgb + w_lgb = 1`
3. **Logistic Blending:** Train meta-model on validation probabilities
   - More robust but requires more data
   - **V2 recommendation:** Start with weighted average, try blending if time permits

**Threshold Calibration:**
- Coarse grid: `[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]`
- Fine grid around best coarse threshold: `np.arange(best_t - 0.05, best_t + 0.05, 0.01)`
- Select threshold that maximizes macro F0.5

### Stage 6: Post-Processing Heuristics (V2 enhancement)
**Current heuristics:**
- Exact name match + same country → force match
- Different countries → force no match
- Name length diff > 50 → force no match

**New V2 heuristics:**
1. **TF-IDF cosine > 0.9** + same country → force match
2. **Character bigram Dice > 0.85** + same country → force match
3. **Address exact match** + same country → force match
4. **Both names empty** + different countries → force no match
5. **Name length ratio < 0.3** → force no match
6. **First 8 chars of name match** + same country → force match

### Stage 7: Test-Time Strategies (V2 enhancement)
- **Test-time augmentation:** Apply multiple similarity thresholds and aggregate
- **Candidate re-ranking:** Re-rank test candidates using ensemble scores
- **Singleton handling:** Ensure all S1 entities appear in output, even with empty matches
- **Uncertainty estimation:** Use prediction entropy to identify low-confidence matches

### Stage 8: Prediction & Output
- Batched prediction on test set (50K pairs/batch)
- Apply heuristics to final predictions
- Ensure all S1 entities present in output
- Write `matching_results.tsv` and `candidate_pairs.tsv`

---

## Detailed Implementation

### 1. Blocking V2

```python
import argparse
import os
import csv
import re
import json
import hashlib
from collections import defaultdict
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

csv.field_size_limit(sys.maxsize)

def normalize(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9]', '', text)
    return text

def extract_keys_v2(row):
    """Generate multiple blocking keys for high recall."""
    name = normalize(row.get('business_name', ''))
    addr = normalize(row.get('business_address', ''))
    country = normalize(row.get('country', ''))
    keys = []
    if country and name:
        keys.append(f"{country}_n6_{name[:6]}")
    if country and addr:
        keys.append(f"{country}_a6_{addr[:6]}")
    if country and name and addr:
        keys.append(f"{country}_n3a3_{name[:3]}_{addr[:3]}")
    if name:
        keys.append(f"n4_{name[:4]}")
    if addr:
        keys.append(f"a4_{addr[:4]}")
    return keys

def build_inverted_index_v2(filepaths, index_path=None):
    """Build blocking index with optional disk caching."""
    if index_path and os.path.exists(index_path):
        print(f"Loading cached index from {index_path}")
        with open(index_path, 'r') as f:
            return defaultdict(list, json.load(f))
    
    print(f"Building blocking index from {len(filepaths)} files...")
    index = defaultdict(list)
    for path in filepaths:
        print(f"  Indexing {os.path.basename(path)}...")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eid = row.get('entity_id', '')
                if not eid:
                    continue
                for k in extract_keys_v2(row):
                    index[k].append(eid)
    
    print(f"Total unique blocking keys: {len(index)}")
    
    if index_path:
        print(f"Caching index to {index_path}")
        with open(index_path, 'w') as f:
            json.dump(dict(index), f)
    
    return index

def compute_candidate_score(s1_text, cand_text):
    """Fast similarity score for candidate ranking."""
    if not s1_text or not cand_text:
        return 0.0
    s1_set = set(s1_text[:20].lower().split())
    cand_set = set(cand_text[:20].lower().split())
    if not s1_set and not cand_set:
        return 0.0
    return len(s1_set & cand_set) / len(s1_set | cand_set)

def generate_candidates_v2(data_dir, out_file, split="train", max_candidates=50, 
                           index_path=None, validate_recall=False, gt_file=None):
    """V2 blocking with similarity-based truncation and adaptive K."""
    s1_path = os.path.join(data_dir, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{split}_source3.tsv")
    
    s23_index = build_inverted_index_v2([s2_path, s3_path], index_path)
    
    print("Building S2/S3 lookup for scoring...")
    s23_lookup = {}
    for path in [s2_path, s3_path]:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eid = row.get('entity_id', '')
                if eid:
                    s23_lookup[eid] = {
                        'name': row.get('business_name', ''),
                        'addr': row.get('business_address', ''),
                        'country': row.get('country', '')
                    }
    
    if validate_recall and gt_file:
        print("Recall validation mode enabled")
        positive_pairs = set()
        with open(gt_file, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                s1_id = row['source1_entity_id']
                matched = str(row['matched_entity_ids']).split(',')
                for m in matched:
                    m = m.strip()
                    if m and m != 'nan':
                        positive_pairs.add((s1_id, m))
    
    print("Generating candidate pairs for Source 1...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    stats = {'total_s1': 0, 'with_candidates': 0, 'avg_candidates': 0, 'recall_hits': 0}
    candidate_counts = []
    
    with open(s1_path, 'r', encoding='utf-8') as fin, \
         open(out_file, 'w', encoding='utf-8') as fout:
        
        reader = csv.DictReader(fin, delimiter='\t')
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for row in tqdm(reader):
            s1_id = row.get('entity_id', '')
            if not s1_id:
                continue
            
            stats['total_s1'] += 1
            keys = extract_keys_v2(row)
            candidates = set()
            
            for k in keys:
                candidates.update(s23_index.get(k, []))
            
            candidates.discard(s1_id)
            
            if candidates:
                stats['with_candidates'] += 1
                candidate_counts.append(len(candidates))
            
            if validate_recall and gt_file:
                recall_hits = sum(1 for c in candidates if (s1_id, c) in positive_pairs)
                stats['recall_hits'] += recall_hits
            
            candidates = list(candidates)
            
            if len(candidates) > max_candidates:
                s1_name = row.get('business_name', '')
                s1_addr = row.get('business_address', '')
                s1_country = row.get('country', '')
                
                def candidate_score(cid):
                    c = s23_lookup.get(cid, {})
                    score = compute_candidate_score(s1_name, c.get('name', ''))
                    score += 0.5 * compute_candidate_score(s1_addr, c.get('addr', ''))
                    if s1_country and c.get('country', '') == s1_country:
                        score += 0.1
                    return score
                
                candidates.sort(key=candidate_score, reverse=True)
                candidates = candidates[:max_candidates]
            
            cand_str = ",".join(candidates)
            fout.write(f"{s1_id}\t{cand_str}\n")
    
    print(f"\nBlocking Statistics:")
    print(f"  Total S1 entities: {stats['total_s1']:,}")
    print(f"  S1 with candidates: {stats['with_candidates']:,}")
    if candidate_counts:
        print(f"  Avg candidates/S1: {np.mean(candidate_counts):.1f}")
        print(f"  Median candidates/S1: {np.median(candidate_counts):.1f}")
        print(f"  Max candidates/S1: {max(candidate_counts):,}")
    if validate_recall and gt_file:
        total_positives = len(positive_pairs)
        recall = stats['recall_hits'] / total_positives if total_positives > 0 else 0
        print(f"  Recall on ground truth: {recall:.4f} ({stats['recall_hits']}/{total_positives})")
    
    print(f"Complete! Candidates saved to {out_file}")
    return stats
```

### 2. Two-Stage Blocking with TF-IDF Re-ranking (V2 Enhancement)

```python
def generate_candidates_two_stage(data_dir, out_file, split="train", 
                                  max_candidates=50, index_path=None):
    """V2: Two-stage blocking with exact keys + TF-IDF re-ranking."""
    s1_path = os.path.join(data_dir, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{split}_source3.tsv")
    
    # Stage 1: Exact key blocking
    print("Stage 1: Exact-key blocking...")
    s23_index = build_inverted_index_v2([s2_path, s3_path], index_path)
    
    # Build S2/S3 text lookup
    s23_lookup = {}
    s23_ids = []
    s23_texts = []
    for path in [s2_path, s3_path]:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            for row in reader:
                eid = row.get('entity_id', '')
                if eid:
                    text = f"{row.get('business_name', '')} {row.get('business_address', '')}"
                    s23_lookup[eid] = text
                    s23_ids.append(eid)
                    s23_texts.append(text)
    
    print("Stage 2: Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=50000)
    tfidf.fit(s23_texts)
    s23_matrix = tfidf.transform(s23_texts)
    id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}
    
    print("Stage 3: Generating and re-ranking candidates...")
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    
    with open(s1_path, 'r', encoding='utf-8') as fin, \
         open(out_file, 'w', encoding='utf-8') as fout:
        
        reader = csv.DictReader(fin, delimiter='\t')
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for row in tqdm(reader):
            s1_id = row.get('entity_id', '')
            if not s1_id:
                continue
            
            keys = extract_keys_v2(row)
            candidates = set()
            for k in keys:
                candidates.update(s23_index.get(k, []))
            candidates.discard(s1_id)
            
            if not candidates:
                fout.write(f"{s1_id}\t\n")
                continue
            
            candidates = list(candidates)
            
            if len(candidates) > max_candidates:
                s1_text = f"{row.get('business_name', '')} {row.get('business_address', '')}"
                s1_vec = tfidf.transform([s1_text])
                
                candidate_indices = [id_to_idx.get(c) for c in candidates if c in id_to_idx]
                if candidate_indices:
                    cand_matrix = s23_matrix[candidate_indices]
                    sims = cosine_similarity(s1_vec, cand_matrix).flatten()
                    
                    top_indices = np.argsort(sims)[-max_candidates:][::-1]
                    candidates = [candidates[i] for i in top_indices if i < len(candidates)]
            
            cand_str = ",".join(candidates)
            fout.write(f"{s1_id}\t{cand_str}\n")
    
    print(f"Complete! Candidates saved to {out_file}")
```

---

## Blocking V2 Configuration

| Parameter | V1 Value | V2 Value | Rationale |
|-----------|----------|----------|-----------|
| Keys per entity | 3 | 5 | Add name/address prefix keys without country |
| Truncation | First N by insertion order | Top-K by similarity score | Keep most promising candidates |
| Max candidates | 50 | Adaptive: 30-50 | Tighter blocks for ranking bonus |
| Index caching | No | Yes | Avoid rebuilding on rerun |
| Recall validation | No | Optional | Measure true recall on GT sample |
| Two-stage | No | Optional | TF-IDF re-ranking for better precision |

---

## Blocking V2 Execution Plan

### Option A: Similarity-Truncated Exact Keys (Fast)
```python
stats = generate_candidates_v2(
    data_dir='dataset/train',
    out_file='output/train_candidates.tsv',
    split='train',
    max_candidates=30,
    index_path='cache/train_index.json',
    validate_recall=True,
    gt_file='dataset/train/train_ground_truth.tsv'
)
```
**Expected runtime:** 1-2 hours
**Memory:** ~8GB

### Option B: Two-Stage Blocking (Slower, Higher Quality)
```python
generate_candidates_two_stage(
    data_dir='dataset/train',
    out_file='output/train_candidates.tsv',
    split='train',
    max_candidates=30,
    index_path='cache/train_index.json'
)
```
**Expected runtime:** 3-4 hours
**Memory:** ~15GB

---

## Blocking V2 Fallback

If two-stage blocking is too slow:
1. Fall back to Option A with similarity truncation
2. Reduce `max_candidates` to 20
3. Cache index aggressively to avoid rebuilds

---

## Blocking V2 Validation

After blocking, verify:
- [ ] Average candidates/S1 < 50
- [ ] Recall on ground truth sample > 0.95
- [ ] No empty candidate lists unless S1 truly has no matches
- [ ] Index cache file exists for reuse
- [ ] Runtime < 4 hours on full dataset

```python
def compute_features_v2(df_pairs, s1_df, s23_df, tfidf_vec=None, char_vec=None):
    """V2 feature engineering with TF-IDF and character n-grams."""
    merged = df_pairs.merge(s1_df, left_on='source1_entity_id', right_on='entity_id', how='inner')
    merged = merged.merge(s23_df, left_on='candidate_entity_id', right_on='entity_id', how='inner')
    merged = merged.rename(columns={
        'business_name_x': 'name1', 'business_address_x': 'addr1', 'country_x': 'country1',
        'business_name_y': 'name2', 'business_address_y': 'addr2', 'country_y': 'country2'
    })
    
    features = {}
    
    # Basic features
    features['name_jw'] = [JaroWinkler.normalized_similarity(n1, n2) for n1, n2 in zip(merged['name1'], merged['name2'])]
    features['addr_jw'] = [JaroWinkler.normalized_similarity(a1, a2) for a1, a2 in zip(merged['addr1'], merged['addr2'])]
    features['name_jaccard'] = [token_jaccard(n1, n2) for n1, n2 in zip(merged['name1'], merged['name2'])]
    features['addr_jaccard'] = [token_jaccard(a1, a2) for a1, a2 in zip(merged['addr1'], merged['addr2'])]
    features['name_partial'] = [fuzz.partial_token_set_ratio(n1, n2) / 100.0 for n1, n2 in zip(merged['name1'], merged['name2'])]
    features['addr_partial'] = [fuzz.partial_token_set_ratio(a1, a2) / 100.0 for a1, a2 in zip(merged['addr1'], merged['addr2'])]
    features['name_sort'] = [fuzz.token_sort_ratio(n1, n2) / 100.0 for n1, n2 in zip(merged['name1'], merged['name2'])]
    features['addr_sort'] = [fuzz.token_sort_ratio(a1, a2) / 100.0 for a1, a2 in zip(merged['addr1'], merged['addr2'])]
    features['country_match'] = [1.0 if c1 == c2 and c1 != '' else 0.0 for c1, c2 in zip(merged['country1'], merged['country2'])]
    
    # V2: TF-IDF cosine similarity
    if tfidf_vec is None:
        tfidf_vec = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=50000)
        all_text = pd.concat([
            (merged['name1'].fillna('') + ' ' + merged['addr1'].fillna('')),
            (merged['name2'].fillna('') + ' ' + merged['addr2'].fillna(''))
        ])
        tfidf_vec.fit(all_text)
    
    features['tfidf_cosine'] = compute_tfidf_cosine(merged, tfidf_vec)
    
    # V2: Character bigram Dice
    if char_vec is None:
        char_vec = CountVectorizer(analyzer='char', ngram_range=(2, 2), binary=True)
        all_text = pd.concat([
            (merged['name1'].fillna('') + ' ' + merged['addr1'].fillna('')),
            (merged['name2'].fillna('') + ' ' + merged['addr2'].fillna(''))
        ])
        char_vec.fit(all_text)
    
    features['char_bigram_dice'] = compute_char_bigram_dice(merged, char_vec)
    
    # V2: Character trigram Jaccard
    features['char_trigram_jaccard'] = compute_char_trigram_jaccard(merged)
    
    # V2: Length ratios
    features['name_len_ratio'] = merged['name1'].str.len() / merged['name2'].str.len().replace(0, 1)
    features['addr_len_ratio'] = merged['addr1'].str.len() / merged['addr2'].str.len().replace(0, 1)
    
    # V2: Token overlap count
    features['token_overlap_count'] = compute_token_overlap_count(merged)
    
    # V2: Prefix matches
    features['name_prefix4_match'] = [1.0 if n1[:4] == n2[:4] and n1[:4] != '' else 0.0 for n1, n2 in zip(merged['name1'], merged['name2'])]
    features['name_prefix8_match'] = [1.0 if n1[:8] == n2[:8] and n1[:8] != '' else 0.0 for n1, n2 in zip(merged['name1'], merged['name2'])]
    
    return pd.DataFrame(features)

def compute_tfidf_cosine(merged, vectorizer):
    """Compute TF-IDF cosine similarity for pairs."""
    text1 = (merged['name1'].fillna('') + ' ' + merged['addr1'].fillna('')).tolist()
    text2 = (merged['name2'].fillna('') + ' ' + merged['addr2'].fillna('')).tolist()
    
    X1 = vectorizer.transform(text1)
    X2 = vectorizer.transform(text2)
    
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(X1, X2).diagonal()
    return sims.tolist()

def compute_char_bigram_dice(merged, vectorizer):
    """Compute character bigram Dice coefficient."""
    text1 = (merged['name1'].fillna('') + ' ' + merged['addr1'].fillna('')).tolist()
    text2 = (merged['name2'].fillna('') + ' ' + merged['addr2'].fillna('')).tolist()
    
    X1 = vectorizer.transform(text1)
    X2 = vectorizer.transform(text2)
    
    # Dice = 2 * |A ∩ B| / (|A| + |B|)
    intersection = np.array(X1.multiply(X2).sum(axis=1)).flatten()
    len1 = np.array(X1.sum(axis=1)).flatten()
    len2 = np.array(X2.sum(axis=1)).flatten()
    
    dice = 2 * intersection / (len1 + len2 + 1e-8)
    return dice.tolist()

def compute_char_trigram_jaccard(merged):
    """Compute character trigram Jaccard similarity."""
    def trigrams(s):
        s = str(s).lower().replace(' ', '')
        if len(s) < 3:
            return set()
        return {s[i:i+3] for i in range(len(s)-2)}
    
    jaccards = []
    for n1, n2, a1, a2 in zip(merged['name1'], merged['name2'], merged['addr1'], merged['addr2']):
        t1 = trigrams(n1) | trigrams(a1)
        t2 = trigrams(n2) | trigrams(a2)
        if not t1 and not t2:
            jaccards.append(1.0)
        else:
            jaccards.append(len(t1 & t2) / len(t1 | t2))
    return jaccards

def compute_token_overlap_count(merged):
    """Compute raw token overlap count."""
    counts = []
    for n1, n2, a1, a2 in zip(merged['name1'], merged['name2'], merged['addr1'], merged['addr2']):
        tokens1 = set(expand_abbrev(n1).split()) | set(expand_abbrev(a1).split())
        tokens2 = set(expand_abbrev(n2).split()) | set(expand_abbrev(a2).split())
        counts.append(len(tokens1 & tokens2))
    return counts
```

### 2. Hard Negative Mining V2

```python
def hard_negative_mining_v2(candidates_file, positive_pairs, s1_dict, s23_df, 
                             keep_ratio=0.1, max_heap_size=10_000_000):
    """
    V2: Two-pass hard negative mining with increased memory budget.
    """
    import heapq
    import csv
    
    positive_list = []
    heap = []
    tiebreak = 0
    
    print(f"V2 Hard Negative Mining: keep_ratio={keep_ratio}, max_heap={max_heap_size:,}")
    
    # Pass 1: Quick Jaccard-based difficulty
    with open(candidates_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            s1_id = row['source1_entity_id']
            cand_id = row['candidate_entity_id']
            
            if (s1_id, cand_id) in positive_pairs:
                positive_list.append({'source1_entity_id': s1_id, 'candidate_entity_id': cand_id, 'label': 1})
                continue
            
            s1 = s1_dict.get(s1_id, {})
            s23 = s23_df.get(cand_id, {})
            name1 = str(s1.get('business_name', '')).lower()
            name2 = str(s23.get('business_name', '')).lower()
            addr1 = str(s1.get('business_address', '')).lower()
            addr2 = str(s23.get('business_address', '')).lower()
            c1 = str(s1.get('country', '')).lower()
            c2 = str(s23.get('country', '')).lower()
            
            nj = token_jaccard(name1, name2)
            aj = token_jaccard(addr1, addr2)
            cm = 1.0 if c1 == c2 and c1 != '' else 0.0
            difficulty = 0.5 * nj + 0.3 * aj + 0.2 * cm
            
            # Bounded heap with early skip
            if len(heap) >= max_heap_size and difficulty <= heap[0][0]:
                continue
            heapq.heappush(heap, (difficulty, tiebreak, {
                'source1_entity_id': s1_id, 'candidate_entity_id': cand_id, 'label': 0
            }))
            tiebreak += 1
            if len(heap) > max_heap_size:
                heapq.heappop(heap)
    
    # Pass 2: Re-score top 1M with TF-IDF cosine
    print("Pass 2: Re-scoring top candidates with TF-IDF...")
    top_candidates = [record for _, _, record in heap[:1_000_000]]
    tfidf_scores = compute_tfidf_cosine_for_pairs(top_candidates, s1_dict, s23_df)
    
    for i, record in enumerate(top_candidates):
        record['tfidf_score'] = tfidf_scores[i]
    
    target_neg = int(len(positive_list) * (1.0 / keep_ratio))
    top_candidates.sort(key=lambda x: x['tfidf_score'], reverse=True)
    hard_negatives = [{'source1_entity_id': r['source1_entity_id'], 
                       'candidate_entity_id': r['candidate_entity_id'], 'label': 0} 
                      for r in top_candidates[:target_neg]]
    
    pair_list = positive_list + hard_negatives
    random.shuffle(pair_list)
    
    print(f"Hard negatives: {len(hard_negatives):,} / {len(heap):,} total")
    print(f"Final training pairs: {len(pair_list):,} (Positives: {len(positive_list):,})")
    
    return pd.DataFrame(pair_list)

def compute_tfidf_cosine_for_pairs(pairs, s1_dict, s23_df):
    """Compute TF-IDF cosine for a list of pairs."""
    texts1 = []
    texts2 = []
    for p in pairs:
        s1 = s1_dict.get(p['source1_entity_id'], {})
        s23 = s23_df.get(p['candidate_entity_id'], {})
        t1 = str(s1.get('business_name', '')) + ' ' + str(s1.get('business_address', ''))
        t2 = str(s23.get('business_name', '')) + ' ' + str(s23.get('business_address', ''))
        texts1.append(t1)
        texts2.append(t2)
    
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3), max_features=50000)
    all_text = texts1 + texts2
    vectorizer.fit(all_text)
    
    X1 = vectorizer.transform(texts1)
    X2 = vectorizer.transform(texts2)
    
    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(X1, X2).diagonal()
    return sims.tolist()
```

### 3. Dual GPU Training

```python
def train_ensemble_dual_gpu(X_train, y_train, X_val, y_val):
    """Train XGBoost on T4 #1 and LightGBM on T4 #2 in parallel."""
    import concurrent.futures
    
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = min(neg_count / pos_count, 100.0) if pos_count > 0 else 5.0
    
    def train_xgb():
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            scale_pos_weight=scale_weight,
            max_depth=4,
            gamma=2,
            min_child_weight=10,
            learning_rate=0.05,
            max_delta_step=3,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=300,
            eval_metric='aucpr',
            n_jobs=-1,
            device='cuda:0',
            seed=42
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False
        )
        return model
    
    def train_lgb():
        lgb_params = {
            'objective': 'binary',
            'metric': 'aucpr',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 1,
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'scale_pos_weight': scale_weight,
            'min_child_weight': 10,
            'max_depth': 4,
            'n_estimators': 300,
            'early_stopping_round': 30,
            'verbose': -1,
            'seed': 42
        }
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
        )
        return model
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        xgb_future = executor.submit(train_xgb)
        lgb_future = executor.submit(train_lgb)
        xgb_model = xgb_future.result()
        lgb_model = lgb_future.result()
    
    # Ensemble validation predictions
    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    lgb_val_probs = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
    
    # Optimize ensemble weights and threshold
    best_w, best_t = optimize_ensemble_weights(xgb_val_probs, lgb_val_probs, y_val)
    
    return xgb_model, lgb_model, best_w, best_t, lgb_model.best_iteration

def optimize_ensemble_weights(xgb_probs, lgb_probs, y_true):
    """Find optimal weights for ensemble and threshold."""
    best_f05 = 0.0
    best_w = 0.5
    best_t = 0.5
    
    for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_lgb = 1.0 - w_xgb
        ensemble_probs = w_xgb * xgb_probs + w_lgb * lgb_probs
        
        for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            preds = (ensemble_probs >= t).astype(int)
            f05 = fbeta_score(y_true, preds, beta=0.5, average='macro', zero_division=0)
            if f05 > best_f05:
                best_f05 = f05
                best_w = w_xgb
                best_t = t
    
    print(f"Optimal ensemble: XGB={best_w:.1f}, LGB={1-best_w:.1f}, threshold={best_t}")
    return best_w, best_t
```

### 4. Post-Processing Heuristics V2

```python
def apply_heuristic_overrides_v2(df_pairs, s1_df, s23_df, tfidf_vec, char_vec, 
                                  model_probs, threshold=0.5):
    """V2: Apply deterministic rules with deeper features."""
    merged = df_pairs.merge(s1_df, left_on='source1_entity_id', right_on='entity_id', how='left')
    merged = merged.merge(s23_df, left_on='candidate_entity_id', right_on='entity_id', how='left')
    merged = merged.rename(columns={
        'business_name_x': 'name1', 'business_address_x': 'addr1', 'country_x': 'country1',
        'business_name_y': 'name2', 'business_address_y': 'addr2', 'country_y': 'country2'
    })
    
    pred = (model_probs >= threshold).astype(int)
    
    name1_lower = merged['name1'].str.lower().fillna('')
    name2_lower = merged['name2'].str.lower().fillna('')
    c1 = merged['country1'].str.lower().fillna('')
    c2 = merged['country2'].str.lower().fillna('')
    
    # Rule 1: Exact name match + same country → force match
    mask1 = (name1_lower == name2_lower) & (name1_lower != '') & (c1 == c2) & (c1 != '')
    pred = np.where(mask1, 1, pred)
    
    # Rule 2: Different countries → force no match
    mask2 = (c1 != c2) & (c1 != '') & (c2 != '')
    pred = np.where(mask2, 0, pred)
    
    # Rule 3: Name length diff > 50 → force no match
    len_diff = (name1_lower.str.len() - name2_lower.str.len()).abs()
    mask3 = len_diff > 50
    pred = np.where(mask3, 0, pred)
    
    # V2 Rule 4: TF-IDF cosine > 0.9 + same country → force match
    tfidf_scores = compute_tfidf_cosine(merged, tfidf_vec)
    mask4 = (pd.Series(tfidf_scores) > 0.9) & (c1 == c2) & (c1 != '')
    pred = np.where(mask4, 1, pred)
    
    # V2 Rule 5: Character bigram Dice > 0.85 + same country → force match
    bigram_scores = compute_char_bigram_dice(merged, char_vec)
    mask5 = (pd.Series(bigram_scores) > 0.85) & (c1 == c2) & (c1 != '')
    pred = np.where(mask5, 1, pred)
    
    # V2 Rule 6: Name length ratio < 0.3 → force no match
    name_len_ratio = merged['name1'].str.len() / merged['name2'].str.len().replace(0, 1)
    mask6 = name_len_ratio < 0.3
    pred = np.where(mask6, 0, pred)
    
    return pred
```

---

## Cross-Validation Strategy (V2 Enhancement)

```python
from sklearn.model_selection import StratifiedKFold

def cross_validate_ensemble(X, y, n_splits=5):
    """
    K-fold cross-validation for robust threshold and weight calibration.
    Returns average validation F0.5 and optimal ensemble config.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    ensemble_configs = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"Fold {fold+1}/{n_splits}")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Train ensemble
        xgb_model, lgb_model, w_xgb, best_t, lgb_best_iteration = train_ensemble_dual_gpu(
            X_train, y_train, X_val, y_val
        )
        
        # Validate
        xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
        lgb_val_probs = lgb_model.predict(X_val, num_iteration=lgb_best_iteration)
        ensemble_probs = w_xgb * xgb_val_probs + (1 - w_xgb) * lgb_val_probs
        
        val_preds = (ensemble_probs >= best_t).astype(int)
        fold_f05 = fbeta_score(y_val, val_preds, beta=0.5, average='macro', zero_division=0)
        fold_scores.append(fold_f05)
        ensemble_configs.append({
            'w_xgb': w_xgb,
            'threshold': best_t,
            'fold': fold,
            'f05': fold_f05
        })
        print(f"Fold {fold+1} F0.5: {fold_f05:.4f}")
    
    avg_f05 = np.mean(fold_scores)
    print(f"Average CV F0.5: {avg_f05:.4f} (±{np.std(fold_scores):.4f})")
    
    # Select best config
    best_config = max(ensemble_configs, key=lambda x: x['f05'])
    return best_config, avg_f05
```

---

## Feature Selection (V2 Enhancement)

```python
def select_features(X, y, model, keep_ratio=0.8):
    """
    Select top features based on model importance.
    Reduces noise and improves generalization.
    """
    model.fit(X, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Keep top features
    n_keep = int(len(X.columns) * keep_ratio)
    selected = importance.head(n_keep)['feature'].tolist()
    
    print(f"Selected {len(selected)}/{len(X.columns)} features")
    print("Top 5 features:", selected[:5])
    
    return selected, importance
```

---

## Memory Profiling (V2 Enhancement)

```python
import psutil
import os

def log_memory_usage(stage_name):
    """Log current memory usage."""
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / 1024 / 1024 / 1024
    print(f"[{stage_name}] Memory usage: {mem_gb:.2f} GB")
    return mem_gb

def memory_profile_decorator(func):
    """Decorator to log memory before/after function."""
    def wrapper(*args, **kwargs):
        log_memory_usage(f"Before {func.__name__}")
        result = func(*args, **kwargs)
        log_memory_usage(f"After {func.__name__}")
        return result
    return wrapper
```

---

## Kaggle-Specific Optimizations

### 1. Data Loading
```python
# Use Kaggle's direct paths
DATA_DIR = '/kaggle/input/entity-resolve-ml-2026'
WORKING_DIR = '/kaggle/working'

# Cache intermediate results
CACHE_DIR = os.path.join(WORKING_DIR, 'cache')
os.makedirs(CACHE_DIR, exist_ok=True)
```

### 2. TF-IDF Caching
```python
# Save fitted vectorizers to disk
tfidf_path = os.path.join(CACHE_DIR, 'tfidf_vectorizer.pkl')
char_vec_path = os.path.join(CACHE_DIR, 'char_vectorizer.pkl')

if os.path.exists(tfidf_path):
    with open(tfidf_path, 'rb') as f:
        tfidf_vec = pickle.load(f)
else:
    tfidf_vec = TfidfVectorizer(...)
    tfidf_vec.fit(...)
    with open(tfidf_path, 'wb') as f:
        pickle.dump(tfidf_vec, f)
```

### 3. Feature Caching
```python
# Save computed features to avoid recomputation
features_path = os.path.join(CACHE_DIR, 'train_features.parquet')
if os.path.exists(features_path):
    X = pd.read_parquet(features_path)
else:
    X = compute_features_v2(...)
    X.to_parquet(features_path)
```

### 4. GPU Memory Management
```python
import torch

def clear_gpu_memory():
    """Clear GPU cache between stages."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()
```

---

## Error Handling & Robustness

```python
class PipelineError(Exception):
    """Custom pipeline error."""
    pass

def safe_stage(stage_name, func, *args, **kwargs):
    """
    Execute pipeline stage with error handling and memory cleanup.
    """
    try:
        print(f"\n{'='*60}")
        print(f"Stage: {stage_name}")
        print(f"{'='*60}")
        
        result = func(*args, **kwargs)
        
        # Force garbage collection after each stage
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ {stage_name} completed successfully")
        return result
        
    except Exception as e:
        print(f"✗ {stage_name} failed: {e}")
        raise PipelineError(f"Stage '{stage_name}' failed: {e}") from e
```

---

## Submission Checklist

### Output Files
- [ ] `output/matching_results.tsv` - Final matches
- [ ] `output/candidate_pairs.tsv` - Blocking candidates
- [ ] `output/xgb_model.json` - XGBoost model
- [ ] `output/lgb_model.txt` - LightGBM model
- [ ] `output/ensemble_config.json` - Ensemble weights and threshold
- [ ] `Documentation_template.md` - Filled documentation

### Validation
- [ ] `validate_submission.py --check-ids` exits 0
- [ ] All S1 entities present in output
- [ ] All matched IDs start with `S2-` or `S3-`
- [ ] No duplicate IDs within lists
- [ ] No self-matches (S1- IDs)

### Blocking Quality
- [ ] Average candidates/S1 < 50
- [ ] Recall on ground truth > 0.95
- [ ] No empty candidate lists for entities with matches
- [ ] Index cache file exists
- [ ] Blocking runtime < 4 hours

### Code Quality
- [ ] All functions have docstrings
- [ ] Random seeds set for reproducibility
- [ ] Error handling in place
- [ ] Memory profiling logged

---

## Rollback Plan

If V2 fails during execution:

1. **Immediate Rollback:** Use V1 pipeline with single XGBoost
2. **Partial Rollback:** Use V2 features but single XGBoost model
3. **Minimal Viable:** Use V1 features + V2 hard negative mining

Always keep V1 outputs as fallback.

---

## Success Criteria

| Metric | Target | Measurement |
|--------|--------|-------------|
| Blocking recall | > 0.95 | Ground truth sample |
| Blocking avg candidates/S1 | < 50 | Candidate file stats |
| Validation F0.5 | > 0.98 | Cross-validation average |
| Training time | < 2 hours | Wall clock |
| Prediction time | < 1 hour | Wall clock |
| Peak RAM | < 28GB | Memory profiling |
| GPU utilization | > 80% | nvidia-smi |
| Validation pass | Yes | validator script |

---

## Final Notes

1. **Start with V1 baseline** to ensure pipeline works end-to-end
2. **Run V2 blocking first** - candidate quality limits the ensemble ceiling
3. **Validate blocking recall** on a ground truth sample before training
4. **Add V2 features incrementally** and measure F0.5 impact
5. **Profile memory at each stage** to catch OOM early
6. **Save all intermediate results** to disk for quick recovery
7. **Document everything** in the notebook for reproducibility

The V2 pipeline is designed to be **modular** - each stage can be enabled/disabled independently for rapid experimentation.
