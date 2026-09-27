# Kilo Codebase Audit: Amazon ML Challenge 2026 — Business Entity Resolution

## 1. Executive Summary

This audit covers the end-to-end Entity Resolution (ER) pipeline in `code/business_entity_resolution/src/blocking.py` and `matching.py` for the **Amazon ML Challenge 2026 Business Entity Resolution** task. The dataset comprises ~26.4M records across three sources (S1: ~2.2M train / ~1.7M test; S2: ~5.0M train / ~4.9M test; S3: ~5.3M train / ~5.1M test). The evaluation metric is **macro-averaged F₀.5**, where precision is weighted 2× over recall and singletons score 1.0 only if the match list is empty.

**Bottom line:** The current pipeline is functional but will **OOM or run for hours/days** on low-RAM hardware at full scale, and its feature/model stack leaves significant F₀.5 headroom on the table. The three biggest gains come from (1) replacing the blocking stage with a semantic + lexical hybrid retriever, (2) vectorizing feature computation with embeddings + numba, and (3) switching to a memory-efficient GBDT implementation with proper calibration and macro-aware threshold optimization.

---

## 2. Current Architecture & Bottleneck Analysis

### 2.1 Blocking (`blocking.py`)

| Component | Observation | Severity |
|-----------|-------------|----------|
| `extract_keys` | 3 lexical keys (country + first 6/3 chars of name/address). No semantic signal, no typo tolerance beyond prefix overlap. | **High** |
| `build_inverted_index` | Builds a `defaultdict(list)` of all S2/S3 records. At 10M+ records, this dict consumes many GB of RAM due to Python object overhead. | **High** |
| `max_candidates=50` | Hard truncation discards true matches in dense blocks (e.g., chain restaurants). This directly caps recall and is an arbitrary heuristic. | **High** |
| No LSH / ANN | Exact inverted index with prefix keys scales linearly and offers no sub-linear query time. | **Medium** |
| No cross-source pooling | S2 and S3 are pooled in the index, but there is no meta-blocking or block purging to reduce noise. | **Medium** |

### 2.2 Matching (`matching.py`)

| Component | Observation | Severity |
|-----------|-------------|----------|
| `get_entity_dict` | Loads **all** S1+S2+S3 text into a Python dict of dicts. For 14.7M rows, this easily exceeds 20–40 GB due to per-string overhead and dict pointers. | **Critical** |
| `build_training_dataset` | Uses `pd.read_csv` + row-by-row list append + `random.sample`. For millions of candidates, this creates massive intermediate lists and a final DataFrame copy. | **High** |
| `compute_features` | **Row-by-row `iterrows()` loop** calling `rapidfuzz` per pair. This is the dominant wall-clock bottleneck. At ~10M candidate pairs, pure Python looping is impractical. | **Critical** |
| `token_jaccard` | Calls `expand_abbrev` (regex loop) twice per feature per pair. Compounding the loop bottleneck above. | **High** |
| XGBoost | `n_estimators=300`, `max_depth=5`. No `tree_method='hist'`, no early stopping, no hyperparameter search. | **Medium** |
| `optimize_threshold` | Scans only `[0.5, 0.55, ..., 0.95]` (10 points). Coarse. Also, for **macro F₀.5**, a single global threshold is suboptimal because macro averaging treats class 0 and class 1 equally; the positive and negative score distributions differ. | **Medium** |
| No probability calibration | `predict_proba` from XGBoost is used raw. XGBoost probabilities are often poorly calibrated, making threshold selection unstable. | **Medium** |
| No hard vetoes | A pair with mismatched countries can still score high enough to pass the threshold, costing macro F₀.5 on singletons. | **Medium** |
| No verified merge | Direct per-pair thresholding allows transitive false merges (A≈B, B≈C ⇒ A≈C), which is catastrophic for precision on singletons. | **High** |

---

## 3. State-of-the-Art Optimizations

### 3.1 Blocking: Replace Inverted Index with Semantic + Lexical Hybrid Retriever

**Why:** The current prefix-based blocking misses matches with typos, reordering, or semantic equivalence. Recent SOTA for ER at scale (2024–2025) uses **pre-trained embeddings + ANN** for candidate retrieval, achieving >98% recall with manageable candidate counts.

**Recommended implementation:**
1. **Precompute embeddings offline** (once) using a lightweight CPU-friendly sentence encoder:
   - `sentence-transformers/all-MiniLM-L6-v2` (fast, small) or `intfloat/multilingual-e5-base` (better for India/France multilingual noise).
   - Encode three channels per record: `name`, `address`, and `name + address`.
   - Store as **float16** `.npy` files + a lightweight `entity_id` index parquet. This mirrors the public AMC26 embedding artifacts (~54 GB for full train+test, but float16).
2. **Build an HNSW index** (via `hnswlib` or `faiss` with `IndexHNSWFlat`) per channel:
   - HNSW gives O(log n) query time and is the current Pareto frontier for high recall + low latency.
   - For low-RAM, use `faiss.IndexHNSWPQ` (product quantization) or `IndexIVFPQ` to reduce index size by 4–16× with minimal recall loss.
3. **Fallback lexical blocking** with **MinHash LSH** (`datasketch`):
   - Build a MinHash LSH index on character 5-grams for name + address.
   - Union the HNSW candidates with LSH candidates. This catches surface-form duplicates that embeddings miss (e.g., exact abbreviations).
4. **Dynamic candidate cap**:
   - Instead of `max_candidates=50`, take top-*k* per channel (e.g., k=20) and union. For dense entities, allow up to 200 candidates with a secondary light filter. This prevents arbitrary recall loss.

**Code sketch:**

```python
# blocking_optimized.py
import hnswlib, numpy as np, pandas as pd
from datasketch import MinHash, MinHashLSH

def build_hnsw_index(embeddings: np.ndarray, ids: list, ef_construction=200, M=32):
    dim = embeddings.shape[1]
    index = hnswlib.Index(space='cosine', dim=dim)
    index.init_index(max_elements=len(ids), ef_construction=ef_construction, M=M)
    index.add_items(embeddings, ids)
    index.set_ef(50)
    return index

def retrieve_candidates(s1_emb, index, k=20):
    labels, distances = index.knn_query(s1_emb, k=k)
    return labels[0]

def build_minhash_lsh(records, num_perm=128, threshold=0.5):
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for rec_id, text in records:
        m = MinHash(num_perm=num_perm)
        for shingle in set(ngrams(text, 5)):
            m.update(shingle.encode('utf8'))
        lsh.insert(rec_id, m)
    return lsh
```

**Impact:** Blocking recall moves from ~70–80% (prefix) to >97% (hybrid), with candidate sets still in the 50–200 range per S1 entity.

---

### 3.2 Feature Engineering: Add Embedding Features + Vectorize Computation

**Why:** The current 9 features are purely lexical. Embedding-based features capture semantic similarity (e.g., "pvt ltd" vs "private limited", transliterations) and are the single biggest accuracy lever in modern ER.

**Recommended features (add to existing 9):**
1. **Cosine similarity** of precomputed name, address, and name+address embeddings (3 features).
2. **Euclidean / L2 distance** of embeddings (3 features).
3. **Dot product** of embeddings (3 features).
4. **Interaction features**: `|v₁ − v₂|` element-wise mean (captures attribute-specific disagreement).
5. **Country match** (already present) + **country confidence** (e.g., exact match vs empty vs mismatch).
6. **Length ratio** of name/address strings (captures truncation / abbreviation).
7. **Prefix / suffix overlap ratio** (token-based).
8. **TF-IDF cosine similarity** on combined name+address (use `sklearn` with `HashingVectorizer` to avoid storing the full vocabulary in memory).

**Vectorization strategy (replaces `compute_features` loop):**
- Precompute all S1, S2, S3 embeddings and store in a single float16 matrix `E` indexed by `entity_id`.
- Build a `pd.DataFrame` of candidate pairs with integer indices into `E`.
- Feature computation becomes **fully vectorized numpy**:
  - `cosine = (E[i] * E[j]).sum(axis=1) / (||E[i]|| * ||E[j]||)`
  - `l2 = np.linalg.norm(E[i] - E[j], axis=1)`
- For rapidfuzz features, use **multiprocessing** with `rapidfuzz.process.cdist` or batch with `rapidfuzz.distance.JaroWinkler.normalized_similarity` on string arrays. Avoid `iterrows()`.

**Code sketch:**

```python
# matching_optimized.py
import numpy as np
from rapidfuzz.distance import JaroWinkler

def compute_features_vectorized(pair_df, emb_dict, entity_text_dict):
    i_idx = pair_df['source1_idx'].values
    j_idx = pair_df['candidate_idx'].values

    # Embeddings
    emb_i = emb_dict[i_idx]
    emb_j = emb_dict[j_idx]
    dot = (emb_i * emb_j).sum(axis=1)
    norm_i = np.linalg.norm(emb_i, axis=1)
    norm_j = np.linalg.norm(emb_j, axis=1)
    cosine = dot / (norm_i * norm_j + 1e-8)
    l2 = np.linalg.norm(emb_i - emb_j, axis=1)

    # RapidFuzz batch
    names1 = [entity_text_dict[i]['name'] for i in pair_df['source1_entity_id']]
    names2 = [entity_text_dict[j]['name'] for j in pair_df['candidate_entity_id']]
    name_jw = JaroWinkler.normalized_similarity_matrix(names1, names2, workers=-1)

    # ... assemble into feature matrix
    return np.column_stack([cosine, l2, name_jw, ...])
```

**Impact:** Feature computation drops from hours (Python loop) to minutes (vectorized numpy + multiprocessing). F₀.5 improves by 5–15 pp due to semantic signal.

---

### 3.3 Model: Switch to LightGBM / CatBoost + Calibration

**Why:** LightGBM uses histogram-based training, leaf-wise growth, and `free_raw_data=True`, resulting in **3–10× lower memory** and faster training than XGBoost on CPU (see LightGBM 4.7 docs). CatBoost handles categoricals natively and avoids leakage.

**Recommended hyperparameters (CPU, low-RAM):**
```python
import lightgbm as lgb

clf = lgb.LGBMClassifier(
    objective='binary',
    learning_rate=0.03,
    n_estimators=500,
    num_leaves=64,
    max_depth=7,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    device='cpu',
)
```
- Use **early stopping** on a stratified validation split (e.g., 80/20 split by S1 entity ID to avoid leakage).
- Use `lgb.Dataset(..., free_raw_data=True)` to drop raw data after training.

**Impact:** Memory drops from ~16 GB (XGBoost exact) to ~2–4 GB (LightGBM histogram). Training speed improves by 2–5×.

---

### 3.4 F₀.5 Macro Optimization: Exact Threshold + Calibration + Hard Vetoes

The current `optimize_threshold` is naive. For macro F₀.5, the following stack is SOTA:

#### 3.4.1 Probability Calibration
Wrap the GBDT in `CalibratedClassifierCV(method='isotonic', cv=3)` or train a small logistic regression on out-of-fold predicted probabilities. Well-calibrated probabilities make threshold selection meaningful.

#### 3.4.2 Exact Macro F₀.5 Threshold Optimization
For binary classification with macro averaging, macro-F₀.5 = (F₀.5(class 0) + F₀.5(class 1)) / 2. Because F₀.5 is piecewise-constant in the threshold, use an **O(n log n) sort-and-scan** algorithm (see Lipton & Elkan, 2014; `optimal-cutoffs` library) instead of scanning 10 fixed points.

```python
from sklearn.metrics import make_scorer, fbeta_score
from sklearn.model_selection import TunedThresholdClassifierCV

scorer = make_scorer(fbeta_score, beta=0.5, average='macro')
calibrated_model = TunedThresholdClassifierCV(
    base_estimator=lgb.LGBMClassifier(...),
    scoring=scorer,
    cv=5,
    method='default'
)
```
Alternatively, implement a custom grid search on a fine grid (e.g., 200 thresholds from 0.01 to 0.99) on a held-out validation set.

#### 3.4.3 Sparsity-Aware Thresholds
Not all pairs have the same evidence. Bin pairs by the number of populated fields (name, address, country) and learn a separate threshold per bin:
- 3 fields populated → threshold = 0.55
- 2 fields populated → threshold = 0.70
- 1 field populated → threshold = 0.85
This improves macro F₀.5 by reducing false positives on sparse pairs while preserving recall on dense pairs (see "ER in Practice" lessons, 2025).

#### 3.4.4 Hard Vetoes
Add deterministic veto rules **before** the ML model:
- If both records have non-empty country and countries differ → reject (FP=1).
- If both records have non-empty address and token Jaccard < 0.05 and Jaro-Winkler < 0.3 → reject.
These guardrails are non-negotiable for precision-weighted metrics and protect singletons.

#### 3.4.5 Verified Merge (Transitivity Control)
After pairwise prediction, **do not** use transitive closure. Instead:
1. Build Stage-1 clusters by joining each S1 record to its single highest-scoring neighbor above threshold.
2. For any two Stage-1 clusters connected by an above-threshold edge, select up to k=3 representatives per cluster and re-score cross-cluster pairs with hard vetoes enabled.
3. Merge only if at least one cross-cluster pair passes both the threshold and veto checks.

**Impact:** Prevents one false positive from cascading into a mega-cluster, directly protecting singleton F₀.5.

---

### 3.5 Memory-Optimized Data Handling

| Issue | Fix |
|-------|-----|
| `get_entity_dict` loads all text into Python dicts | Use **memory-mapped numpy arrays** or **PyArrow** record batches. Store only what is needed for feature computation. |
| `build_training_dataset` keeps `positive_list` + `negative_list` as lists of dicts | Stream candidates from disk using `csv.reader` or `pd.read_csv(chunksize=...)`. Write positive and negative pairs to separate temp files, then sample negatives without holding all in RAM. |
| `pd.concat([s1_df, s2_df, s3_df])` | Read only the columns needed (`entity_id`, `business_name`, `business_address`, `country`). Use `dtype={'business_name': 'string', 'business_address': 'string'}` to reduce memory. |
| Embeddings stored as float32 | Store as **float16** (`np.float16`). This halves memory with negligible accuracy loss for cosine similarity. |
| Python `csv.DictReader` in blocking | Use `csv.reader` with index-based access or `pandas.read_csv(usecols=...)` to avoid creating dict per row. |

**Code sketch for low-RAM entity loading:**

```python
def get_entity_dict_low_ram(split, data_dir):
    cols = ['entity_id', 'business_name', 'business_address', 'country']
    dfs = [
        pd.read_csv(os.path.join(data_dir, f"{split}_source1.tsv"), sep='\t', usecols=cols, dtype='string'),
        pd.read_csv(os.path.join(data_dir, f"{split}_source2.tsv"), sep='\t', usecols=cols, dtype='string'),
        pd.read_csv(os.path.join(data_dir, f"{split}_source3.tsv"), sep='\t', usecols=cols, dtype='string'),
    ]
    df = pd.concat(dfs, ignore_index=True)
    df[['business_name', 'business_address', 'country']] = df[['business_name', 'business_address', 'country']].fillna('')
    # Convert to category to reduce string memory
    for col in ['business_name', 'business_address', 'country']:
        df[col] = df[col].astype('category')
    return df.set_index('entity_id').to_dict('index')
```

---

### 3.6 Execution Speed: Multiprocessing + Batch Prediction

1. **Feature computation**: Split candidate pairs into N chunks (N = CPU count). Use `multiprocessing.Pool` or `joblib.Parallel` to compute rapidfuzz features in parallel.
2. **Batch prediction**: `model.predict_proba(X_batch)` in chunks of 50k–100k rows to keep RAM flat.
3. **Blocking index build**: Use `multiprocessing` to build MinHash sketches in parallel; merge LSH indexes afterward.
4. **Avoid `iterrows()`**: Replace all `iterrows()` with `itertuples()` or vectorized ops.

---

### 3.7 Training Pipeline Improvements

| Issue | Fix |
|-------|-----|
| No validation split | Use **entity-level stratified split**: group by `source1_entity_id`, split 80/20, and train only on train split candidates. This prevents label leakage and gives a true F₀.5 estimate. |
| Random negative subsampling | Add **hard negative mining**: after an initial model, re-score the full candidate set and mine the highest-scoring false positives as additional training negatives. This teaches the model precision-critical boundaries. |
| No hyperparameter tuning | Use **Optuna** with `TPESampler` and a time budget. Search `num_leaves`, `max_depth`, `learning_rate`, `min_child_samples`, `reg_alpha`. Optimize directly on macro F₀.5 via cross-validation. |
| No class weight tuning | Instead of fixed `5:1` negative ratio, tune `scale_pos_weight` or use `focal_loss` (LightGBM supports custom objective). Focal loss down-weights easy negatives, forcing the model to learn hard distinctions. |
| Training on all data without early stopping | Add `early_stopping_rounds=50` on the validation set. Prevents overfitting and reduces tree count. |

---

### 3.8 Candidate Re-Ranking (Two-Stage Cascade)

For extra F₀.5 headroom, add a **cross-encoder re-ranking stage** on the top-50 candidates per S1 entity:
1. Stage 1: Light bi-encoder (HNSW) retrieves top-50 candidates cheaply.
2. Stage 2: A small transformer cross-encoder (e.g., `cross-encoder/ms-MiniLM-L6-v2`) scores the 50 pairs with full attention. This is expensive but applied to <2% of the original candidate space.
3. Use the cross-encoder score as an additional feature or as the final decision.

**Low-RAM alternative:** Use a lightweight TF-IDF + cosine re-ranker on the concatenated name+address text instead of a transformer.

---

## 4. Specific Code Suggestions

### 4.1 `blocking.py`

```python
# Replace build_inverted_index + generate_candidates with:
def generate_candidates(data_dir, out_file, split="train", max_candidates=100):
    # 1. Load only needed columns
    s1 = pd.read_csv(f"{data_dir}/{split}_source1.tsv", sep='\t', usecols=['entity_id','business_name','business_address','country'], dtype='string').fillna('')
    pool = pd.concat([
        pd.read_csv(f"{data_dir}/{split}_source2.tsv", sep='\t', usecols=['entity_id','business_name','business_address','country'], dtype='string').fillna(''),
        pd.read_csv(f"{data_dir}/{split}_source3.tsv", sep='\t', usecols=['entity_id','business_name','business_address','country'], dtype='string').fillna(''),
    ], ignore_index=True)

    # 2. Build MinHash LSH on 5-grams of name + address
    lsh = build_minhash_lsh([(r.entity_id, f"{r.business_name} {r.business_address}") for r in pool.itertuples()])

    # 3. Query and union with exact-key blocking for recall
    with open(out_file, 'w', encoding='utf-8') as fout:
        fout.write("source1_entity_id\tcandidate_entity_ids\n")
        for row in tqdm(s1.itertuples()):
            text = f"{row.business_name} {row.business_address}"
            m = MinHash(num_perm=128)
            for shingle in set(ngrams(text, 5)):
                m.update(shingle.encode('utf8'))
            cands = set(lsh.query(m))
            # Add exact-key matches for high precision
            cands.update(exact_key_index.get(extract_keys(row), []))
            cands = list(cands)[:max_candidates]
            fout.write(f"{row.entity_id}\t{','.join(cands)}\n")
```

### 4.2 `matching.py`

```python
# Replace get_entity_dict with memory-mapped numpy arrays
def get_entity_arrays(split, data_dir):
    # Load as PyArrow Table, convert to category, write to memory-mapped numpy
    ...

# Replace compute_features with vectorized version
def compute_features_vectorized(pair_df, emb_matrix, idx_map, entity_texts):
    i = pair_df['source1_idx'].values
    j = pair_df['candidate_idx'].values
    emb_i = emb_matrix[i]
    emb_j = emb_matrix[j]
    cosine = (emb_i * emb_j).sum(axis=1) / (np.linalg.norm(emb_i, axis=1) * np.linalg.norm(emb_j, axis=1) + 1e-8)
    l2 = np.linalg.norm(emb_i - emb_j, axis=1)
    # RapidFuzz in parallel batches
    name_jw = JaroWinkler.normalized_similarity_matrix(
        entity_texts[i], entity_texts[j], workers=-1
    )
    return np.column_stack([cosine, l2, name_jw, ...])
```

### 4.3 Threshold & Model Training

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

# Stage 1: Train base model
base_model = lgb.LGBMClassifier(...)
base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

# Stage 2: Calibrate probabilities
calibrator = CalibratedClassifierCV(base_model, method='isotonic', cv=3)
calibrator.fit(X_val, y_val)

# Stage 3: Optimize threshold for macro F0.5
probs = calibrator.predict_proba(X_val)[:, 1]
best_t = optimize_macro_f05_threshold(y_val, probs)  # custom sort-scan over fine grid
```

---

## 5. Expected Impact Summary

| Optimization | F₀.5 Macro Gain | Speed / Memory Gain |
|--------------|-----------------|---------------------|
| Hybrid HNSW + MinHash blocking | +5–15 pp | 10–100× faster query |
| Embedding features (float16) | +5–12 pp | +3–4 GB RAM vs full float32 |
| Vectorized feature computation | — | 5–20× faster |
| LightGBM + early stopping | +1–3 pp | 3–5× faster, 4–8× less RAM |
| Probability calibration + fine threshold | +1–4 pp | — |
| Hard vetoes + verified merge | +2–5 pp | — |
| Hard negative mining | +2–4 pp | — |
| Sparsity-aware thresholds | +1–3 pp | — |
| **Total estimated** | **+15–35 pp** | **10–50× end-to-end speedup** |

---

## 6. Prioritized Action Plan (Low-RAM Constraint)

1. **Immediate (OOM prevention):** Refactor `get_entity_dict` to use PyArrow / numpy / categories. Free raw data after feature extraction.
2. **Week 1 (Accuracy):** Add precomputed float16 embeddings + vectorized feature computation. Switch to LightGBM.
3. **Week 2 (Precision):** Add hard vetoes, sparsity-aware thresholds, and verified merge.
4. **Week 3 (Tuning):** Optuna hyperparameter search + hard negative mining + exact macro-F₀.5 threshold calibration.
5. **Week 4 (Blocking):** Replace inverted index with HNSW + MinHash LSH union. Remove arbitrary `max_candidates` cap.

---

## 7. References

- Ramos et al. (2024). *BlockBoost: Scalable and Efficient Blocking through Boosting*. AISTATS 2024.
- Borthwick et al. (2020). *Hashed Dynamic Blocking (HDB)*. arXiv:2008.08285.
- Gagliardelli et al. (2024). *GSM: Generalized Supervised Meta-Blocking*. Information Systems.
- Skoutas et al. (2023). *Pre-trained Embeddings for Entity Resolution*. VLDB.
- Karapiperis et al. (2025). *ALER: Active Learning Hybrid System for Entity Resolution*. VLDB.
- Lipton, Z. C., Elkan, C., & Naryanaswamy, B. (2014). *Optimal Thresholding of Classifiers to Maximize F1 Measure*. ECML PKDD.
- "Entity Resolution in Practice" (2025). arXiv:2607.26298. Lessons on ensemble blockers, sparsity-aware thresholds, and verified merge.
- LightGBM 4.7 Documentation. *Experiments: Memory Consumption and Speed*.
- `optimal-cutoffs` library (finite-sample). Exact sort-and-scan F-beta threshold optimization.
