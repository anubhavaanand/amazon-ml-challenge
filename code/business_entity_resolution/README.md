# Business Entity Resolution Pipeline

This repository contains the end-to-end pipeline for the Amazon ML Challenge 2026.

## Requirements
To run this pipeline, install the requirements:
```bash
pip install -r requirements.txt
```

## Pipeline Overview
The pipeline consists of two stages designed to scale efficiently while maximizing the F_0.5 score.

### Stage 1: Candidate Generation (Blocking)
`src/blocking.py`
This stage generates a high-recall candidate set using character 3-gram TF-IDF embeddings and highly optimized sparse matrix multiplication. For every Source 1 entity, it pulls the Top-20 most similar entities from Source 2 and 3.

### Stage 2: Precision Matching (Inference)
`src/matching.py`
This stage iterates over the candidate pairs and computes pairwise features (Levenshtein distance, Jaccard similarity, and Exact Country Match). It then uses an XGBoost classifier to apply a strict precision threshold (maximizing the F_0.5 score where Precision is weighted 2x over Recall).

## Reproduction Steps

1. **Candidate Generation (Train Set)**
```bash
python3 src/blocking.py --data_dir ../../dataset/train --out_file ../../output/train_candidates.tsv
```

2. **Train the XGBoost Model**
```bash
python3 src/matching.py --data_dir ../../dataset/train --candidates_file ../../output/train_candidates.tsv --mode train
```

3. **Candidate Generation (Test Set)**
```bash
python3 src/blocking.py --data_dir ../../dataset/test --out_file ../../output/candidate_pairs.tsv
```

4. **Inference (Test Set)**
```bash
python3 src/matching.py --data_dir ../../dataset/test --candidates_file ../../output/candidate_pairs.tsv --out_file ../../output/matching_results.tsv --mode predict --threshold 0.75
```

5. **Validate Output**
```bash
python3 ../../utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir ../../dataset/test
```
