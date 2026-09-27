# Large-Scale Business Entity Resolution Pipeline

This repository contains an end-to-end Machine Learning pipeline designed to perform **Business Entity Resolution (ER)** at a massive scale. The objective of this project is to deduplicate and link highly noisy, unstructured business records (such as company names and addresses) across multiple independent data sources.

## Overview

In real-world commercial datasets, business identities are often fragmented, misspelled, abbreviated, or missing standard identifiers. This project tackles the challenge of identifying which records refer to the exact same real-world business entity.

### Key Capabilities
- **Massive Scale Blocking:** Implements a multi-stage blocking algorithm using inverted indexing to reduce the mathematical search space. Capable of generating and filtering over **76 million** candidate pairs in under an hour.
- **High-Performance Parallelization:** A custom `joblib`-based parallel processing engine (with threading backends and lazy generator chunking) that maximizes multi-core CPU utilization while keeping memory footprint strictly under 3GB.
- **Advanced Text Similarity Features:** Extracts fuzzy-matching features utilizing highly optimized C++ libraries (`rapidfuzz`) to compute Jaro-Winkler, Token Jaccard, Partial Set Ratio, and Token Sort Ratios across names and addresses.
- **XGBoost / LightGBM Ensembles:** Uses gradient boosted tree models (optimized for precision-heavy F0.5 metrics) to classify candidate pairs as matches or non-matches.

## Architecture

The pipeline consists of three core stages:
1. **Candidate Generation (Blocking):** 
   - Uses localized multi-key exact matching and TF-IDF Cosine Similarity re-ranking to drastically cut the $N \times M$ search space.
2. **Feature Engineering:** 
   - Dynamically computes string distance metrics on-the-fly across parallel worker threads.
3. **Inference & Heuristics:** 
   - XGBoost probabilities are combined with deterministic heuristic overrides (e.g., exact matches, conflicting countries, severe length discrepancies) to produce the final deduplicated linking map.

## Project Structure

```
├── src/
│   ├── blocking/
│   │   └── blocking_v2.py       # TF-IDF & Inverted Index blocking logic
│   ├── matching/
│   │   ├── predict_parallel.py  # High-performance parallel inference engine
│   │   └── ensemble_v2.py       # Dual-GPU LightGBM + XGBoost training script
│   └── utils/
│       └── validate_submission.py # Helper to validate structural integrity of output graphs
├── xgb_model.pkl                # Pre-trained gradient boosted model
└── README.md
```

## Running the Pipeline

To run the inference engine on a multi-core machine:

```bash
# 1. Install dependencies
pip install pandas numpy xgboost rapidfuzz joblib scikit-learn

# 2. Execute parallel prediction
python src/matching/predict_parallel.py
```
*(Ensure your data sources are located in `dataset/` and your blocking output is mapped to the `output/` directory as specified in the script).*

## License
MIT License
