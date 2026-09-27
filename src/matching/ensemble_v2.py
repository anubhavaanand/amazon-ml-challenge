"""
Amazon ML Challenge 2026 — V2 Ensemble Script

Designed for Kaggle Dual T4 GPUs.
Trains XGBoost + LightGBM on V2 blocking candidates with deep text-similarity features,
hard negative mining, and Stratified K-Fold cross-validation to find the optimal
ensemble threshold for macro F0.5.
"""

import re
import heapq
import argparse
import os
import sys
import csv
import json
import pickle
import random
import time
import gc
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
import lightgbm as lgb


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)


# ---------------------------------------------------------------------------
# Abbreviation expansion
# ---------------------------------------------------------------------------
_ABBREV = {
    r"\bpvt\b": "private",
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bllc\b": "limited liability company",
    r"\bltd\b": "limited",
    r"\bco\b": "company",
    r"\bintl\b": "international",
    r"\bstr\b": "street",
    r"\bave\b": "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b": "drive",
    r"\brd\b": "road",
    r"\bln\b": "lane",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bsoln\.?\b": "solution",
    r"\bma\b": "massachusetts",
}


def expand_abbrev(text: str) -> str:
    text = str(text).lower()
    for pat, repl in _ABBREV.items():
        text = re.sub(pat, repl, text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Similarity helpers
# ---------------------------------------------------------------------------
def token_jaccard(a: str, b: str) -> float:
    A = set(expand_abbrev(a).split())
    B = set(expand_abbrev(b).split())
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def trigrams(s: str) -> set[str]:
    s = str(s).lower().replace(" ", "")
    if len(s) < 3:
        return set()
    return {s[i : i + 3] for i in range(len(s) - 2)}


# ---------------------------------------------------------------------------
# Feature engineering V2
# ---------------------------------------------------------------------------
def compute_features_v2(
    df_pairs: pd.DataFrame,
    s1_df: pd.DataFrame,
    s23_df: pd.DataFrame,
    tfidf_vec: Optional[TfidfVectorizer] = None,
    char_vec: Optional[CountVectorizer] = None,
) -> pd.DataFrame:
    merged = df_pairs.merge(
        s1_df, left_on="source1_entity_id", right_on="entity_id", how="inner"
    )
    merged = merged.merge(
        s23_df, left_on="candidate_entity_id", right_on="entity_id", how="inner"
    )
    merged = merged.rename(
        columns={
            "business_name_x": "name1",
            "business_address_x": "addr1",
            "country_x": "country1",
            "business_name_y": "name2",
            "business_address_y": "addr2",
            "country_y": "country2",
        }
    )

    n1_list = merged["name1"].astype(str).tolist()
    n2_list = merged["name2"].astype(str).tolist()
    a1_list = merged["addr1"].astype(str).tolist()
    a2_list = merged["addr2"].astype(str).tolist()
    c1_list = merged["country1"].astype(str).tolist()
    c2_list = merged["country2"].astype(str).tolist()

    features = {
        "name_jw": [JaroWinkler.normalized_similarity(a, b) for a, b in zip(n1_list, n2_list)],
        "addr_jw": [JaroWinkler.normalized_similarity(a, b) for a, b in zip(a1_list, a2_list)],
        "name_jaccard": [token_jaccard(a, b) for a, b in zip(n1_list, n2_list)],
        "addr_jaccard": [token_jaccard(a, b) for a, b in zip(a1_list, a2_list)],
        "name_partial": [
            fuzz.partial_token_set_ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)
        ],
        "addr_partial": [
            fuzz.partial_token_set_ratio(a, b) / 100.0 for a, b in zip(a1_list, a2_list)
        ],
        "name_sort": [
            fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(n1_list, n2_list)
        ],
        "addr_sort": [
            fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(a1_list, a2_list)
        ],
        "country_match": [
            1.0 if x == y and x != "" else 0.0 for x, y in zip(c1_list, c2_list)
        ],
    }

    # TF-IDF cosine
    if tfidf_vec is None:
        tfidf_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=50000)
        all_text = pd.concat(
            [
                pd.Series(n1_list) + " " + pd.Series(a1_list),
                pd.Series(n2_list) + " " + pd.Series(a2_list),
            ]
        ).fillna("")
        tfidf_vec.fit(all_text)

    text1 = pd.Series(n1_list).fillna("") + " " + pd.Series(a1_list).fillna("")
    text2 = pd.Series(n2_list).fillna("") + " " + pd.Series(a2_list).fillna("")
    X1 = tfidf_vec.transform(text1)
    X2 = tfidf_vec.transform(text2)
    features["tfidf_cosine"] = cosine_similarity(X1, X2).diagonal().tolist()

    # Character bigram Dice
    if char_vec is None:
        char_vec = CountVectorizer(analyzer="char", ngram_range=(2, 2), binary=True)
        all_text = pd.concat(
            [
                pd.Series(n1_list) + " " + pd.Series(a1_list),
                pd.Series(n2_list) + " " + pd.Series(a2_list),
            ]
        ).fillna("")
        char_vec.fit(all_text)

    X1_b = char_vec.transform(text1)
    X2_b = char_vec.transform(text2)
    intersection = np.array(X1_b.multiply(X2_b).sum(axis=1)).flatten()
    len1 = np.array(X1_b.sum(axis=1)).flatten()
    len2 = np.array(X2_b.sum(axis=1)).flatten()
    features["char_bigram_dice"] = (2 * intersection / (len1 + len2 + 1e-8)).tolist()

    # Character trigram Jaccard
    features["char_trigram_jaccard"] = [
        len(trigrams(a) & trigrams(b)) / len(trigrams(a) | trigrams(b))
        if (trigrams(a) | trigrams(b))
        else 1.0
        for a, b in zip(n1_list, n2_list)
    ]

    # Length ratios
    features["name_len_ratio"] = np.array(
        [len(a) for a in n1_list]
    ) / np.array([max(len(b), 1) for b in n2_list])
    features["addr_len_ratio"] = np.array(
        [len(a) for a in a1_list]
    ) / np.array([max(len(b), 1) for b in a2_list])

    # Token overlap count
    features["token_overlap_count"] = [
        len((set(expand_abbrev(a).split()) | set(expand_abbrev(c).split()))
            & (set(expand_abbrev(b).split()) | set(expand_abbrev(d).split())))
        for a, b, c, d in zip(n1_list, n2_list, a1_list, a2_list)
    ]

    # Prefix matches
    features["name_prefix4_match"] = [
        1.0 if a[:4] == b[:4] and a[:4] != "" else 0.0 for a, b in zip(n1_list, n2_list)
    ]
    features["name_prefix8_match"] = [
        1.0 if a[:8] == b[:8] and a[:8] != "" else 0.0 for a, b in zip(n1_list, n2_list)
    ]

    return pd.DataFrame(features)


# ---------------------------------------------------------------------------
# Hard negative mining V2
# ---------------------------------------------------------------------------
def hard_negative_mining_v2(
    candidates_file: str,
    positive_pairs: set,
    s1_dict: dict,
    s23_dict: dict,
    keep_ratio: float = 0.1,
    max_heap_size: int = 10_000_000,
) -> pd.DataFrame:
    positive_list = []
    heap = []
    tiebreak = 0

    print(
        f"[ensemble] Hard negative mining: keep_ratio={keep_ratio}, max_heap={max_heap_size:,}"
    )

    with open(candidates_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row["source1_entity_id"]
            cand_id = row["candidate_entity_id"].strip()

            if not cand_id or cand_id == "nan":
                continue

            if (s1_id, cand_id) in positive_pairs:
                positive_list.append(
                    {"source1_entity_id": s1_id, "candidate_entity_id": cand_id, "label": 1}
                )
                continue

            s1 = s1_dict.get(s1_id, {})
            s23 = s23_dict.get(cand_id, {})
            name1 = str(s1.get("business_name", "")).lower()
            name2 = str(s23.get("business_name", "")).lower()
            addr1 = str(s1.get("business_address", "")).lower()
            addr2 = str(s23.get("business_address", "")).lower()
            c1 = str(s1.get("country", "")).lower()
            c2 = str(s23.get("country", "")).lower()

            nj = token_jaccard(name1, name2)
            aj = token_jaccard(addr1, addr2)
            cm = 1.0 if c1 == c2 and c1 != "" else 0.0
            difficulty = 0.5 * nj + 0.3 * aj + 0.2 * cm

            if len(heap) >= max_heap_size and difficulty <= heap[0][0]:
                continue
            heapq.heappush(
                heap,
                (difficulty, tiebreak, {"source1_entity_id": s1_id, "candidate_entity_id": cand_id, "label": 0}),
            )
            tiebreak += 1
            if len(heap) > max_heap_size:
                heapq.heappop(heap)

    target_neg = int(len(positive_list) * max(1.0, 1.0 / keep_ratio))
    heap.sort(reverse=True)
    hard_negatives = [record for _, _, record in heap[:target_neg]]

    pair_list = positive_list + hard_negatives
    random.shuffle(pair_list)

    print(
        f"[ensemble] Hard negatives: {len(hard_negatives):,} / {len(heap):,} total"
    )
    print(
        f"[ensemble] Final training pairs: {len(pair_list):,} (Positives: {len(positive_list):,})"
    )
    return pd.DataFrame(pair_list)


# ---------------------------------------------------------------------------
# Dual-GPU training
# ---------------------------------------------------------------------------
def train_ensemble_dual_gpu(X_train, y_train, X_val, y_val):
    import concurrent.futures

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_weight = min(neg_count / pos_count, 100.0) if pos_count > 0 else 5.0

    def train_xgb():
        model = xgb.XGBClassifier(
            objective="binary:logistic",
            scale_pos_weight=scale_weight,
            max_depth=4,
            gamma=2,
            min_child_weight=10,
            learning_rate=0.05,
            max_delta_step=3,
            subsample=0.8,
            colsample_bytree=0.8,
            n_estimators=300,
            eval_metric="aucpr",
            n_jobs=-1,
            tree_method="hist",
            device="cuda",
            seed=42,
        )
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=30,
            verbose=False,
        )
        return model

    def train_lgb():
        lgb_params = {
            "objective": "binary",
            "metric": "aucpr",
            "device": "gpu",
            "gpu_platform_id": 0,
            "gpu_device_id": 1,
            "gpu_use_dp": "false",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": scale_weight,
            "min_child_weight": 10,
            "max_depth": 4,
            "n_estimators": 300,
            "early_stopping_round": 30,
            "verbose": -1,
            "seed": 42,
        }
        lgb_train = lgb.Dataset(X_train, label=y_train)
        lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
        model = lgb.train(
            lgb_params,
            lgb_train,
            valid_sets=[lgb_val],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        return model

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        xgb_future = executor.submit(train_xgb)
        lgb_future = executor.submit(train_lgb)
        xgb_model = xgb_future.result()
        lgb_model = lgb_future.result()

    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    lgb_val_probs = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)

    best_w, best_t = optimize_ensemble_weights(xgb_val_probs, lgb_val_probs, y_val)

    return xgb_model, lgb_model, best_w, best_t, lgb_model.best_iteration


# ---------------------------------------------------------------------------
# Ensemble weight / threshold optimization
# ---------------------------------------------------------------------------
def optimize_ensemble_weights(xgb_probs, lgb_probs, y_true):
    best_f05 = 0.0
    best_w = 0.5
    best_t = 0.5

    for w_xgb in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w_lgb = 1.0 - w_xgb
        ensemble_probs = w_xgb * xgb_probs + w_lgb * lgb_probs

        for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
            preds = (ensemble_probs >= t).astype(int)
            f05 = fbeta_score(y_true, preds, beta=0.5, average="macro", zero_division=0)
            if f05 > best_f05:
                best_f05 = f05
                best_w = w_xgb
                best_t = t

    print(
        f"[ensemble] Optimal ensemble: XGB={best_w:.1f}, LGB={1-best_w:.1f}, threshold={best_t}"
    )
    return best_w, best_t


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------
def cross_validate_ensemble(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []
    ensemble_configs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        print(f"\n[ensemble] Fold {fold}/{n_splits}")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        xgb_model, lgb_model, w_xgb, best_t, lgb_best_iteration = train_ensemble_dual_gpu(
            X_train, y_train, X_val, y_val
        )

        xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
        lgb_val_probs = lgb_model.predict(X_val, num_iteration=lgb_best_iteration)
        ensemble_probs = w_xgb * xgb_val_probs + (1 - w_xgb) * lgb_val_probs

        val_preds = (ensemble_probs >= best_t).astype(int)
        fold_f05 = fbeta_score(y_val, val_preds, beta=0.5, average="macro", zero_division=0)
        fold_scores.append(fold_f05)
        ensemble_configs.append(
            {
                "w_xgb": w_xgb,
                "threshold": best_t,
                "fold": fold,
                "f05": fold_f05,
            }
        )
        print(f"[ensemble] Fold {fold} F0.5: {fold_f05:.4f}")

        del xgb_model, lgb_model
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    avg_f05 = float(np.mean(fold_scores))
    std_f05 = float(np.std(fold_scores))
    print(f"\n[ensemble] Average CV F0.5: {avg_f05:.4f} (±{std_f05:.4f})")

    best_config = max(ensemble_configs, key=lambda x: x["f05"])
    return best_config, avg_f05


# ---------------------------------------------------------------------------
# Test prediction
# ---------------------------------------------------------------------------
def predict_ensemble(
    xgb_model, lgb_model, X, w_xgb: float, threshold: float, lgb_best_iteration: int, batch_size: int = 50000
):
    all_preds = []
    n_batches = (len(X) + batch_size - 1) // batch_size

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, len(X))
        X_batch = X[start:end]

        xgb_probs = xgb_model.predict_proba(X_batch)[:, 1]
        lgb_probs = lgb_model.predict(X_batch, num_iteration=lgb_best_iteration)
        ensemble_probs = w_xgb * xgb_probs + (1 - w_xgb) * lgb_probs
        all_preds.append((ensemble_probs >= threshold).astype(int))

    return np.concatenate(all_preds)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def load_ground_truth(file_path: str) -> set:
    truth_df = pd.read_csv(file_path, sep="\t")
    positive_pairs = set()
    for _, row in truth_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched = str(row["matched_entity_ids"]).split(",")
        for m in matched:
            m = m.strip()
            if m and m != "nan":
                positive_pairs.add((s1_id, m))
    return positive_pairs


def get_entity_dfs(split: str, data_dir: str):
    cols = ["entity_id", "business_name", "business_address", "country"]
    s1_df = pd.read_csv(
        os.path.join(data_dir, f"{split}_source1.tsv"), sep="\t", usecols=cols, dtype="string"
    ).fillna("")
    s2_df = pd.read_csv(
        os.path.join(data_dir, f"{split}_source2.tsv"), sep="\t", usecols=cols, dtype="string"
    ).fillna("")
    s3_df = pd.read_csv(
        os.path.join(data_dir, f"{split}_source3.tsv"), sep="\t", usecols=cols, dtype="string"
    ).fillna("")
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    return s1_df, s23_df


def df_to_dicts(s1_df: pd.DataFrame, s23_df: pd.DataFrame):
    s1_dict = s1_df.set_index("entity_id").to_dict("index")
    s23_dict = s23_df.set_index("entity_id").to_dict("index")
    return s1_dict, s23_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — V2 Ensemble")
    parser.add_argument("--data_dir", type=str, default="dataset/train")
    parser.add_argument("--candidates_file", type=str, default="output/v2_train_candidates.tsv")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict"])
    parser.add_argument("--model_dir", type=str, default="output")
    parser.add_argument("--out_file", type=str, default="output/matching_results.tsv")
    parser.add_argument("--test_candidates", type=str, default="output/v2_test_candidates.tsv")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--keep_ratio", type=float, default=0.1)
    parser.add_argument("--max_heap_size", type=int, default=10_000_000)
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    if args.mode == "train":
        print("[ensemble] Loading ground truth...")
        positive_pairs = load_ground_truth(os.path.join(args.data_dir, "train_ground_truth.tsv"))

        print("[ensemble] Loading entity data...")
        s1_df, s23_df = get_entity_dfs("train", args.data_dir)
        s1_dict, s23_dict = df_to_dicts(s1_df, s23_df)

        print("[ensemble] Hard negative mining...")
        df_pairs = hard_negative_mining_v2(
            args.candidates_file,
            positive_pairs,
            s1_dict,
            s23_dict,
            keep_ratio=args.keep_ratio,
            max_heap_size=args.max_heap_size,
        )

        print("[ensemble] Computing V2 features...")
        X = compute_features_v2(df_pairs, s1_df, s23_df)
        y = df_pairs["label"]

        print(f"[ensemble] Feature matrix shape: {X.shape}")
        print(f"[ensemble] Positive rate: {y.mean():.4f}")

        print("[ensemble] Starting stratified K-fold cross-validation...")
        best_config, avg_f05 = cross_validate_ensemble(X, y, n_splits=args.n_splits)

        print("\n[ensemble] Training final model on full training data...")
        xgb_model, lgb_model, w_xgb, best_t, lgb_best_iteration = train_ensemble_dual_gpu(
            X, y, X, y
        )

        print("[ensemble] Saving models...")
        xgb_model.save_model(os.path.join(args.model_dir, "xgb_model.json"))
        lgb_model.save_model(os.path.join(args.model_dir, "lgb_model.txt"))
        with open(os.path.join(args.model_dir, "ensemble_config.json"), "w") as f:
            json.dump(
                {
                    "w_xgb": w_xgb,
                    "threshold": best_t,
                    "lgb_best_iteration": int(lgb_best_iteration),
                    "cv_avg_f05": avg_f05,
                    "keep_ratio": args.keep_ratio,
                },
                f,
                indent=2,
            )

        print(
            f"[ensemble] Done. Best ensemble: XGB={w_xgb:.1f}, LGB={1-w_xgb:.1f}, threshold={best_t}"
        )
        print(f"[ensemble] CV average F0.5: {avg_f05:.4f}")

    else:
        print("[ensemble] Prediction mode")
        print("[ensemble] Loading test candidates...")
        test_candidates = pd.read_csv(args.test_candidates, sep="\t")
        s1_df, s23_df = get_entity_dfs("test", args.data_dir)

        pair_list = []
        for _, row in test_candidates.iterrows():
            s1_id = row["source1_entity_id"]
            cands = str(row["candidate_entity_ids"]).split(",")
            for c in cands:
                c = c.strip()
                if c and c != "nan":
                    pair_list.append(
                        {"source1_entity_id": s1_id, "candidate_entity_id": c}
                    )

        df_pairs_test = pd.DataFrame(pair_list)
        print(f"[ensemble] Test pairs: {len(df_pairs_test):,}")

        print("[ensemble] Computing features...")
        X_test = compute_features_v2(df_pairs_test, s1_df, s23_df)

        print("[ensemble] Loading models...")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(args.model_dir, "xgb_model.json"))
        lgb_model = lgb.Booster(model_file=os.path.join(args.model_dir, "lgb_model.txt"))

        with open(os.path.join(args.model_dir, "ensemble_config.json"), "r") as f:
            config = json.load(f)
        w_xgb = config["w_xgb"]
        threshold = config["threshold"]
        lgb_best_iteration = config.get("lgb_best_iteration", lgb_model.best_iteration)

        print("[ensemble] Predicting...")
        preds = predict_ensemble(
            xgb_model,
            lgb_model,
            X_test,
            w_xgb=w_xgb,
            threshold=threshold,
            lgb_best_iteration=lgb_best_iteration,
            batch_size=50000,
        )
        df_pairs_test["is_match"] = preds

        matches = (
            df_pairs_test[df_pairs_test["is_match"] == 1]
            .groupby("source1_entity_id")["candidate_entity_id"]
            .apply(list)
        )
        match_dict = matches.to_dict()

        print(f"[ensemble] Writing final submission to {args.out_file}...")
        with open(args.out_file, "w", encoding="utf-8") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for _, row in test_candidates.iterrows():
                s1_id = row["source1_entity_id"]
                m = match_dict.get(s1_id, [])
                f.write(f"{s1_id}\t{','.join(m)}\n")

        print("[ensemble] Complete!")


if __name__ == "__main__":
    main()
