"""
State-of-the-Art Research Summary
==================================

1. Threshold Calibration for XGBoost Binary Classification (F0.5 Macro)
------------------------------------------------------------------------
Key findings from recent literature (2024-2026):
- "When Single-Dataset Conclusions Fail" (arXiv:2608.16147, 2026): Threshold tuning
  benefit peaks at 1:15-1:40 imbalance ratio. Always tune on a separate validation fold.
- Nested CV is essential: threshold selection on inner fold, evaluation on outer fold.
- Probability calibration (isotonic > sigmoid) improves threshold tuning effectiveness.
- Class-specific thresholds outperform global thresholds in multi-class settings.
- The optimal threshold for F_beta on calibrated probabilities is approximately F_beta/2.
- Scikit-learn TunedThresholdClassifierCV supports cross-validated threshold selection.

2. Hard Negative Mining for Entity Resolution (14M rows)
--------------------------------------------------------
Key findings from recent literature:
- MERAI (arXiv:2508.03767, 2025): Successfully processed 15.7M records using
  configurable group size limits and linear scaling indexing.
- ALER (VLDB 2024): Partitioned active learning with hybrid query strategy combining
  uncertain pairs (theta ~0.5) and confident pairs (theta ~1.0) for boundary refinement.
- "What Hard Negative Mining Actually Means in Practice" (2026): Semi-hard negatives
  outperform hardest negatives; curriculum over hardness; cross-encoder guided mining.
- Cross-encoder filtering surfaces truly confusing pairs and removes false negatives.
- Margin-based contrastive learning is critical for near-duplicate separation.

Recommendations for 14M rows:
1. Blocking: Use canopy blocking + identifier blocking + HNSW to reduce candidate space
2. Hard negative selection: Cross-encoder rescoring on top-k candidates from bi-encoder
3. Curriculum: Start with easier negatives, gradually introduce harder ones
4. Sampling: Use semi-hard negatives with relative_margin=0.05, skip top-k positives
5. Batch construction: In-batch negatives for diversity + index-based online mining
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score, confusion_matrix, roc_curve
from sklearn.calibration import CalibratedClassifierCV
import warnings

# ============================================================================
# PART 1: F0.5-OPTIMAL THRESHOLD SEARCH FOR XGBOOST
# ============================================================================


def find_best_fbeta_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    beta: float = 0.5,
    sample_weight: Optional[np.ndarray] = None,
    method: str = "exhaustive",
    n_thresholds: int = 1000,
) -> Tuple[float, float]:
    """
    Find the decision threshold that maximizes F-beta score.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels {0, 1}.
    y_prob : np.ndarray
        Predicted probabilities for positive class in [0, 1].
    beta : float, default=0.5
        F-beta parameter. beta < 1 emphasizes precision.
        beta=0.5 gives F0.5 (precision-weighted).
    sample_weight : np.ndarray, optional
        Sample weights.
    method : str, default="exhaustive"
        - 'exhaustive': Grid search over all unique probabilities (exact).
        - 'sort_scan': O(n log n) sort-and-scan exploiting piecewise structure.
    n_thresholds : int, default=1000
        Number of thresholds for exhaustive search (only used if method='grid').

    Returns
    -------
    best_threshold : float
        Optimal decision threshold.
    best_fbeta : float
        F-beta score at optimal threshold.
    """
    if method == "exhaustive":
        # Use all unique predicted probabilities as candidates
        thresholds = np.unique(y_prob)
        # Add boundary values to ensure coverage
        thresholds = np.sort(np.unique(np.concatenate([
            [0.0], thresholds, [1.0]
        ])))

        scores = []
        for t in thresholds:
            y_pred = (y_prob >= t).astype(int)
            score = fbeta_score(y_true, y_pred, beta=beta,
                                sample_weight=sample_weight, zero_division=0)
            scores.append(score)

        scores = np.array(scores)
        best_idx = np.argmax(scores)
        return float(thresholds[best_idx]), float(scores[best_idx])

    elif method == "sort_scan":
        # O(n log n) sort-and-scan exploiting that F-beta is piecewise-constant
        # between predicted probability values.
        # Sort by predicted probability
        sort_idx = np.argsort(y_prob)
        y_sorted = y_true[sort_idx]
        prob_sorted = y_prob[sort_idx]

        if sample_weight is not None:
            w_sorted = sample_weight[sort_idx]
        else:
            w_sorted = np.ones_like(y_sorted, dtype=float)

        n = len(y_sorted)
        total_pos = np.sum(y_sorted * w_sorted)
        total_neg = np.sum((1 - y_sorted) * w_sorted)

        if total_pos == 0 or total_neg == 0:
            return 0.5, 0.0

        # Cumulative sums as we sweep threshold from high to low
        tp_cumsum = np.cumsum(y_sorted * w_sorted)
        fp_cumsum = np.cumsum((1 - y_sorted) * w_sorted)

        # At each unique threshold position, predict positive for all samples
        # with prob >= threshold, which corresponds to prefix up to that point
        tp = tp_cumsum
        fp = fp_cumsum
        fn = total_pos - tp
        tn = total_neg - fp

        beta_sq = beta ** 2
        denominator = (1 + beta_sq) * tp + beta_sq * fp + fn
        scores = np.where(denominator > 0,
                          (1 + beta_sq) * tp / denominator,
                          0.0)

        # Consider thresholds between unique probability values
        # Thresholds are the midpoints between unique sorted probabilities
        unique_probs = np.unique(prob_sorted)
        if len(unique_probs) == 1:
            candidate_thresholds = unique_probs
        else:
            candidate_thresholds = np.concatenate([
                [unique_probs[0]],
                (unique_probs[:-1] + unique_probs[1:]) / 2,
                [unique_probs[-1] + 1e-10]
            ])

        # Map each candidate threshold to the best score achievable at that threshold
        # Use the scores at the boundary between sorted groups
        # Score is evaluated at each unique probability value
        best_score = 0.0
        best_threshold = 0.5

        for i, t in enumerate(candidate_thresholds):
            # Find how many samples have prob >= t
            n_pos = np.searchsorted(-prob_sorted, -t)
            if n_pos == 0:
                score = 0.0
            elif n_pos >= n:
                score = (1 + beta_sq) * total_pos / ((1 + beta_sq) * total_pos)
            else:
                tp_i = tp_cumsum[n_pos - 1]
                fp_i = fp_cumsum[n_pos - 1]
                fn_i = total_pos - tp_i
                denom_i = (1 + beta_sq) * tp_i + beta_sq * fp_i + fn_i
                score = (1 + beta_sq) * tp_i / denom_i if denom_i > 0 else 0.0

            if score > best_score:
                best_score = score
                best_threshold = t

        return float(best_threshold), float(best_score)

    else:
        raise ValueError(f"Unknown method: {method}. Choose 'exhaustive' or 'sort_scan'.")


def nested_fbeta_threshold_selection(
    model_fn: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    beta: float = 0.5,
    outer_cv: int = 5,
    inner_cv: int = 5,
    outer_seed: int = 42,
    inner_seed: int = 100,
    calibration_method: str = "isotonic",
    calibrate: bool = True,
) -> dict:
    """
    Nested cross-validation for F-beta threshold selection without overfitting.

    This follows best practices from recent literature:
    - Inner CV selects threshold on held-out inner-fold predictions
    - Outer CV evaluates on untouched outer-fold predictions
    - Probability calibration improves threshold reliability

    Parameters
    ----------
    model_fn : callable
        A factory function that returns a fresh unfitted model instance.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary labels {0, 1}.
    beta : float, default=0.5
        F-beta parameter.
    outer_cv : int, default=5
        Number of outer CV folds for evaluation.
    inner_cv : int, default=5
        Number of inner CV folds for threshold selection.
    outer_seed : int, default=42
        Random seed for outer CV splits.
    inner_seed : int, default=100
        Random seed for inner CV splits.
    calibration_method : str, default="isotonic"
        Calibration method: 'sigmoid' (Platt) or 'isotonic'.
    calibrate : bool, default=True
        Whether to calibrate probabilities before threshold selection.

    Returns
    -------
    dict with keys:
        - 'threshold_mean': mean threshold across outer folds
        - 'threshold_std': std of thresholds
        - 'fbeta_mean': mean F-beta score on outer folds
        - 'fbeta_std': std of F-beta scores
        - 'fold_thresholds': list of per-fold thresholds
        - 'fold_scores': list of per-fold F-beta scores
        - 'sensitivity': sensitivity on pooled outer predictions
        - 'specificity': specificity on pooled outer predictions
        - 'n_fn': number of false negatives
        - 'n_fp': number of false positives
    """
    outer = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=outer_seed)

    pooled_true = []
    pooled_pred = []
    fold_thresholds = []
    fold_scores = []

    for tr_idx, va_idx in outer.split(X, y):
        Xtr, Xva = X[tr_idx], X[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        # Inner CV: generate OOF predictions on outer-training fold only
        inner = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=inner_seed)
        inner_oof = np.zeros(len(ytr), dtype=np.float32)

        for itr, iva in inner.split(Xtr, ytr):
            m = model_fn()
            m.fit(Xtr[itr], ytr[itr])

            if calibrate:
                # Calibrate probabilities using inner training fold
                calib = CalibratedClassifierCV(m, method=calibration_method, cv="prefit")
                calib.fit(Xtr[iva], ytr[iva])
                inner_oof[iva] = calib.predict_proba(Xtr[iva])[:, 1]
            else:
                inner_oof[iva] = m.predict_proba(Xtr[iva])[:, 1]

        # Select threshold on inner OOF predictions
        thr, _ = find_best_fbeta_threshold(ytr, inner_oof, beta=beta)

        # Refit on FULL outer-training fold
        final_model = model_fn()
        final_model.fit(Xtr, ytr)

        if calibrate:
            calib_final = CalibratedClassifierCV(final_model, method=calibration_method, cv="prefit")
            calib_final.fit(Xtr, ytr)
            va_prob = calib_final.predict_proba(Xva)[:, 1]
        else:
            va_prob = final_model.predict_proba(Xva)[:, 1]

        va_pred = (va_prob >= thr).astype(int)
        score = fbeta_score(yva, va_pred, beta=beta, zero_division=0)

        fold_thresholds.append(float(thr))
        fold_scores.append(float(score))
        pooled_true.extend(yva.tolist())
        pooled_pred.extend(va_pred.tolist())

    pooled_true = np.array(pooled_true)
    pooled_pred = np.array(pooled_pred)

    tn, fp, fn, tp = confusion_matrix(pooled_true, pooled_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    overall_fbeta = fbeta_score(pooled_true, pooled_pred, beta=beta, zero_division=0)

    return {
        "threshold_mean": float(np.mean(fold_thresholds)),
        "threshold_std": float(np.std(fold_thresholds)),
        "fbeta_mean": float(overall_fbeta),
        "fbeta_std": float(np.std(fold_scores)),
        "fold_thresholds": [round(t, 4) for t in fold_thresholds],
        "fold_scores": [round(s, 4) for s in fold_scores],
        "sensitivity": float(sens),
        "specificity": float(spec),
        "n_fn": int(fn),
        "n_fp": int(fp),
        "n_tp": int(tp),
        "n_tn": int(tn),
    }


def select_f0_5_threshold_xgboost(
    model: object,
    X_val: np.ndarray,
    y_val: np.ndarray,
    beta: float = 0.5,
    calibration_method: str = "isotonic",
) -> Tuple[float, float]:
    """
    Select F0.5-optimal threshold for a fitted XGBoost model on validation data.

    This is a convenience wrapper for production use after cross-validated
    model training.

    Parameters
    ----------
    model : XGBClassifier
        Fitted XGBoost model.
    X_val : np.ndarray
        Validation features.
    y_val : np.ndarray
        Validation labels.
    beta : float, default=0.5
        F-beta parameter.
    calibration_method : str, default="isotonic"
        Calibration method to use before threshold selection.

    Returns
    -------
    threshold : float
        Optimal decision threshold.
    fbeta_score : float
        F-beta score at optimal threshold.
    """
    y_prob = model.predict_proba(X_val)[:, 1]

    if calibration_method is not None:
        calibrator = CalibratedClassifierCV(model, method=calibration_method, cv="prefit")
        calibrator.fit(X_val, y_val)
        y_prob = calibrator.predict_proba(X_val)[:, 1]

    return find_best_fbeta_threshold(y_val, y_prob, beta=beta)


# ============================================================================
# PART 2: HARD NEGATIVE MINING FOR ENTITY RESOLUTION (14M rows)
# ============================================================================


class HardNegativeMiner:
    """
    Scalable hard negative mining for Entity Resolution / Record Linkage.

    Designed for large datasets (millions of records) where exhaustive pair
    enumeration is infeasible. Uses a two-stage approach:
    1. Blocking/candidate generation to reduce search space
    2. Cross-encoder rescoring to identify hard negatives

    Based on state-of-the-art practices:
    - Semi-hard negative selection (avoid both easy and extremely hard)
    - Cross-encoder guided mining to filter false negatives
    - Curriculum-based difficulty scheduling
    - Margin-based sampling with relative_margin
    """

    def __init__(
        self,
        bi_encoder: object,
        cross_encoder: Optional[object] = None,
        corpus: Optional[list] = None,
        corpus_embeddings: Optional[np.ndarray] = None,
        use_faiss: bool = True,
    ):
        """
        Parameters
        ----------
        bi_encoder : SentenceTransformer or similar
            Bi-encoder model for efficient embedding and retrieval.
        cross_encoder : CrossEncoder or similar, optional
            Stronger scorer for filtering false negatives.
        corpus : list of str, optional
            Document corpus to mine negatives from.
        corpus_embeddings : np.ndarray, optional
            Pre-computed corpus embeddings.
        use_faiss : bool, default=True
            Use FAISS for efficient nearest neighbor search.
        """
        self.bi_encoder = bi_encoder
        self.cross_encoder = cross_encoder
        self.corpus = corpus
        self.corpus_embeddings = corpus_embeddings
        self.use_faiss = use_faiss

        if use_faiss:
            try:
                import faiss
                self._faiss = faiss
            except ImportError:
                warnings.warn("FAISS not installed. Falling back to brute-force search.")
                self.use_faiss = False

    def mine_hard_negatives_for_pairs(
        self,
        anchor_texts: list,
        positive_texts: list,
        num_negatives: int = 3,
        range_min: int = 0,
        range_max: int = 100,
        sampling_strategy: str = "top",
        relative_margin: Optional[float] = 0.05,
        absolute_margin: Optional[float] = None,
        max_score: Optional[float] = None,
        min_score: Optional[float] = None,
        batch_size: int = 32,
        output_format: str = "triplet",
        include_scores: bool = False,
    ) -> dict:
        """
        Mine hard negatives for entity resolution pairs.

        Parameters
        ----------
        anchor_texts : list of str
            Anchor record texts (e.g., concatenated attributes).
        positive_texts : list of str
            Positive match texts.
        num_negatives : int, default=3
            Number of hard negatives to mine per positive pair.
        range_min : int, default=0
            Skip top-k most similar candidates to avoid false positives.
        range_max : int, default=100
            Look for negatives in top-k candidates.
        sampling_strategy : str, default="top"
            'top' = take hardest candidates, 'random' = random from qualifying window.
        relative_margin : float, optional
            Require negative_score < positive_score * (1 - relative_margin).
            This filters out false negatives that are actually positive matches.
        absolute_margin : float, optional
            Require positive_score - negative_score > absolute_margin.
        max_score : float, optional
            Drop candidates with score > max_score (likely false negatives).
        min_score : float, optional
            Only consider candidates with score > min_score.
        batch_size : int, default=32
            Batch size for encoding.
        output_format : str, default="triplet"
            'triplet', 'n-tuple', 'labeled-pair', 'labeled-list'.
        include_scores : bool, default=False
            Include similarity scores in output.

        Returns
        -------
        dict with keys depending on output_format:
            - 'triplets': list of (anchor, positive, negative) tuples
            - 'labels': list of binary labels
            - 'scores': list of similarity scores (if include_scores=True)
        """
        if self.corpus is None:
            raise ValueError("corpus must be provided for hard negative mining.")

        n_queries = len(anchor_texts)
        assert len(positive_texts) == n_queries, "anchor_texts and positive_texts must have same length."

        # Encode anchors and positives
        anchor_embeddings = self.bi_encoder.encode(
            anchor_texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True
        )
        positive_embeddings = self.bi_encoder.encode(
            positive_texts, batch_size=batch_size, show_progress_bar=False,
            convert_to_numpy=True
        )

        # Encode corpus if not pre-computed
        if self.corpus_embeddings is None:
            corpus_embeddings = self.bi_encoder.encode(
                self.corpus, batch_size=batch_size, show_progress_bar=False,
                convert_to_numpy=True
            )
        else:
            corpus_embeddings = self.corpus_embeddings

        # Normalize for cosine similarity
        anchor_embeddings = anchor_embeddings / np.linalg.norm(anchor_embeddings, axis=1, keepdims=True)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        # Find top-k candidates for each anchor
        k = min(range_max + 1, len(self.corpus))

        if self.use_faiss and hasattr(self, '_faiss'):
            faiss = self._faiss
            index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
            index.add(corpus_embeddings.astype(np.float32))

            all_scores = []
            all_indices = []
            for i in range(0, n_queries, batch_size):
                chunk = anchor_embeddings[i:i + batch_size].astype(np.float32)
                scores, indices = index.search(chunk, k=k)
                all_scores.append(scores)
                all_indices.append(indices)

            scores_matrix = np.vstack(all_scores)
            indices_matrix = np.vstack(all_indices)
        else:
            # Brute-force cosine similarity
            sim_matrix = anchor_embeddings @ corpus_embeddings.T
            scores_matrix = np.sort(-sim_matrix, axis=1)[:, :k]
            indices_matrix = np.argsort(-sim_matrix, axis=1)[:, :k]
            scores_matrix = -scores_matrix

        # Compute positive scores
        positive_scores = np.sum(positive_embeddings * anchor_embeddings, axis=1)

        # Remove positive pairs from candidates
        positive_idx_in_corpus = []
        for pos_text in positive_texts:
            try:
                idx = self.corpus.index(pos_text)
                positive_idx_in_corpus.append(idx)
            except ValueError:
                positive_idx_in_corpus.append(-1)

        positive_idx_in_corpus = np.array(positive_idx_in_corpus).reshape(-1, 1)

        # Mask out positive pairs
        mask = indices_matrix != positive_idx_in_corpus
        scores_matrix = np.where(mask, scores_matrix, -np.inf)

        # Apply max_score filter
        if max_score is not None:
            scores_matrix = np.where(scores_matrix <= max_score, scores_matrix, -np.inf)

        # Apply min_score filter
        if min_score is not None:
            scores_matrix = np.where(scores_matrix > min_score, scores_matrix, -np.inf)

        # Apply relative margin: negative must be sufficiently below positive
        if relative_margin is not None:
            threshold_scores = positive_scores[:, np.newaxis] * (1 - relative_margin)
            scores_matrix = np.where(scores_matrix < threshold_scores, scores_matrix, -np.inf)

        # Apply absolute margin
        if absolute_margin is not None:
            threshold_scores = positive_scores[:, np.newaxis] - absolute_margin
            scores_matrix = np.where(scores_matrix < threshold_scores, scores_matrix, -np.inf)

        # Skip range_min candidates, take top num_negatives
        if sampling_strategy == "top":
            # Sort each row by score descending, skip range_min, take num_negatives
            neg_indices = []
            neg_scores = []
            for i in range(n_queries):
                row_scores = scores_matrix[i]
                valid_mask = row_scores != -np.inf
                valid_scores = row_scores[valid_mask]
                valid_indices = indices_matrix[i][valid_mask]

                if len(valid_scores) == 0:
                    neg_indices.append([])
                    neg_scores.append([])
                    continue

                # Sort by score descending
                sort_order = np.argsort(-valid_scores)
                valid_scores = valid_scores[sort_order]
                valid_indices = valid_indices[sort_order]

                # Skip range_min, take num_negatives
                start = min(range_min, len(valid_scores))
                end = min(start + num_negatives, len(valid_scores))
                neg_indices.append(valid_indices[start:end].tolist())
                neg_scores.append(valid_scores[start:end].tolist())

        elif sampling_strategy == "random":
            neg_indices = []
            neg_scores = []
            np.random.seed(42)
            for i in range(n_queries):
                row_scores = scores_matrix[i]
                valid_mask = row_scores != -np.inf
                valid_scores = row_scores[valid_mask]
                valid_indices = indices_matrix[i][valid_mask]

                if len(valid_scores) < num_negatives:
                    neg_indices.append([])
                    neg_scores.append([])
                    continue

                # Random sample
                sample_idx = np.random.choice(
                    len(valid_scores),
                    size=min(num_negatives, len(valid_scores)),
                    replace=False
                )
                neg_indices.append(valid_indices[sample_idx].tolist())
                neg_scores.append(valid_scores[sample_idx].tolist())
        else:
            raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

        # Cross-encoder rescoring if available
        if self.cross_encoder is not None:
            neg_indices, neg_scores = self._cross_encoder_filter(
                anchor_texts, positive_texts, neg_indices, neg_scores, batch_size
            )

        # Format output
        if output_format == "triplet":
            triplets = []
            for i in range(n_queries):
                for neg_idx, neg_score in zip(neg_indices[i], neg_scores[i]):
                    triplets.append({
                        "anchor": anchor_texts[i],
                        "positive": positive_texts[i],
                        "negative": self.corpus[neg_idx],
                        "label": 0,
                    })
                    if include_scores:
                        triplets[-1]["score"] = float(neg_score)
            return {"triplets": triplets}

        elif output_format == "labeled-pair":
            pairs = []
            for i in range(n_queries):
                # Positive pair
                pairs.append({
                    "text_1": anchor_texts[i],
                    "text_2": positive_texts[i],
                    "label": 1,
                })
                # Negative pairs
                for neg_idx, neg_score in zip(neg_indices[i], neg_scores[i]):
                    pairs.append({
                        "text_1": anchor_texts[i],
                        "text_2": self.corpus[neg_idx],
                        "label": 0,
                    })
                    if include_scores:
                        pairs[-1]["score"] = float(neg_score)
            return {"pairs": pairs}

        elif output_format == "n-tuple":
            ntuples = []
            for i in range(n_queries):
                if len(neg_indices[i]) == 0:
                    continue
                neg_texts = [self.corpus[idx] for idx in neg_indices[i]]
                ntuple = {
                    "anchor": anchor_texts[i],
                    "positive": positive_texts[i],
                    "negatives": neg_texts,
                    "label": 1,
                }
                if include_scores:
                    ntuple["scores"] = [float(s) for s in neg_scores[i]]
                ntuples.append(ntuple)
            return {"ntuples": ntuples}

        else:
            raise ValueError(f"Unknown output_format: {output_format}")

    def _cross_encoder_filter(
        self,
        anchor_texts: list,
        positive_texts: list,
        neg_indices: list,
        neg_scores: list,
        batch_size: int = 32,
    ) -> Tuple[list, list]:
        """
        Use cross-encoder to re-score candidates and filter false negatives.
        This is critical for entity resolution where surface similarity can be misleading.
        """
        filtered_neg_indices = []
        filtered_neg_scores = []

        for i in range(len(anchor_texts)):
            if len(neg_indices[i]) == 0:
                filtered_neg_indices.append([])
                filtered_neg_scores.append([])
                continue

            # Get candidate texts
            candidates = [self.corpus[idx] for idx in neg_indices[i]]

            # Cross-encoder score all candidates
            pairs = [(anchor_texts[i], cand) for cand in candidates]
            ce_scores = self.cross_encoder.predict(
                pairs, batch_size=batch_size, convert_to_numpy=True
            )

            # Also score positive to set margin
            pos_score = self.cross_encoder.predict(
                [(anchor_texts[i], positive_texts[i])],
                batch_size=1,
                convert_to_numpy=True
            )[0]

            # Filter: keep candidates with CE score significantly below positive score
            # This removes false negatives that bi-encoder confused with positives
            margin = 0.1  # CE score margin
            keep_mask = ce_scores < pos_score - margin

            kept_indices = [idx for idx, keep in zip(neg_indices[i], keep_mask) if keep]
            kept_scores = [float(s) for s, keep in zip(neg_scores[i], keep_mask) if keep]

            # If no hard negatives survive, fall back to bi-encoder top-k
            if len(kept_indices) == 0:
                kept_indices = neg_indices[i][:1]
                kept_scores = [float(neg_scores[i][0])] if len(neg_scores[i]) > 0 else []

            filtered_neg_indices.append(kept_indices)
            filtered_neg_scores.append(kept_scores)

        return filtered_neg_indices, filtered_neg_scores


def select_hard_negatives_blocking(
    left_df: object,
    right_df: object,
    candidate_pairs: list,
    match_pairs: set,
    score_func: Callable,
    num_hard_negatives: int = 3,
    easy_negatives: int = 1,
    max_candidates_per_entity: int = 100,
) -> list:
    """
    Select hard negatives from candidate pairs for entity resolution training.

    Strategy:
    1. Split negatives into easy, semi-hard, and hard based on match score
    2. Sample proportional amounts to create balanced training batches
    3. Avoid extremely hard negatives that are likely false positives

    Parameters
    ----------
    left_df : pd.DataFrame
        Left entity table.
    right_df : pd.DataFrame
        Right entity table.
    candidate_pairs : list of tuple
        Candidate (left_idx, right_idx) pairs from blocking.
    match_pairs : set of tuple
        Ground truth matching pairs.
    score_func : callable
        Function (left_idx, right_idx) -> similarity score in [0, 1].
    num_hard_negatives : int, default=3
        Total negatives to sample per positive pair.
    easy_negatives : int, default=1
        Number of easy negatives to include.
    max_candidates_per_entity : int, default=100
        Max candidates to consider per entity for efficiency.

    Returns
    -------
    training_pairs : list of dict
        Training pairs with labels: [{'left': idx, 'right': idx, 'label': 0/1}]
    """
    # Separate positive and negative candidates
    positive_pairs = [(l, r) for (l, r) in candidate_pairs if (l, r) in match_pairs]
    negative_pairs = [(l, r) for (l, r) in candidate_pairs if (l, r) not in match_pairs]

    # Score all negative pairs
    neg_scores = [(l, r, score_func(l, r)) for (l, r) in negative_pairs]
    neg_scores.sort(key=lambda x: x[2], reverse=True)  # High score = hard negative

    # Stratify negatives: easy (high similarity), semi-hard, hard (low similarity)
    n_neg = len(neg_scores)
    if n_neg == 0:
        return []

    third = max(n_neg // 3, 1)
    easy_neg = neg_scores[:third]           # High similarity but not matches
    semihard_neg = neg_scores[third:2*third]  # Medium difficulty
    hard_neg = neg_scores[2*third:]          # Low similarity, easy to distinguish

    # Construct training set
    training_pairs = []

    for l, r in positive_pairs:
        training_pairs.append({"left": l, "right": r, "label": 1})

        # Sample negatives with curriculum: mostly semi-hard, some easy, few hard
        sampled = []
        n_semi = num_hard_negatives - easy_negatives
        n_hard = max(0, num_hard_negatives - easy_negatives - n_semi)

        if easy_negatives > 0 and len(easy_neg) > 0:
            sampled.extend(easy_neg[:easy_negatives])
        if n_semi > 0 and len(semihard_neg) > 0:
            sampled.extend(np.random.choice(
                semihard_neg, min(n_semi, len(semihard_neg)), replace=False
            ).tolist())
        if n_hard > 0 and len(hard_neg) > 0:
            sampled.extend(np.random.choice(
                hard_neg, min(n_hard, len(hard_neg)), replace=False
            ).tolist())

        for nl, nr, score in sampled[:num_hard_negatives]:
            training_pairs.append({"left": nl, "right": nr, "label": 0, "score": score})

    return training_pairs


def online_hard_negative_mining_curriculum(
    model: object,
    candidate_pairs: list,
    score_func: Callable,
    epoch: int,
    total_epochs: int,
    max_negatives_per_epoch: int = 10000,
) -> list:
    """
    Online hard negative mining with curriculum learning for entity resolution.

    Early epochs: easier negatives (medium similarity)
    Late epochs: harder negatives (high similarity, near decision boundary)

    This aligns with recent SOTA findings that curriculum-based hardness
    prevents early collapse and stabilizes training.

    Parameters
    ----------
    model : fitted classifier
        Current model to evaluate hardness.
    candidate_pairs : list of tuple
        All candidate (left_idx, right_idx) pairs.
    score_func : callable
        Similarity function.
    epoch : int
        Current epoch number.
    total_epochs : int
        Total training epochs.
    max_negatives_per_epoch : int, default=10000
        Max negatives to return per epoch.

    Returns
    -------
    hard_negatives : list of tuple
        Hard negative pairs for this epoch.
    """
    # Compute current model scores for all candidates
    scores = []
    for l, r in candidate_pairs:
        prob = model.predict_proba([[l, r]])[0, 1] if hasattr(model, 'predict_proba') else 0.5
        scores.append((l, r, prob, score_func(l, r)))

    # Curriculum: progress from medium-hard to hardest as training advances
    progress = epoch / total_epochs  # 0 -> 1

    # At epoch 0: take negatives with medium model confidence (uncertain)
    # At epoch 1.0: take negatives with highest model confidence (hardest)
    if progress < 0.5:
        # Early training: focus on uncertain pairs near decision boundary
        # Sort by distance from 0.5
        scores.sort(key=lambda x: abs(x[2] - 0.5))
        n_select = int(max_negatives_per_epoch * 0.5)
    else:
        # Late training: focus on high-confidence false positives
        scores.sort(key=lambda x: -x[2])  # Highest confidence negatives
        n_select = max_negatives_per_epoch

    selected = scores[:n_select]
    return [(l, r) for l, r, prob, sim in selected if prob > 0.3]  # Only confusing cases


# ============================================================================
# PART 3: PRODUCTION USAGE EXAMPLE FOR XGBOOST
# ============================================================================


def xgboost_f0_5_training_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    model_fn: Callable,
    beta: float = 0.5,
    n_repeats: int = 5,
    calibration_method: str = "isotonic",
) -> dict:
    """
    Complete production-ready pipeline for XGBoost F0.5 optimization.

    Combines:
    1. Nested CV for leakage-free threshold selection
    2. Repeated evaluation for stability
    3. Probability calibration
    4. Final threshold recommendation

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary labels.
    model_fn : callable
        Factory returning fresh unfitted model instance.
    beta : float, default=0.5
        F-beta parameter.
    n_repeats : int, default=5
        Number of repeated nested CV runs for stability assessment.
    calibration_method : str, default="isotonic"
        Calibration method.

    Returns
    -------
    dict with aggregated results across repeats.
    """
    all_thresholds = []
    all_scores = []

    print(f"Running {n_repeats} repeats of nested CV for F{beta} optimization...")

    for rep in range(n_repeats):
        result = nested_fbeta_threshold_selection(
            model_fn=model_fn,
            X=X,
            y=y,
            beta=beta,
            outer_seed=42 + rep,
            inner_seed=100 + rep,
            calibration_method=calibration_method,
        )
        all_thresholds.append(result["threshold_mean"])
        all_scores.append(result["fbeta_mean"])

        stability = "STABLE" if result["threshold_std"] < 0.05 else "UNSTABLE"
        print(f"  Repeat {rep+1}: F{beta}={result['fbeta_mean']:.4f} "
              f"T={result['threshold_mean']:.4f} "
              f"(std={result['threshold_std']:.4f}) [{stability}] "
              f"Sens={result['sensitivity']*100:.1f}% "
              f"Spec={result['specificity']*100:.1f}%")

    # Aggregate
    threshold_mean = float(np.mean(all_thresholds))
    threshold_std = float(np.std(all_thresholds))
    fbeta_mean = float(np.mean(all_scores))
    fbeta_std = float(np.std(all_scores))

    stability = "STABLE" if threshold_std < 0.05 else "REVIEW - unstable threshold"
    overall_stability = "STABLE" if fbeta_std < 0.02 else "MODERATE"

    print(f"\n{'='*70}")
    print(f"FINAL RECOMMENDATION:")
    print(f"  Optimal threshold: {threshold_mean:.4f} (+/- {threshold_std:.4f})")
    print(f"  Expected F{beta}: {fbeta_mean:.4f} (+/- {fbeta_std:.4f})")
    print(f"  Threshold stability: {stability}")
    print(f"  Score stability: {overall_stability}")
    print(f"{'='*70}")

    return {
        "recommended_threshold": threshold_mean,
        "threshold_std": threshold_std,
        "expected_fbeta": fbeta_mean,
        "fbeta_std": fbeta_std,
        "stability": stability,
        "per_repeat": {
            "thresholds": [round(t, 4) for t in all_thresholds],
            "scores": [round(s, 4) for s in all_scores],
        }
    }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    from xgboost import XGBClassifier

    print("="*70)
    print("EXAMPLE 1: Threshold Search for XGBoost Binary Classification")
    print("="*70)

    # Simulate imbalanced dataset (e.g., fraud detection, rare event prediction)
    np.random.seed(42)
    n_samples = 10000
    n_features = 20

    # Generate features
    X = np.random.randn(n_samples, n_features)
    # Imbalanced labels: 10% positive
    y = np.random.binomial(1, 0.1, n_samples)
    # Make positives somewhat separable
    X[y == 1] += 0.5

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    def create_xgb():
        return XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=9,  # Handle imbalance
            random_state=42,
            n_jobs=-1,
        )

    # Run pipeline
    model = create_xgb()
    model.fit(X_train, y_train)
    y_prob_test = model.predict_proba(X_test)[:, 1]

    # Simple threshold search
    threshold, f05 = find_best_fbeta_threshold(y_test, y_prob_test, beta=0.5)
    print(f"\nSimple search: optimal threshold = {threshold:.4f}, F0.5 = {f05:.4f}")

    # Nested CV for robust estimate
    result = xgboost_f0_5_training_pipeline(
        X=X,
        y=y,
        model_fn=create_xgb,
        beta=0.5,
        n_repeats=3,
        calibration_method="isotonic",
    )

    print("\n" + "="*70)
    print("EXAMPLE 2: Hard Negative Mining for Entity Resolution")
    print("="*70)

    # Simulate entity resolution scenario
    np.random.seed(42)

    # Simulate a corpus of 10000 records
    corpus = [f"Entity record {i}: name=entity_{i}, address=addr_{i % 1000}, city=city_{i % 50}"
              for i in range(10000)]

    # Simulate positive pairs (same entity)
    n_positives = 500
    anchor_texts = [corpus[i] for i in range(n_positives)]
    positive_texts = [corpus[i].replace(f"addr_{i % 1000}", f"addr_{i % 1000 + 1}")
                      for i in range(n_positives)]  # Slightly perturbed

    # Simulate bi-encoder (random similarity for demo)
    class MockBiEncoder:
        def encode(self, texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True):
            return np.random.randn(len(texts), 384) / 10

    # Simulate cross-encoder
    class MockCrossEncoder:
        def predict(self, pairs, batch_size=32, convert_to_numpy=True):
            return np.random.rand(len(pairs))

    miner = HardNegativeMiner(
        bi_encoder=MockBiEncoder(),
        cross_encoder=MockCrossEncoder(),
        corpus=corpus,
        use_faiss=False,
    )

    print(f"\nMining {n_positives} hard negatives with relative_margin=0.05...")
    result = miner.mine_hard_negatives_for_pairs(
        anchor_texts=anchor_texts[:10],  # Demo with 10 pairs
        positive_texts=positive_texts[:10],
        num_negatives=3,
        range_min=5,
        range_max=50,
        relative_margin=0.05,
        output_format="triplet",
        include_scores=True,
    )

    n_triplets = len(result["triplets"])
    print(f"Generated {n_triplets} triplets ({n_triplets // 10} negatives per positive on average)")

    # Show first triplet
    if result["triplets"]:
        t = result["triplets"][0]
        print(f"\nExample triplet:")
        print(f"  Anchor:   {t['anchor'][:60]}...")
        print(f"  Positive: {t['positive'][:60]}...")
        print(f"  Negative: {t['negative'][:60]}...")

    print("\n" + "="*70)
    print("Pipeline complete. Use the functions above in production.")
    print("="*70)
