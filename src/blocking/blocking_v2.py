"""
Amazon ML Challenge 2026 — V2 Blocking Script

Implements the V2 blocking pipeline:
  Stage 1: Multi-key exact blocking with 5 keys per entity
  Stage 2: Optional TF-IDF cosine similarity re-ranking to keep top-K candidates

Outputs:
  v2_train_candidates.tsv
  v2_test_candidates.tsv

Memory target: <= 16GB RAM on full dataset
"""

import re
import argparse
import os
import sys
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


csv.field_size_limit(sys.maxsize)


def normalize(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def extract_keys_v2(row: dict) -> list[str]:
    name = normalize(row.get("business_name", ""))
    addr = normalize(row.get("business_address", ""))
    country = normalize(row.get("country", ""))
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


def build_inverted_index_v2(filepaths: list[str], index_path: str | None = None) -> dict:
    if index_path and os.path.exists(index_path):
        print(f"[blocking] Loading cached index from {index_path}")
        with open(index_path, "r", encoding="utf-8") as f:
            return defaultdict(list, json.load(f))

    print(f"[blocking] Building inverted index from {len(filepaths)} file(s)...")
    index: dict[str, list[str]] = defaultdict(list)
    for path in filepaths:
        print(f"[blocking]   Indexing {os.path.basename(path)}...")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row.get("entity_id", "")
                if not eid:
                    continue
                for key in extract_keys_v2(row):
                    index[key].append(eid)

    print(f"[blocking] Total unique blocking keys: {len(index):,}")

    if index_path:
        print(f"[blocking] Caching index to {index_path}")
        os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(dict(index), f)

    return index


def generate_candidates_v2(
    data_dir: str,
    out_file: str,
    split: str = "train",
    max_candidates: int = 50,
    index_path: str | None = None,
    validate_recall: bool = False,
    gt_file: str | None = None,
) -> dict:
    s1_path = os.path.join(data_dir, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{split}_source3.tsv")

    s23_index = build_inverted_index_v2([s2_path, s3_path], index_path)

    positive_pairs: set[tuple[str, str]] = set()
    if validate_recall and gt_file:
        print("[blocking] Recall validation enabled")
        with open(gt_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                s1_id = row["source1_entity_id"]
                matched = str(row["matched_entity_ids"]).split(",")
                for m in matched:
                    m = m.strip()
                    if m and m != "nan":
                        positive_pairs.add((s1_id, m))

    print("[blocking] Generating candidate pairs for Source 1...")
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    stats = {
        "total_s1": 0,
        "with_candidates": 0,
        "candidate_counts": [],
        "recall_hits": 0,
        "skipped_empty": 0,
    }

    with open(s1_path, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fout.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in tqdm(reader):
            s1_id = row.get("entity_id", "")
            if not s1_id:
                continue

            stats["total_s1"] += 1
            candidates: set[str] = set()
            for key in extract_keys_v2(row):
                candidates.update(s23_index.get(key, []))
            candidates.discard(s1_id)

            if candidates:
                stats["with_candidates"] += 1
                stats["candidate_counts"].append(len(candidates))
            else:
                stats["skipped_empty"] += 1

            if validate_recall and gt_file:
                stats["recall_hits"] += sum(1 for c in candidates if (s1_id, c) in positive_pairs)

            candidates_list = list(candidates)
            if len(candidates_list) > max_candidates:
                candidates_list = candidates_list[:max_candidates]

            fout.write(f"{s1_id}\t{','.join(candidates_list)}\n")

    print("\n[blocking] Blocking Statistics")
    print(f"  Total S1 entities         : {stats['total_s1']:,}")
    print(f"  S1 with candidates        : {stats['with_candidates']:,}")
    print(f"  S1 with no candidates     : {stats['skipped_empty']:,}")
    if stats["candidate_counts"]:
        print(f"  Avg candidates/S1         : {np.mean(stats['candidate_counts']):.1f}")
        print(f"  Median candidates/S1      : {np.median(stats['candidate_counts']):.1f}")
        print(f"  Max candidates/S1         : {max(stats['candidate_counts']):,}")
    if validate_recall and gt_file and positive_pairs:
        total_positives = len(positive_pairs)
        recall = stats["recall_hits"] / total_positives if total_positives > 0 else 0.0
        print(f"  Recall on ground truth    : {recall:.4f} ({stats['recall_hits']}/{total_positives})")

    print(f"[blocking] Complete! Candidates saved to {out_file}")
    return stats


def generate_candidates_two_stage(
    data_dir: str,
    out_file: str,
    split: str = "train",
    max_candidates: int = 50,
    index_path: str | None = None,
) -> None:
    s1_path = os.path.join(data_dir, f"{split}_source1.tsv")
    s2_path = os.path.join(data_dir, f"{split}_source2.tsv")
    s3_path = os.path.join(data_dir, f"{split}_source3.tsv")

    print("[blocking][two-stage] Stage 1: exact-key blocking...")
    s23_index = build_inverted_index_v2([s2_path, s3_path], index_path)

    s23_ids: list[str] = []
    s23_texts: list[str] = []
    for path in [s2_path, s3_path]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                eid = row.get("entity_id", "")
                if eid:
                    s23_ids.append(eid)
                    s23_texts.append(
                        f"{row.get('business_name', '')} {row.get('business_address', '')}"
                    )

    print("[blocking][two-stage] Stage 2: fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), max_features=50000)
    tfidf.fit(s23_texts)
    s23_matrix = tfidf.transform(s23_texts)
    id_to_idx = {eid: i for i, eid in enumerate(s23_ids)}

    print("[blocking][two-stage] Stage 3: generating and re-ranking candidates...")
    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)

    with open(s1_path, "r", encoding="utf-8") as fin, open(out_file, "w", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter="\t")
        fout.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in tqdm(reader):
            s1_id = row.get("entity_id", "")
            if not s1_id:
                continue

            candidates: set[str] = set()
            for key in extract_keys_v2(row):
                candidates.update(s23_index.get(key, []))
            candidates.discard(s1_id)

            if not candidates:
                fout.write(f"{s1_id}\t\n")
                continue

            candidates_list = list(candidates)
            if len(candidates_list) > max_candidates:
                s1_text = f"{row.get('business_name', '')} {row.get('business_address', '')}"
                s1_vec = tfidf.transform([s1_text])

                candidate_indices = [id_to_idx.get(c) for c in candidates_list if c in id_to_idx]
                if candidate_indices:
                    cand_matrix = s23_matrix[candidate_indices]
                    sims = cosine_similarity(s1_vec, cand_matrix).flatten()

                    top_indices = np.argsort(sims)[-max_candidates:][::-1]
                    candidates_list = [candidates_list[i] for i in top_indices if i < len(candidates_list)]

            fout.write(f"{s1_id}\t{','.join(candidates_list)}\n")

    print(f"[blocking][two-stage] Complete! Candidates saved to {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 — V2 Blocking")
    parser.add_argument("--data_dir", type=str, default="dataset/train", help="Path to split TSV directory")
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--out_file", type=str, default=None, help="Output candidate TSV path")
    parser.add_argument("--max_candidates", type=int, default=50, help="Max candidates per S1")
    parser.add_argument("--index_path", type=str, default=None, help="Path to cache the inverted index JSON")
    parser.add_argument("--validate_recall", action="store_true", help="Enable recall validation against ground truth")
    parser.add_argument("--gt_file", type=str, default=None, help="Ground truth TSV path for recall validation")
    parser.add_argument(
        "--mode",
        type=str,
        default="one_stage",
        choices=["one_stage", "two_stage"],
        help="Blocking mode: similarity-truncated exact keys or TF-IDF re-ranking",
    )
    args = parser.parse_args()

    split = args.split
    data_dir = args.data_dir
    out_file = args.out_file or os.path.join("output", f"v2_{split}_candidates.tsv")
    index_path = args.index_path or os.path.join("cache", f"v2_{split}_index.json")

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(index_path) or ".", exist_ok=True)

    gt_file = args.gt_file or os.path.join(data_dir, "train_ground_truth.tsv")

    start = time.time()
    if args.mode == "two_stage":
        generate_candidates_two_stage(
            data_dir=data_dir,
            out_file=out_file,
            split=split,
            max_candidates=args.max_candidates,
            index_path=index_path,
        )
    else:
        generate_candidates_v2(
            data_dir=data_dir,
            out_file=out_file,
            split=split,
            max_candidates=args.max_candidates,
            index_path=index_path,
            validate_recall=args.validate_recall,
            gt_file=gt_file if args.validate_recall else None,
        )
    end = time.time()

    print(f"[blocking] Elapsed: {end - start:.2f}s")


if __name__ == "__main__":
    main()
