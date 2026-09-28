import unittest
from collections import Counter
from rapidfuzz import fuzz
from src.blocking.blocking_v2 import (
    normalize,
    get_tokens,
    extract_keys_v2,
    STOPWORDS,
)


class TestBlockingV2(unittest.TestCase):
    def test_normalize(self):
        self.assertEqual(normalize("Acme Corp., LLC!"), "acmecorpllc")
        self.assertEqual(normalize("123 Main St. #4B"), "123mainst4b")
        self.assertEqual(normalize(""), "")
        self.assertEqual(normalize(None), "")

    def test_get_tokens(self):
        tokens = get_tokens("Google Cloud Platform, Inc.")
        self.assertIn("google", tokens)
        self.assertIn("cloud", tokens)
        self.assertIn("platform", tokens)
        self.assertIn("inc", tokens)

    def test_extract_keys_v2(self):
        row = {
            "business_name": "International Business Machines Corp",
            "business_address": "1 New Orchard Rd, Armonk, NY",
            "country": "US",
        }
        keys = extract_keys_v2(row)
        self.assertGreaterEqual(len(keys), 3)
        self.assertTrue(any(k.startswith("us_n6_") for k in keys))
        self.assertTrue(any(k.startswith("us_a6_") for k in keys))
        self.assertTrue(any(k.startswith("us_n3a3_") for k in keys))
        self.assertTrue(any(k.startswith("us_t1_") for k in keys))

    def test_extract_keys_stopwords_handling(self):
        row = {
            "business_name": "The Global Logistics LLC",
            "business_address": "45 Industrial Pkwy",
            "country": "GB",
        }
        keys = extract_keys_v2(row)
        # "The" is in STOPWORDS, so first meaningful token should be "global"
        self.assertTrue(any(k.startswith("gb_t1_global") for k in keys))

    def test_extract_keys_empty_fields(self):
        row = {"business_name": "", "business_address": "", "country": ""}
        keys = extract_keys_v2(row)
        self.assertIsInstance(keys, list)
        self.assertEqual(len(keys), 0)

    def test_inverted_index_matching_and_ranking(self):
        # Simulate index
        target_records = [
            {"entity_id": "target_1", "business_name": "Apple Store SoHo", "business_address": "103 Prince St, New York", "country": "US"},
            {"entity_id": "target_2", "business_name": "Apple Inc. Headquarters", "business_address": "1 Infinite Loop, Cupertino", "country": "US"},
            {"entity_id": "target_3", "business_name": "Orange Groceries", "business_address": "500 Market St", "country": "US"},
        ]
        index = {}
        target_names = {}
        for rec in target_records:
            eid = rec["entity_id"]
            target_names[eid] = rec["business_name"]
            for k in extract_keys_v2(rec):
                index.setdefault(k, []).append(eid)

        query = {"business_name": "Apple Store NYC", "business_address": "103 Prince Street, NY", "country": "US"}
        cand_counts = Counter()
        for k in extract_keys_v2(query):
            for target_id in index.get(k, []):
                cand_counts[target_id] += 1

        self.assertIn("target_1", cand_counts)
        self.assertIn("target_2", cand_counts)
        self.assertNotIn("target_3", cand_counts)

        # target_1 has both name and address matches with query, so it should rank higher than target_2
        self.assertGreaterEqual(cand_counts["target_1"], cand_counts["target_2"])


if __name__ == "__main__":
    unittest.main()
