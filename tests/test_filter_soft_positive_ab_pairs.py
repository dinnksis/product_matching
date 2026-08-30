from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts/filter_soft_positive_ab_pairs.py"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "filter_soft_positive_ab_pairs_test_module", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


filterer = load_script()


BASE_POLICY = {
    **filterer.DEFAULT_POLICY,
    "minimum_quality_score": 0.0,
    "maximum_per_category": 100,
    "maximum_per_rule": 2,
    "maximum_per_rule_transition": 1,
    "maximum_per_category_product_type": 100,
}


def candidate(
    candidate_key: str,
    *,
    quality: float = 0.90,
    category: str = "Аптека",
    product_type: str = "лубрикант",
    rule_id: str = "rule-a",
    transition: str = "75 г|100 г",
    global_a: str | None = None,
    global_b: str | None = None,
    task_index: int = 1,
    evidence_type: str = "source_exact_transition",
) -> dict[str, object]:
    """A fully selection- and decision-frame-compatible production row."""

    return {
        "candidate_key": candidate_key,
        "source_tier": "A",
        "source_run_signature": "a" * 64,
        "source_task_index": task_index,
        "source_id1": -task_index * 2,
        "source_id2": -task_index * 2 - 1,
        "duplicate_sources": [{"candidate_key": candidate_key}],
        "pair_content_sha256": hashlib.sha256(candidate_key.encode()).hexdigest(),
        "category": category,
        "product_type": product_type,
        "product_type_normalized": filterer.normalize_text(product_type),
        "rule_id": rule_id,
        "source_rule_id": f"source-{rule_id}",
        "concept": "weight",
        "relation": "different_value",
        "required_attribute_key": "Вес изделия",
        "original_value": "75 г",
        "new_value": "100 г",
        "transition_key": transition,
        "rule_probability": 0.90,
        "rule_support": 8,
        "rule_singleton_support": 2,
        "cross_split_p80": True,
        "category_values_grounded": True,
        "human_category_original_support": 4,
        "human_category_new_support": 3,
        "human_scope_original_support": 2,
        "human_scope_new_support": 1,
        "evidence_type": evidence_type,
        "evidence_source": "rule_source_examples",
        "evidence_value": transition,
        "minimum_title_tokens": 5,
        "name_similarity": 0.80,
        "brand_context_grounded": True,
        "score_ab": 0.13,
        "score_ba": 0.11,
        "baseline_score": 0.12,
        "score_order_gap": 0.02,
        "quality_score": quality,
        "forbidden_ood_category": False,
        "validation_fact_overlap": False,
        "global_card_key_a": global_a or f"card-a-{candidate_key}",
        "global_card_key_b": global_b or f"card-b-{candidate_key}",
    }


def write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class FilterSoftPositiveABPairsTest(unittest.TestCase):
    def test_181_style_exact_content_duplicate_ignores_ids_and_json_order(self) -> None:
        first = {
            "id": -26464,
            "name": "лубрикант durex 75 г",
            "attributes": (
                '{"Тип товара":"лубрикант","Бренд":"durex",'
                '"Вес изделия":"75 г"}'
            ),
            "category": "Аптека",
        }
        second = {
            "id": -61308,
            "name": "лубрикант durex 100 г",
            "attributes": (
                '{"Тип товара":"лубрикант","Бренд":"durex",'
                '"Вес изделия":"100 г"}'
            ),
            "category": "Аптека",
        }
        duplicate_first = {
            "id": -19272,
            "name": "лубрикант durex 75 г",
            "attributes": (
                '{"Вес изделия":"75 г","Бренд":"durex",'
                '"Тип товара":"лубрикант"}'
            ),
            "category": "Аптека",
        }
        duplicate_second = {
            "id": -67890,
            "name": "лубрикант durex 100 г",
            "attributes": (
                '{"Бренд":"durex","Вес изделия":"100 г",'
                '"Тип товара":"лубрикант"}'
            ),
            "category": "Аптека",
        }

        self.assertEqual(
            filterer.canonical_card_key(first),
            filterer.canonical_card_key(duplicate_first),
        )
        pair_key = filterer.canonical_pair_key(first, second)
        self.assertEqual(pair_key, filterer.canonical_pair_key(second, first))
        self.assertEqual(
            pair_key,
            filterer.canonical_pair_key(duplicate_second, duplicate_first),
        )

        # Production dedup is deliberately exact. Surface changes are not
        # folded into the verified 181 cross-run overlaps.
        changed_case = dict(duplicate_first, name="лубрикант Durex 75 г")
        changed_space = dict(duplicate_first, name="лубрикант  durex 75 г")
        self.assertNotEqual(
            filterer.canonical_card_key(first),
            filterer.canonical_card_key(changed_case),
        )
        self.assertNotEqual(
            filterer.canonical_card_key(first),
            filterer.canonical_card_key(changed_space),
        )

    def test_score_uses_production_fields_and_prefers_strong_evidence(self) -> None:
        strong = candidate("strong")
        strong.pop("quality_score")
        weak = candidate("weak", evidence_type="human_category_attribute_values")
        weak.update(
            {
                "rule_probability": 0.80,
                "rule_support": 3,
                "rule_singleton_support": 0,
                "cross_split_p80": False,
                "minimum_title_tokens": 4,
                "name_similarity": 0.99,
                "brand_context_grounded": False,
                "baseline_score": 0.75,
            }
        )
        weak.pop("quality_score")

        strong_score = filterer.score_candidate(strong, BASE_POLICY)
        weak_score = filterer.score_candidate(weak, BASE_POLICY)
        for score in (strong_score, weak_score):
            self.assertTrue(math.isfinite(score))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
        self.assertGreater(strong_score, weak_score)

    def test_selection_returns_reasons_and_enforces_rule_transition_caps(self) -> None:
        rows = [
            candidate("best", quality=0.99, task_index=1),
            candidate("same-transition", quality=0.98, task_index=2),
            candidate(
                "second-transition", quality=0.97, transition="50 г|60 г", task_index=3
            ),
            candidate(
                "third-transition", quality=0.96, transition="20 г|30 г", task_index=4
            ),
        ]

        selected, reasons = filterer.select_candidates(rows, BASE_POLICY)
        reversed_selected, reversed_reasons = filterer.select_candidates(
            list(reversed(rows)), BASE_POLICY
        )
        self.assertEqual(
            [row["candidate_key"] for row in selected],
            ["best", "second-transition"],
        )
        self.assertEqual(
            [row["candidate_key"] for row in selected],
            [row["candidate_key"] for row in reversed_selected],
        )
        self.assertEqual(reasons, reversed_reasons)
        self.assertEqual(reasons["best"], [])
        self.assertEqual(reasons["second-transition"], [])
        self.assertEqual(reasons["same-transition"], ["rule_transition_cap_reached"])
        self.assertEqual(reasons["third-transition"], ["rule_cap_reached"])

        decisions = filterer._candidate_decisions_frame(rows, reasons)
        reason_by_key = decisions.set_index("candidate_key")["rejection_reasons"].to_dict()
        self.assertEqual(reason_by_key["best"], "[]")
        self.assertEqual(
            reason_by_key["same-transition"], '["rule_transition_cap_reached"]'
        )
        self.assertTrue(
            decisions.loc[decisions["candidate_key"] == "best", "selected"].item()
        )

    def test_selection_enforces_profile_category_and_global_card_caps(self) -> None:
        profile_policy = {
            **BASE_POLICY,
            "maximum_per_category_product_type": 2,
            "maximum_per_rule": 20,
            "maximum_per_rule_transition": 20,
        }
        profile_rows = [
            candidate(
                f"profile-{index}",
                quality=0.99 - index / 100,
                rule_id=f"r{index}",
                transition=f"t{index}",
                task_index=index,
            )
            for index in (1, 2, 3)
        ]
        selected, reasons = filterer.select_candidates(profile_rows, profile_policy)
        self.assertEqual(
            [row["candidate_key"] for row in selected], ["profile-1", "profile-2"]
        )
        self.assertEqual(
            reasons["profile-3"], ["category_product_type_cap_reached"]
        )

        category_policy = {
            **profile_policy,
            "maximum_per_category": 2,
            "maximum_per_category_product_type": 20,
        }
        category_rows = [
            candidate(
                f"category-{index}",
                quality=0.99 - index / 100,
                product_type=f"type-{index}",
                rule_id=f"category-rule-{index}",
                transition=f"category-transition-{index}",
                task_index=index,
            )
            for index in (1, 2, 3)
        ]
        selected, reasons = filterer.select_candidates(category_rows, category_policy)
        self.assertEqual(
            [row["candidate_key"] for row in selected], ["category-1", "category-2"]
        )
        self.assertEqual(reasons["category-3"], ["category_cap_reached"])

        reuse_rows = [
            candidate(
                "reuse-first", quality=0.99, global_a="shared-card", task_index=1
            ),
            candidate(
                "reuse-second",
                quality=0.98,
                category="Дом и сад",
                product_type="ваза",
                rule_id="other-rule",
                transition="other-transition",
                global_a="shared-card",
                task_index=2,
            ),
        ]
        selected, reasons = filterer.select_candidates(reuse_rows, profile_policy)
        self.assertEqual([row["candidate_key"] for row in selected], ["reuse-first"])
        self.assertEqual(reasons["reuse-second"], ["global_card_reuse"])

    def test_exact_weight_stats_use_ordered_little_endian_float64(self) -> None:
        weights = [0.31, 0.47, 0.83]
        stats = filterer.exact_weight_stats(weights)
        expected = np.asarray(weights, dtype="<f8")
        self.assertEqual(stats["version"], filterer.WEIGHT_STATS_VERSION)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["sum"], float(expected.sum(dtype=np.float64)))
        self.assertEqual(stats["min"], min(weights))
        self.assertEqual(stats["max"], max(weights))
        self.assertEqual(
            stats["sha256"], hashlib.sha256(expected.tobytes(order="C")).hexdigest()
        )
        self.assertNotEqual(
            stats["sha256"],
            filterer.exact_weight_stats(list(reversed(weights)))["sha256"],
        )
        for invalid in ([0.5, 0.0], [0.5, float("nan")], []):
            with self.subTest(invalid=invalid), self.assertRaises(filterer.FilterError):
                filterer.exact_weight_stats(invalid)

    def test_validate_artifact_rejects_a_post_manifest_file_tamper_early(self) -> None:
        """Exercise the hash gate without constructing 31,176 decision rows."""

        with tempfile.TemporaryDirectory(prefix="filter-soft-positive-") as raw:
            output = Path(raw)
            signature_payload = {"fixture": "post-manifest-hash-gate"}
            common = {
                "contract_version": filterer.CONTRACT_VERSION,
                "label_source": filterer.LABEL_SOURCE,
                "pair_count": 1,
                "target_counts": {"0": 0, "1": 1},
                "sample_weight_stats": filterer.exact_weight_stats([0.5]),
                "run_signature": filterer.sha256_json(signature_payload),
            }
            documents = {
                "summary.json": {**common, "signature_payload": signature_payload},
                "selection_report.json": dict(common),
                "validation_report.json": {
                    **common,
                    "valid": True,
                    "validation_fact_overlap_count": 0,
                    "forbidden_ood_categories": [],
                },
            }
            for name, value in documents.items():
                write_json(output / name, value)
            parquet_names = (
                "items.parquet",
                "pairs.parquet",
                "pair_generation_metadata.parquet",
                "candidate_decisions.parquet",
            )
            for name in parquet_names:
                (output / name).write_bytes(f"fixture:{name}".encode())

            manifest_files = {}
            for name in (*documents, *parquet_names):
                path = output / name
                manifest_files[name] = {
                    "bytes": path.stat().st_size,
                    "sha256": filterer.sha256_file(path),
                }
            write_json(
                output / "build_manifest.json",
                {
                    **common,
                    "signature_payload": signature_payload,
                    "files": manifest_files,
                },
            )

            with (output / "pairs.parquet").open("ab") as stream:
                stream.write(b":tampered")
            with self.assertRaisesRegex(
                filterer.FilterError, "manifest hash mismatch for pairs.parquet"
            ):
                filterer.validate_artifact(output)


if __name__ == "__main__":
    unittest.main()
