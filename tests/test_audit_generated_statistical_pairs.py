from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_generated_statistical_pairs.py"


def load_auditor():
    spec = importlib.util.spec_from_file_location(
        "audit_generated_statistical_pairs", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load generated-pair auditor")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GeneratedStatisticalPairAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.auditor = load_auditor()

    def build_fixture(self, root: Path) -> tuple[Path, Path]:
        source = root / "active-source"
        source.mkdir()
        category = "Обувь"
        material_forward = [
            {
                "generation_rule_id": "r0",
                "concept": "material",
                "attribute_key": "Материал",
                "original_value": "хлопок",
                "new_value": "100% хлопок",
            }
        ]
        material_reverse = [
            {
                "generation_rule_id": "r0",
                "concept": "material",
                "attribute_key": "Материал",
                "original_value": "100% хлопок",
                "new_value": "хлопок",
            }
        ]
        two_rules = [
            {
                "generation_rule_id": "r2",
                "concept": "color",
                "attribute_key": "Цвет",
                "original_value": "красный",
                "new_value": "синий",
            },
            {
                "generation_rule_id": "r3",
                "concept": "size",
                "attribute_key": "Размер",
                "original_value": "40",
                "new_value": "41",
            },
        ]
        applications = [
            material_forward,
            material_reverse,
            material_forward,
            two_rules,
        ]
        product_types = ["кеды", "кеды", "кеды", "ботинки"]
        rule_ids = [["r0"], ["r0"], ["r1"], ["r2", "r3"]]
        scheduled_rule_ids = [["r0"], ["r0"], ["r1"], ["rx", "r3"]]
        profiles = [["p0"], ["p0"], ["p1"], ["px", "p3"]]
        metadata_rows = []
        run_seed = 17
        for task in range(4):
            signature = self.auditor._semantic_pair_signature(
                category, product_types[task], applications[task]
            )
            source_style_id = 100 + task
            pair_task = self.auditor.PairGenerationTask(
                task_index=task,
                mutated_id=-1 - task,
                seed=int(
                    self.auditor.stable_hash64(run_seed, source_style_id)
                    % (2**31 - 1)
                ),
                anchor={"id": source_style_id},
            )
            nonce_common = {
                "task_seed_offset": 0,
                "task_retry_round": 0,
                "selection_attempt": 1,
            }
            metadata_rows.append(
                {
                    "task_index": task,
                    "id1": source_style_id,
                    "id2": -1 - task,
                    "source_style_id": source_style_id,
                    "target": 0,
                    "category": category,
                    "product_type": product_types[task],
                    "applications_json": json.dumps(
                        applications[task], ensure_ascii=False
                    ),
                    "semantic_signature": "tampered" if task == 2 else signature,
                    "semantic_signature_version": (
                        self.auditor.SEMANTIC_SIGNATURE_VERSION
                    ),
                    "attempt_diversity_version": (
                        self.auditor.ATTEMPT_DIVERSITY_VERSION
                    ),
                    **nonce_common,
                    "anchor_attempt": 1,
                    "mutation_attempt": 1,
                    "pair_attempts_config": 2,
                    "anchor_attempts_config": 2,
                    "mutation_attempts_config": 2,
                    "anchor_diversity_nonce_sha256": self.auditor.hashlib.sha256(
                        self.auditor._attempt_diversity_nonce(
                            pair_task,
                            **nonce_common,
                            stage="anchor",
                            stage_attempt=1,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "mutation_diversity_nonce_sha256": self.auditor.hashlib.sha256(
                        self.auditor._attempt_diversity_nonce(
                            pair_task,
                            **nonce_common,
                            stage="mutation",
                            stage_attempt=1,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "global_rejection_feedback_count": task,
                    "forbidden_semantic_signature_count": task // 2,
                    "forbidden_card_key_count": task * 2,
                    "rule_ids": json.dumps(rule_ids[task]),
                    "scheduled_rule_ids": json.dumps(scheduled_rule_ids[task]),
                    "scheduled_rule_profile_ids": json.dumps(profiles[task]),
                    "rule_count": len(rule_ids[task]),
                }
            )
        pd.DataFrame(metadata_rows).to_parquet(
            source / "pair_generation_metadata.parquet", index=False
        )

        card_a = {
            "name": "кеды SKU: AB-10 хлопок",
            "attributes": json.dumps(
                {
                    "Тип товара": "кеды",
                    "Артикул": "AB-10",
                    "Модель": "123",
                    "Материал": "хлопок",
                },
                ensure_ascii=False,
            ),
            "category": category,
        }
        card_b = {
            "name": "кеды SKU: AB-10 100% хлопок",
            "attributes": json.dumps(
                {
                    "Тип товара": "кеды",
                    "Артикул": "AB-10",
                    "Модель": "123",
                    "Материал": "100% хлопок",
                },
                ensure_ascii=False,
            ),
            "category": category,
        }
        card_c = {
            "name": "кеды north хлопок",
            "attributes": json.dumps(
                {"Тип товара": "кеды", "Бренд": "north", "Материал": "хлопок"},
                ensure_ascii=False,
            ),
            "category": category,
        }
        card_d = {
            "name": "кеды north 100% хлопок",
            "attributes": json.dumps(
                {
                    "Тип товара": "кеды",
                    "Бренд": "north",
                    "Материал": "100% хлопок",
                },
                ensure_ascii=False,
            ),
            "category": category,
        }
        card_e = {
            "name": "ботинки trail красные размер 40",
            "attributes": json.dumps(
                {
                    "Тип товара": "ботинки",
                    "Бренд": "trail",
                    "Цвет": "красный",
                    "Размер": "40",
                },
                ensure_ascii=False,
            ),
            "category": category,
        }
        card_f = {
            "name": "ботинки trail синие размер 41",
            "attributes": json.dumps(
                {
                    "Тип товара": "ботинки",
                    "Бренд": "trail",
                    "Цвет": "синий",
                    "Размер": "41",
                },
                ensure_ascii=False,
            ),
            "category": category,
        }
        base_cards = [card_a, card_b, card_c, card_e]
        mutated_cards = [card_b, card_a, card_d, card_f]
        base = pd.DataFrame(
            [{"id": 100 + index, **card} for index, card in enumerate(base_cards)]
        )
        mutated = pd.DataFrame(
            [{"id": -1 - index, **card} for index, card in enumerate(mutated_cards)]
        )
        pairs = pd.DataFrame(
            {
                "id1": [100, 101, 102, 103],
                "id2": [-1, -2, -3, -4],
                "target": [0, 0, 0, 0],
            }
        )
        base.to_parquet(source / "base_items.parquet", index=False)
        mutated.to_parquet(source / "mutated_items.parquet", index=False)
        pairs.to_parquet(source / "pairs.parquet", index=False)
        pd.concat([base, mutated], ignore_index=True).to_parquet(
            source / "items.parquet", index=False
        )
        (source / "summary.json").write_text(
            json.dumps(
                {
                    "seed": run_seed,
                    "attempt_diversity_version": (
                        self.auditor.ATTEMPT_DIVERSITY_VERSION
                    ),
                    "semantic_signature_limit": 2,
                    "semantic_signature_unique_count": 2,
                    "semantic_signature_max_count": 3,
                    "rule_catalog_summary": {"selectable_rules": 4},
                }
            ),
            encoding="utf-8",
        )

        reference = root / "human.parquet"
        pd.DataFrame(
            [
                {
                    "id": 1,
                    "name": "длинное человеческое название товара для сравнения",
                    "attributes": json.dumps(
                        {
                            "Тип": "кеды",
                            "Бренд": "human",
                            "Материал": "кожа",
                            "Размер": "41",
                            "Цвет": "черный",
                        },
                        ensure_ascii=False,
                    ),
                    "category": category,
                },
                {
                    "id": 2,
                    "name": "человеческая карточка другой категории",
                    "attributes": json.dumps(
                        {"Тип": "чай", "Бренд": "human", "Вес": "100 г"},
                        ensure_ascii=False,
                    ),
                    "category": "Продукты питания",
                },
            ]
        ).to_parquet(reference, index=False)
        return source, reference

    def test_recomputes_signatures_cap_and_rule_schedule_statistics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-pair-audit-") as raw:
            source, reference = self.build_fixture(Path(raw))
            report = self.auditor.audit_source(
                source, reference_items=reference, sample_limit=10
            )

        semantic = report["semantic_signatures"]
        self.assertEqual(semantic["unique_count"], 2)
        self.assertEqual(semantic["max_count"], 3)
        self.assertEqual(semantic["cap"], 2)
        self.assertFalse(semantic["within_cap"])
        self.assertEqual(semantic["value_count_histogram"], {"1": 1, "3": 1})
        self.assertEqual(
            report["issues"]["semantic_signature_mismatch"][
                "sample_task_indices"
            ],
            [2],
        )
        self.assertEqual(
            report["issues"]["semantic_signature_cap_exceeded"][
                "sample_task_indices"
            ],
            [0, 1, 2],
        )
        rules = report["rules"]
        self.assertEqual(rules["actual"]["counts"]["r0"], 2)
        self.assertEqual(rules["two_rule"]["actual_tasks"], 1)
        self.assertEqual(rules["two_rule"]["actual_fraction"], 0.25)
        self.assertEqual(rules["alignment"]["mismatch_rows"], 1)
        self.assertEqual(rules["alignment"]["application_mismatch_rows"], 1)
        self.assertEqual(
            report["issues"]["scheduled_actual_rule_mismatch"][
                "sample_task_indices"
            ],
            [3],
        )
        self.assertEqual(
            report["issues"]["application_actual_rule_mismatch"][
                "sample_task_indices"
            ],
            [2],
        )
        self.assertEqual(rules["categories"]["counts"], {"Обувь": 4})
        self.assertEqual(rules["product_types"]["counts"], {"ботинки": 1, "кеды": 3})
        attempt = report["attempt_diversity"]
        self.assertTrue(attempt["protocol_consistent"])
        self.assertEqual(attempt["version_match_rows"], 4)
        self.assertEqual(attempt["anchor_nonce_hash_recomputed_rows"], 4)
        self.assertEqual(attempt["mutation_nonce_hash_recomputed_rows"], 4)

    def test_flags_mixed_attempt_protocol_and_tampered_nonce(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-pair-audit-") as raw:
            source, _ = self.build_fixture(Path(raw))
            metadata_path = source / "pair_generation_metadata.parquet"
            metadata = pd.read_parquet(metadata_path)
            metadata.loc[0, "attempt_diversity_version"] = "legacy"
            metadata.loc[1, "anchor_diversity_nonce_sha256"] = "0" * 64
            metadata.loc[2, "mutation_diversity_nonce_sha256"] = "invalid"
            metadata.to_parquet(metadata_path, index=False)
            report = self.auditor.audit_source(
                source, reference_items=None, sample_limit=10
            )

        attempt = report["attempt_diversity"]
        self.assertFalse(attempt["protocol_consistent"])
        self.assertEqual(
            attempt["observed_versions"],
            sorted([self.auditor.ATTEMPT_DIVERSITY_VERSION, "legacy"]),
        )
        self.assertEqual(
            report["issues"]["attempt_diversity_version_mismatch"][
                "sample_task_indices"
            ],
            [0],
        )
        self.assertEqual(
            report["issues"]["anchor_diversity_nonce_mismatch"][
                "sample_task_indices"
            ],
            [1],
        )
        self.assertEqual(
            report["issues"]["invalid_mutation_diversity_nonce_sha256"][
                "sample_task_indices"
            ],
            [2],
        )

    def test_detects_orderless_duplicates_and_quality_heuristics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-pair-audit-") as raw:
            source, _ = self.build_fixture(Path(raw))
            report = self.auditor.audit_source(
                source, reference_items=None, sample_limit=2
            )

        duplicates = report["duplicates"]
        self.assertEqual(duplicates["duplicate_full_card_groups"], 2)
        self.assertEqual(
            duplicates["duplicate_order_insensitive_card_pair_groups"], 1
        )
        self.assertEqual(
            report["issues"]["duplicate_order_insensitive_card_pair"][
                "sample_task_indices"
            ],
            [0, 1],
        )
        self.assertEqual(report["heuristics"]["sku_or_article_affected_tasks"], 2)
        self.assertEqual(report["heuristics"]["numeric_only_model_affected_tasks"], 2)
        self.assertEqual(
            report["heuristics"]["material_synonym_mutation_affected_tasks"], 3
        )
        self.assertEqual(
            report["issues"]["material_synonym_mutation"]["sample_task_indices"],
            [0, 1],
        )

    def test_cli_writes_outside_source_prints_same_json_and_keeps_source_unchanged(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="generated-pair-audit-") as raw:
            root = Path(raw)
            source, reference = self.build_fixture(root)
            output = root / "reports" / "audit.json"
            before = {
                path.name: path.read_bytes()
                for path in sorted(source.iterdir())
                if path.is_file()
            }
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.auditor.main(
                    [
                        "--source-dir",
                        str(source),
                        "--reference-items",
                        str(reference),
                        "--output",
                        str(output),
                        "--sample-limit",
                        "3",
                    ]
                )
            after = {
                path.name: path.read_bytes()
                for path in sorted(source.iterdir())
                if path.is_file()
            }
            printed = json.loads(stdout.getvalue())
            written = json.loads(output.read_text(encoding="utf-8"))

            self.assertEqual(before, after)
            self.assertEqual(printed, written)
            complexity = printed["complexity"]
            self.assertEqual(complexity["human_reference_all"]["rows"], 2)
            self.assertEqual(
                complexity["human_reference_matching_synthetic_categories"]["rows"],
                1,
            )
            self.assertIsNotNone(
                complexity["synthetic_vs_human_matching_categories"][
                    "attribute_mean_ratio"
                ]
            )

            with self.assertRaisesRegex(ValueError, "outside --source-dir"):
                self.auditor.main(
                    [
                        "--source-dir",
                        str(source),
                        "--reference-items",
                        str(reference),
                        "--output",
                        str(source / "forbidden-report.json"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
