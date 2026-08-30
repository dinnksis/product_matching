from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from item_pipeline.normalization import stable_hash64
from item_pipeline.pair_generate import (
    ATTEMPT_DIVERSITY_VERSION,
    PairGenerationTask,
    SEMANTIC_SIGNATURE_VERSION,
    _attempt_diversity_nonce,
    _semantic_pair_signature,
)
from item_pipeline.rule_schedule import SCHEDULE_VERSION


ROOT = Path(__file__).resolve().parents[1]
FREEZER_PATH = ROOT / "scripts" / "freeze_generated_pair_dataset.py"
UPLOADER_PATH = ROOT / "scripts" / "push_generation_rule_pairs_dataset.py"

EXPECTED_SEMANTIC_PROVENANCE: dict[str, object] = {
    "semantic_signature_retry": True,
    "semantic_signature_version": SEMANTIC_SIGNATURE_VERSION,
    "semantic_signature_limit": 2,
    "semantic_signature_unique_count": 1,
    "semantic_signature_max_count": 2,
    "semantic_signature_retry_events": 4,
    "semantic_signature_retry_events_this_run": 2,
}
EXPECTED_ATTEMPT_PROVENANCE: dict[str, object] = {
    "attempt_diversity_version": ATTEMPT_DIVERSITY_VERSION,
}
EXPECTED_SCHEDULE_PROVENANCE: dict[str, object] = {
    "balanced_rule_schedule": True,
    "profile_capacity_policy_version": "test-capacity-policy-v1",
    "profile_capacity_policy_sha256": "a" * 64,
    "rule_schedule_version": SCHEDULE_VERSION,
    "rule_schedule_sha256": "schedule-sha",
    "rule_schedule_seed": 17,
    "rule_schedule_config": {"two_rule_fraction": 0.5},
    "planned_rule_schedule": {"future_schedule_field": "preserve-me"},
    "scheduled_tasks": 2,
    "eligible_rules": 1,
    "eligible_rule_profiles": 1,
    "primary_rule_usage": {"rule-1": 2},
    "category_task_quotas": {"Электроника": 2},
    "requested_two_rule_tasks": 1,
    "scheduled_two_rule_fraction": 0.5,
    "completed_scheduled_tasks": 2,
    "pending_scheduled_tasks": 0,
    "realized_rule_schedule": {
        "completed_scheduled_tasks": 2,
        "pending_scheduled_tasks": 0,
        "realized_primary_rule_coverage": 1,
        "future_realized_field": "preserve-me-too",
    },
    "realized_primary_rule_coverage": 1,
    "max_identical_scheduled_bundle_count": 1,
}


def load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_source(source_dir: Path) -> dict[str, object]:
    source_dir.mkdir(parents=True)
    item_rows: list[dict[str, object]] = []
    metadata_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, int]] = []
    for position, source_id in enumerate((101, 102), start=1):
        mutated_id = -position
        brand = f"brand-{position}"
        left_attributes = {
            "Тип товара": "чехол",
            "Бренд": brand,
            "Цвет товара": "красный",
        }
        right_attributes = {**left_attributes, "Цвет товара": "синий"}
        item_rows.extend(
            [
                {
                    "id": source_id,
                    "name": f"чехол {brand} красный",
                    "attributes": json.dumps(left_attributes, ensure_ascii=False),
                    "category": "Электроника",
                },
                {
                    "id": mutated_id,
                    "name": f"чехол {brand} синий",
                    "attributes": json.dumps(right_attributes, ensure_ascii=False),
                    "category": "Электроника",
                },
            ]
        )
        pair_rows.append({"id1": source_id, "id2": mutated_id, "target": 0})
        application = {
            "concept": "color",
            "attribute_key": "Цвет товара",
            "original_value": "красный",
            "new_value": "синий",
        }
        task_index = position - 1
        task = PairGenerationTask(
            task_index=task_index,
            mutated_id=mutated_id,
            seed=int(stable_hash64(17, source_id) % (2**31 - 1)),
            anchor={"id": source_id},
        )
        nonce_common = {
            "task_seed_offset": 0,
            "task_retry_round": 0,
            "selection_attempt": 1,
        }
        anchor_nonce_sha256 = hashlib.sha256(
            _attempt_diversity_nonce(
                task,
                **nonce_common,
                stage="anchor",
                stage_attempt=1,
            ).encode("utf-8")
        ).hexdigest()
        mutation_nonce_sha256 = hashlib.sha256(
            _attempt_diversity_nonce(
                task,
                **nonce_common,
                stage="mutation",
                stage_attempt=1,
            ).encode("utf-8")
        ).hexdigest()
        metadata_rows.append(
            {
                "id1": source_id,
                "id2": mutated_id,
                "target": 0,
                "task_index": task_index,
                "source_style_id": source_id,
                "category": "Электроника",
                "product_type": "чехол",
                "rule_count": 1,
                "rule_ids": json.dumps(["rule-1"]),
                "applications_json": json.dumps([application], ensure_ascii=False),
                "semantic_signature": _semantic_pair_signature(
                    "Электроника", "чехол", [application]
                ),
                "semantic_signature_version": SEMANTIC_SIGNATURE_VERSION,
                "attempt_diversity_version": ATTEMPT_DIVERSITY_VERSION,
                "task_seed_offset": 0,
                "task_retry_round": 0,
                "selection_attempt": 1,
                "anchor_attempt": 1,
                "mutation_attempt": 1,
                "pair_attempts_config": 2,
                "anchor_attempts_config": 2,
                "mutation_attempts_config": 2,
                "anchor_diversity_nonce_sha256": anchor_nonce_sha256,
                "mutation_diversity_nonce_sha256": mutation_nonce_sha256,
                "global_rejection_feedback_count": task_index,
                "forbidden_semantic_signature_count": task_index,
                "forbidden_card_key_count": task_index * 2,
                "balanced_rule_schedule": True,
                "rule_schedule_version": SCHEDULE_VERSION,
                "rule_schedule_sha256": "schedule-sha",
                "scheduled_primary_rule_id": "rule-1",
                "scheduled_primary_profile_id": "profile-1",
                "scheduled_primary_task_cap": None,
                "scheduled_secondary_rule_id": None,
                "scheduled_secondary_profile_id": None,
                "scheduled_rule_ids": json.dumps(["rule-1"]),
                "scheduled_rule_profile_ids": json.dumps(["profile-1"]),
                "profile_capacity_policy_version": "test-capacity-policy-v1",
                "profile_capacity_policy_sha256": "a" * 64,
            }
        )

    pd.DataFrame(item_rows).to_parquet(source_dir / "items.parquet", index=False)
    pd.DataFrame(pair_rows).to_parquet(source_dir / "pairs.parquet", index=False)
    pd.DataFrame(metadata_rows).to_parquet(
        source_dir / "pair_generation_metadata.parquet", index=False
    )
    summary: dict[str, object] = {
        "generated_pairs": 2,
        "pending": 0,
        "validation_valid": True,
        "run_signature": "run-signature",
        "model": "fake-qwen",
        "structured_output": False,
        "prompt_sha256": "prompt-sha",
        "seed": 17,
        "rule_catalogs": [{"path": "rules.json", "sha256": "rules-sha"}],
        "rule_tiers": ["STAT_TEST"],
        "eligible_rule_profiles": 1,
        "profile_capacity_policy_version": "test-capacity-policy-v1",
        "profile_capacity_policy_sha256": "a" * 64,
        **EXPECTED_ATTEMPT_PROVENANCE,
        **EXPECTED_SEMANTIC_PROVENANCE,
        **EXPECTED_SCHEDULE_PROVENANCE,
    }
    (source_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (source_dir / "validation_report.json").write_text(
        json.dumps({"valid": True, "pairs": 2}, indent=2), encoding="utf-8"
    )
    return summary


class GeneratedPairProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.freezer = load_script("generated_pair_freezer_test", FREEZER_PATH)
        cls.uploader = load_script("generated_pair_uploader_test", UPLOADER_PATH)

    def test_freeze_preserves_semantic_and_schedule_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pair-provenance-freeze-") as raw:
            root = Path(raw)
            source_dir, output_dir = root / "source", root / "frozen"
            build_source(source_dir)
            validation = {"valid": True, "pairs": 2}
            with mock.patch.object(
                self.freezer, "validate_pair_dataset", return_value=validation
            ):
                result = self.freezer.freeze(source_dir, output_dir, 2)

            summary = result["summary"]
            for key, expected in EXPECTED_SEMANTIC_PROVENANCE.items():
                with self.subTest(key=key):
                    self.assertEqual(summary[f"source_{key}"], expected)
            self.assertEqual(
                summary["source_rule_schedule"], EXPECTED_SCHEDULE_PROVENANCE
            )
            self.assertEqual(
                summary["source_generation_provenance"],
                {
                    "semantic_signature": EXPECTED_SEMANTIC_PROVENANCE,
                    "attempt_diversity": EXPECTED_ATTEMPT_PROVENANCE,
                    "rule_schedule": EXPECTED_SCHEDULE_PROVENANCE,
                },
            )
            self.assertEqual(
                result["manifest"]["source_generation_provenance"],
                summary["source_generation_provenance"],
            )
            self.assertEqual(
                result["manifest"]["source_semantic_signature"],
                summary["source_generation_provenance"]["semantic_signature"],
            )
            self.assertEqual(
                result["manifest"]["source_rule_schedule"],
                summary["source_rule_schedule"],
            )
            frozen_schedule = summary["frozen_rule_schedule"]
            frozen_attempt = summary["frozen_attempt_diversity"]
            self.assertEqual(
                frozen_attempt["attempt_diversity_version"],
                ATTEMPT_DIVERSITY_VERSION,
            )
            self.assertEqual(frozen_attempt["selected_task_count"], 2)
            self.assertEqual(frozen_attempt["anchor_nonce_hash_valid_count"], 2)
            self.assertEqual(frozen_attempt["anchor_nonce_hash_unique_count"], 2)
            self.assertEqual(frozen_attempt["mutation_nonce_hash_valid_count"], 2)
            self.assertEqual(frozen_attempt["mutation_nonce_hash_unique_count"], 2)
            self.assertEqual(
                frozen_attempt["global_rejection_feedback_count_distribution"],
                {"0": 1, "1": 1},
            )
            self.assertEqual(frozen_schedule["selected_task_count"], 2)
            self.assertTrue(frozen_schedule["full_primary_rule_coverage"])
            self.assertTrue(
                frozen_schedule["full_primary_rule_profile_coverage"]
            )
            self.assertEqual(frozen_schedule["primary_rule_usage"], {"rule-1": 2})
            self.assertEqual(
                frozen_schedule["primary_rule_profile_usage"], {"profile-1": 2}
            )
            self.assertEqual(frozen_schedule["semantic_signature_unique_count"], 1)
            self.assertEqual(frozen_schedule["semantic_signature_max_count"], 2)
            self.assertEqual(
                result["manifest"]["frozen_rule_schedule"], frozen_schedule
            )
            self.assertEqual(
                result["manifest"]["frozen_attempt_diversity"], frozen_attempt
            )
            persisted_summary = json.loads(
                (output_dir / "summary.json").read_text(encoding="utf-8")
            )
            persisted_manifest = json.loads(
                (output_dir / "freeze_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted_summary, summary)
            self.assertEqual(
                persisted_manifest["source_generation_provenance"],
                summary["source_generation_provenance"],
            )

    def test_freeze_drops_cross_category_normalized_card_duplicate(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="pair-global-card-freeze-"
        ) as raw:
            root = Path(raw)
            source_dir, output_dir = root / "source", root / "frozen"
            source_dir.mkdir()
            item_rows = [
                {
                    "id": 101,
                    "name": "Ёж Галант",
                    "attributes": json.dumps(
                        {"Цвет": "Синёе", "Материал": "Сталь"},
                        ensure_ascii=False,
                    ),
                    "category": "Галантерея",
                },
                {
                    "id": -1,
                    "name": "ёж галант красный",
                    "attributes": json.dumps(
                        {"Цвет": "Красный", "Материал": "Сталь"},
                        ensure_ascii=False,
                    ),
                    "category": "Галантерея",
                },
                {
                    "id": 102,
                    "name": "ЕЖ ГАЛАНТ",
                    "attributes": json.dumps(
                        {" материал ": "СТАЛЬ", "ЦВЕТ": "СИНЕЕ"},
                        ensure_ascii=False,
                    ),
                    "category": "Ювелирные изделия",
                },
                {
                    "id": -2,
                    "name": "подвеска серебряная",
                    "attributes": json.dumps(
                        {"Тип": "Подвеска", "Материал": "Серебро"},
                        ensure_ascii=False,
                    ),
                    "category": "Ювелирные изделия",
                },
                {
                    "id": 103,
                    "name": "чехол зелёный",
                    "attributes": json.dumps(
                        {"Тип": "Чехол", "Цвет": "Зелёный"},
                        ensure_ascii=False,
                    ),
                    "category": "Электроника",
                },
                {
                    "id": -3,
                    "name": "чехол жёлтый",
                    "attributes": json.dumps(
                        {"Тип": "Чехол", "Цвет": "Жёлтый"},
                        ensure_ascii=False,
                    ),
                    "category": "Электроника",
                },
            ]
            pd.DataFrame(item_rows).to_parquet(
                source_dir / "items.parquet", index=False
            )
            pd.DataFrame(
                [
                    {"id1": 101, "id2": -1, "target": 0},
                    {"id1": 102, "id2": -2, "target": 0},
                    {"id1": 103, "id2": -3, "target": 0},
                ]
            ).to_parquet(source_dir / "pairs.parquet", index=False)
            pd.DataFrame(
                [
                    {"id1": 101, "id2": -1, "task_index": 0},
                    {"id1": 102, "id2": -2, "task_index": 1},
                    {"id1": 103, "id2": -3, "task_index": 2},
                ]
            ).to_parquet(
                source_dir / "pair_generation_metadata.parquet", index=False
            )
            (source_dir / "summary.json").write_text(
                json.dumps({"generated_pairs": 3}), encoding="utf-8"
            )
            schedule_provenance = {
                "semantic_signature_version": SEMANTIC_SIGNATURE_VERSION,
                "semantic_signature_limit": 2,
                "semantic_signature_unique_count": 2,
                "semantic_signature_max_count": 1,
                "full_primary_rule_coverage": True,
                "full_primary_rule_profile_coverage": True,
            }
            attempt_provenance = {
                "version": self.freezer.FROZEN_ATTEMPT_DIVERSITY_VERSION,
                "attempt_diversity_version": ATTEMPT_DIVERSITY_VERSION,
                "selected_task_count": 2,
            }
            validation = {"valid": True, "pairs": 2}
            with (
                mock.patch.object(
                    self.freezer, "validate_pair_dataset", return_value=validation
                ),
                mock.patch.object(
                    self.freezer,
                    "frozen_subset_provenance",
                    return_value=schedule_provenance,
                ),
                mock.patch.object(
                    self.freezer,
                    "attempt_diversity_provenance",
                    return_value=attempt_provenance,
                ),
            ):
                result = self.freezer.freeze(source_dir, output_dir, 2)

            frozen_pairs = pd.read_parquet(output_dir / "pairs.parquet")
            self.assertEqual(frozen_pairs["id1"].tolist(), [101, 103])
            dropped = json.loads(
                (output_dir / "dropped_pairs.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                dropped,
                [{"id1": 102, "id2": -2, "reason": "duplicate_global_card"}],
            )
            provenance = result["summary"]["frozen_global_card_uniqueness"]
            self.assertEqual(
                provenance["version"],
                self.freezer.FROZEN_GLOBAL_CARD_UNIQUENESS_VERSION,
            )
            self.assertEqual(provenance["source_card_count"], 6)
            self.assertEqual(provenance["source_unique_card_count"], 5)
            self.assertEqual(provenance["source_duplicate_card_group_count"], 1)
            self.assertEqual(
                provenance["source_cross_category_duplicate_card_group_count"],
                1,
            )
            self.assertEqual(provenance["frozen_card_count"], 4)
            self.assertEqual(provenance["frozen_unique_card_count"], 4)
            self.assertEqual(provenance["frozen_duplicate_card_group_count"], 0)
            self.assertEqual(
                provenance["global_card_collision_drop_pair_count"], 1
            )
            self.assertEqual(
                provenance["drop_reason_counts"], {"duplicate_global_card": 1}
            )
            self.assertTrue(
                result["summary"]["frozen_rule_schedule"][
                    "full_primary_rule_coverage"
                ]
            )
            self.assertEqual(
                result["manifest"]["frozen_global_card_uniqueness"],
                provenance,
            )

    def test_frozen_subset_accepts_signed_absence_of_capacity_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="policy-free-freeze-") as raw:
            source_dir = Path(raw) / "source"
            summary = build_source(source_dir)
            metadata_path = source_dir / "pair_generation_metadata.parquet"
            metadata = pd.read_parquet(metadata_path)
            metadata["profile_capacity_policy_version"] = ""
            metadata["profile_capacity_policy_sha256"] = ""
            metadata["scheduled_primary_task_cap"] = None
            summary["profile_capacity_policy_version"] = ""
            summary["profile_capacity_policy_sha256"] = ""
            provenance = self.freezer.frozen_subset_provenance(metadata, summary)
            self.assertEqual(provenance["profile_capacity_policy_version"], "")
            self.assertEqual(provenance["profile_capacity_policy_sha256"], "")
            self.assertEqual(provenance["primary_rule_profile_caps"], {})

    def test_uploader_manifest_carries_frozen_generation_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pair-provenance-upload-") as raw:
            root = Path(raw)
            source_dir, frozen_dir, stage_dir = (
                root / "source",
                root / "frozen",
                root / "stage",
            )
            build_source(source_dir)
            validation = {"valid": True, "pairs": 2}
            with mock.patch.object(
                self.freezer, "validate_pair_dataset", return_value=validation
            ):
                frozen = self.freezer.freeze(source_dir, frozen_dir, 2)
            with mock.patch.object(
                self.uploader, "validate_pair_dataset", return_value=validation
            ):
                upload_manifest = self.uploader.build_payload(
                    frozen_dir,
                    stage_dir,
                    "testowner",
                    2,
                    dataset_slug="generated-pairs-test",
                    artifact_tag="test2",
                    label_source="generated_test",
                )

            source_provenance = upload_manifest["source_provenance"]
            generation_provenance = frozen["summary"][
                "source_generation_provenance"
            ]
            self.assertEqual(
                source_provenance["semantic_signature"],
                generation_provenance["semantic_signature"],
            )
            self.assertEqual(
                source_provenance["rule_schedule"],
                generation_provenance["rule_schedule"],
            )
            self.assertEqual(
                source_provenance["attempt_diversity"],
                generation_provenance["attempt_diversity"],
            )
            self.assertEqual(
                source_provenance["frozen_rule_schedule"],
                frozen["summary"]["frozen_rule_schedule"],
            )
            self.assertEqual(
                source_provenance["frozen_semantic_signature"],
                frozen["summary"]["frozen_semantic_signature"],
            )
            self.assertEqual(
                source_provenance["frozen_attempt_diversity"],
                frozen["summary"]["frozen_attempt_diversity"],
            )
            self.assertEqual(source_provenance["run_signature"], "run-signature")
            self.assertIn("freeze_manifest.json", upload_manifest["files"])
            staged_freeze_manifest = json.loads(
                (stage_dir / "freeze_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                staged_freeze_manifest["frozen_global_card_uniqueness"],
                frozen["summary"]["frozen_global_card_uniqueness"],
            )
            self.assertEqual(
                upload_manifest["files"]["freeze_manifest.json"]["sha256"],
                hashlib.sha256(
                    (stage_dir / "freeze_manifest.json").read_bytes()
                ).hexdigest(),
            )
            persisted = json.loads(
                (stage_dir / "upload_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(persisted, upload_manifest)

    def test_uploader_manifest_extracts_direct_raw_schedule_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pair-provenance-raw-upload-") as raw:
            root = Path(raw)
            source_dir, stage_dir = root / "source", root / "stage"
            build_source(source_dir)
            validation = {"valid": True, "pairs": 2}
            with mock.patch.object(
                self.uploader, "validate_pair_dataset", return_value=validation
            ):
                upload_manifest = self.uploader.build_payload(
                    source_dir,
                    stage_dir,
                    "testowner",
                    2,
                    dataset_slug="generated-pairs-raw-test",
                    artifact_tag="raw2",
                    label_source="generated_test",
                )

            self.assertEqual(
                upload_manifest["source_provenance"]["semantic_signature"],
                EXPECTED_SEMANTIC_PROVENANCE,
            )
            self.assertEqual(
                upload_manifest["source_provenance"]["rule_schedule"],
                EXPECTED_SCHEDULE_PROVENANCE,
            )
            self.assertEqual(
                upload_manifest["source_provenance"]["attempt_diversity"],
                EXPECTED_ATTEMPT_PROVENANCE,
            )

    def test_freeze_rejects_mixed_or_summary_mismatched_attempt_protocol(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pair-provenance-reject-") as raw:
            root = Path(raw)
            source_dir, output_dir = root / "source", root / "frozen"
            build_source(source_dir)
            metadata_path = source_dir / "pair_generation_metadata.parquet"
            metadata = pd.read_parquet(metadata_path)
            metadata.loc[0, "attempt_diversity_version"] = "legacy"
            metadata.to_parquet(metadata_path, index=False)
            validation = {"valid": True, "pairs": 2}
            with (
                mock.patch.object(
                    self.freezer, "validate_pair_dataset", return_value=validation
                ),
                self.assertRaisesRegex(ValueError, "mixes or uses unexpected"),
            ):
                self.freezer.freeze(source_dir, output_dir, 2)

            metadata.loc[0, "attempt_diversity_version"] = (
                ATTEMPT_DIVERSITY_VERSION
            )
            metadata.to_parquet(metadata_path, index=False)
            summary_path = source_dir / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["attempt_diversity_version"] = "legacy"
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            with (
                mock.patch.object(
                    self.freezer, "validate_pair_dataset", return_value=validation
                ),
                self.assertRaisesRegex(
                    ValueError, "source summary has an unexpected"
                ),
            ):
                self.freezer.freeze(source_dir, output_dir, 2)

    def test_uploader_rejects_tampered_attempt_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pair-upload-reject-") as raw:
            root = Path(raw)
            source_dir, stage_dir = root / "source", root / "stage"
            build_source(source_dir)
            metadata_path = source_dir / "pair_generation_metadata.parquet"
            metadata = pd.read_parquet(metadata_path)
            metadata.loc[0, "mutation_diversity_nonce_sha256"] = "0" * 64
            metadata.to_parquet(metadata_path, index=False)
            validation = {"valid": True, "pairs": 2}
            with (
                mock.patch.object(
                    self.uploader, "validate_pair_dataset", return_value=validation
                ),
                self.assertRaises(SystemExit),
            ):
                self.uploader.build_payload(
                    source_dir,
                    stage_dir,
                    "testowner",
                    2,
                    dataset_slug="generated-pairs-reject-test",
                    artifact_tag="reject2",
                    label_source="generated_test",
                )


if __name__ == "__main__":
    unittest.main()
