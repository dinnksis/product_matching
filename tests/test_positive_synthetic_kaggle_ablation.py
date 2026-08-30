from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HUMAN_ALLOWLIST_VERSION = "non_title_metadata_exact_signatures_v1"
HUMAN_ALLOWLIST_FAMILIES = {
    "warranty": [["warranty"], ["duration", "warranty"]],
}


def load_launcher():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "launch_positive_synthetic_kaggle_ablation.py"
    spec = importlib.util.spec_from_file_location(
        "positive_synthetic_kaggle_launcher_test", path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PositiveSyntheticKaggleAblationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = load_launcher()

    def make_source(
        self,
        root: Path,
        *,
        duplicate_card: bool = False,
        target: int = 1,
    ) -> tuple[Path, Path]:
        source = root / "source"
        source.mkdir()
        duplicate_name = "Чехол Альфа"
        duplicate_attributes = json.dumps(
            {
                "Тип товара": "чехол",
                "Бренд": "Альфа",
                "Цвет": "чёрный",
            },
            ensure_ascii=False,
        )
        items = pd.DataFrame(
            [
                {
                    "id": 101,
                    "name": duplicate_name,
                    "attributes": duplicate_attributes,
                    "category": "Электроника",
                },
                {
                    "id": 102,
                    "name": "Альфа чехол для телефона",
                    "attributes": json.dumps(
                        {
                            "Тип товара": "чехол",
                            "Производитель": "Альфа",
                            "Материал": "силикон",
                        },
                        ensure_ascii=False,
                    ),
                    "category": "Электроника",
                },
                {
                    "id": 201,
                    "name": duplicate_name if duplicate_card else "Лампа Бета",
                    "attributes": (
                        duplicate_attributes
                        if duplicate_card
                        else json.dumps(
                            {
                                "Тип товара": "лампа",
                                "Бренд": "Бета",
                                "Мощность": "12 Вт",
                            },
                            ensure_ascii=False,
                        )
                    ),
                    "category": "Дом и сад",
                },
                {
                    "id": 202,
                    "name": "Бета настольная лампа 12 Вт",
                    "attributes": json.dumps(
                        {
                            "Тип": "настольная лампа",
                            "Производитель": "Бета",
                            "Цвет свечения": "тёплый",
                        },
                        ensure_ascii=False,
                    ),
                    "category": "Дом и сад",
                },
            ]
        )
        pairs = pd.DataFrame(
            [
                {"id1": 101, "id2": 102, "target": target},
                {"id1": 201, "id2": 202, "target": target},
            ]
        )
        metadata = pairs.assign(
            task_index=[8, 3], generator_stage=["tier_a", "tier_b"]
        )
        items.to_parquet(source / "items.parquet", index=False)
        pairs.to_parquet(source / "pairs.parquet", index=False)
        metadata.to_parquet(
            source / "pair_generation_metadata.parquet", index=False
        )
        provenance = root / "generation_report.json"
        provenance.write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model": "qwen3.5-397b-a17b-fp8",
                    "workers": 60,
                    "pairs": 2,
                }
            ),
            encoding="utf-8",
        )
        return source, provenance

    def make_hardened_provenance(
        self,
        source: Path,
        *,
        dataset_kind: str,
        run_signature: str = "a" * 64,
    ) -> tuple[list[Path], str, str]:
        if dataset_kind == "human_skeleton_ab":
            builder = "soft_positive_human_skeleton_overlay_v2"
            validator = "soft_positive_human_skeleton_validator_v2"
            label_source = "soft_positive_human_skeleton_ab_v1"
            metadata_filename = "pair_generation_metadata.parquet"
        elif dataset_kind == "near_duplicate":
            builder = "surface_positive_augmentation_v1"
            validator = "surface_positive_no_atomic_change_validator_v1"
            label_source = "deterministic_surface_only_positive_v1"
            metadata_filename = "pair_provenance.parquet"
        else:
            raise ValueError(dataset_kind)

        original_metadata = pd.read_parquet(
            source / "pair_generation_metadata.parquet"
        )
        metadata = original_metadata.assign(builder_version=builder)
        if dataset_kind == "human_skeleton_ab":
            metadata = metadata.assign(
                run_signature=run_signature,
                non_target_values_preserved=True,
                observed_label1_transition=True,
                evidence_human_label=1,
                construction_mode=["overlay", "source_pair_surface"],
                overlay_allowlist_version=HUMAN_ALLOWLIST_VERSION,
                overlay_allowlist_family=["warranty", ""],
                semantic_keys_compatible_for_overlay=[True, False],
                schema_safe_for_overlay=[True, False],
                required_attribute_key=["гарантийный срок", "цвет"],
                target_key_a=["гарантийный срок", "цвет"],
                target_key_b=["срок гарантии", "цвет"],
                target_value_a=["12 месяцев", "чёрный"],
                target_value_b=["24 месяца", "белый"],
                concept=["warranty_period", ""],
                composition_index=[0, 1],
            )
        else:
            metadata = metadata.assign(
                no_atomic_change=True,
                label_source=label_source,
            )
        metadata.to_parquet(source / metadata_filename, index=False)

        pair_count = len(metadata)
        item_count = pair_count * 2
        targets = {"0": 0, "1": pair_count}
        source_provenance = {"fixture_contract": "positive-source-test-v1"}
        if dataset_kind == "human_skeleton_ab":
            signed_source_dir = source.parent / "signed_source"
            validation_dir = source.parent / "frozen_validation"
            signed_source_dir.mkdir()
            validation_dir.mkdir()
            source_task_metadata = pd.DataFrame(
                {
                    "composition_index": [0, 1, 2],
                    "task_index": [8, 3, 5],
                }
            )
            source_metadata_path = (
                signed_source_dir / "pair_generation_metadata.parquet"
            )
            source_task_metadata.to_parquet(source_metadata_path, index=False)
            leaked_product_text = (
                "Категория: Тест\nНазвание: Контрольная утечка\nТип товара: тест"
            )
            validation_items = pd.DataFrame(
                {
                    "id": [9001, 9002],
                    "product_text": [
                        leaked_product_text,
                        "Категория: Тест\nНазвание: Иной контрольный товар",
                    ],
                }
            )
            validation_items.to_parquet(
                validation_dir / "items.parquet", index=False
            )
            pd.DataFrame(
                {"id1": pd.Series([9001], dtype="int64"),
                 "id2": pd.Series([9002], dtype="int64")}
            ).to_parquet(
                validation_dir / "iid_validation_pairs.parquet", index=False
            )
            for split in ("hard", "ood"):
                pd.DataFrame(
                    {
                        "id1": pd.Series([], dtype="int64"),
                        "id2": pd.Series([], dtype="int64"),
                    }
                ).to_parquet(
                    validation_dir / f"{split}_validation_pairs.parquet",
                    index=False,
                )

            def provenance_record(path: Path) -> dict[str, object]:
                raw = path.read_bytes()
                return {
                    "path": str(path.resolve()),
                    "bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }

            source_provenance.update(
                {
                    "source_metadata": provenance_record(source_metadata_path),
                    "validation_items": provenance_record(
                        validation_dir / "items.parquet"
                    ),
                    "validation_iid_pairs": provenance_record(
                        validation_dir / "iid_validation_pairs.parquet"
                    ),
                    "validation_hard_pairs": provenance_record(
                        validation_dir / "hard_validation_pairs.parquet"
                    ),
                    "validation_ood_pairs": provenance_record(
                        validation_dir / "ood_validation_pairs.parquet"
                    ),
                }
            )
            dropped_tokens = list(
                self.launcher.human_skeleton_builder.fact_tokens(
                    leaked_product_text
                )
            )
            validation_overlap_filter = {
                "version": "frozen_validation_serialized_fact_exclusion_v1",
                "fact_key_version": (
                    "ordered_fact_tokens_serialize_product_6000_v1"
                ),
                "source_task_count": 3,
                "emitted_pair_count": 2,
                "dropped_pair_count": 1,
                "dropped_task_ids": [2],
                "dropped_pairs_by_split": {"iid": 1, "hard": 0, "ood": 0},
                "dropped_endpoints_by_side": {"a": 1, "b": 0},
                "dropped_pairs_by_construction_mode": {
                    "source_pair_surface": 1
                },
                "unique_dropped_card_facts": 1,
                "postfilter_overlapping_card_count": 0,
                "validation_reference": {
                    "split_pair_counts": {"iid": 1, "hard": 0, "ood": 0},
                    "split_item_counts": {"iid": 2, "hard": 0, "ood": 0},
                    "unique_item_ids": 2,
                    "unique_fact_keys": 2,
                },
                "dropped_pairs": [
                    {
                        "composition_index": 2,
                        "source_task_index": 5,
                        "generated_id1": -999,
                        "generated_id2": -1000,
                        "construction_mode": "source_pair_surface",
                        "overlap_endpoints": [
                            {
                                "side": "a",
                                "generated_item_id": -999,
                                "fact_key_sha256": (
                                    self.launcher.human_skeleton_builder.sha256_json(
                                        dropped_tokens
                                    )
                                ),
                                "fact_tokens": dropped_tokens,
                                "validation_matches_by_split": {"iid": [9001]},
                            }
                        ],
                    }
                ],
            }
            overlay_allowlist = {
                "version": HUMAN_ALLOWLIST_VERSION,
                "families": HUMAN_ALLOWLIST_FAMILIES,
                "task_counts_by_family": {
                    "warranty": 1,
                    "not_allowlisted": 1,
                },
                "construction_mode_counts_by_family": {
                    "overlay": {"warranty": 1},
                    "source_pair_surface": {"not_allowlisted": 1},
                },
            }
            skeleton_distribution = {
                "construction_modes": {
                    "overlay": 1,
                    "source_pair_surface": 1,
                },
                "total_assignments": 2,
                "unique": 2,
                "unique_fraction": 1.0,
                "max_reuse": 1,
                "overlay_max_reuse": 1,
                "fallback_max_reuse": 1,
            }
            fact_clone_diagnostics = {
                "version": "punctuation_insensitive_fact_clone_diagnostics_v1",
                "nonblocking": True,
                "human_positive_reference": {
                    "unique_cards": 4,
                    "unique_pairs": 2,
                },
                "all": {
                    "cards": {
                        "total": 4,
                        "unique": 4,
                        "excess_clones": 0,
                        "fact_identical_to_human_positive": 2,
                    },
                    "pairs": {
                        "total": 2,
                        "unique": 2,
                        "excess_clones": 0,
                        "fact_identical_to_human_positive": 1,
                    },
                },
                "by_construction_mode": {
                    "overlay": {
                        "cards": {
                            "total": 2,
                            "unique": 2,
                            "excess_clones": 0,
                            "fact_identical_to_human_positive": 0,
                        },
                        "pairs": {
                            "total": 1,
                            "unique": 1,
                            "excess_clones": 0,
                            "fact_identical_to_human_positive": 0,
                        },
                    },
                    "source_pair_surface": {
                        "cards": {
                            "total": 2,
                            "unique": 2,
                            "excess_clones": 0,
                            "fact_identical_to_human_positive": 2,
                        },
                        "pairs": {
                            "total": 1,
                            "unique": 1,
                            "excess_clones": 0,
                            "fact_identical_to_human_positive": 1,
                        },
                    },
                },
            }
            validation = {
                "version": validator,
                "valid": True,
                "errors": {},
                "checked_pairs": pair_count,
                "pairs": pair_count,
                "items": item_count,
                "target_counts": targets,
                "unique_cards": item_count,
                "non_target_values_preserved": True,
                "source_task_count": 3,
                "validation_overlap_filter": validation_overlap_filter,
            }
            gates = {
                "unique_skeleton_fraction_gte_0_55": True,
                "shared_keys_median_delta_abs_lte_2": True,
            }
            distribution = {
                "schema_version": 1,
                "builder_version": builder,
                "valid": True,
                "gates": gates,
                "generated": {"pairs": pair_count},
                "overlay_allowlist": overlay_allowlist,
                "skeletons": skeleton_distribution,
                "fact_clone_diagnostics": fact_clone_diagnostics,
                "validation_overlap_filter": validation_overlap_filter,
            }
            extra_summary = {
                "distribution_gates": gates,
                "overlay_allowlist": overlay_allowlist,
                "skeleton_distribution": skeleton_distribution,
                "fact_clone_diagnostics": fact_clone_diagnostics,
                "source_task_count": 3,
                "validation_overlap_filter": validation_overlap_filter,
                "config": {
                    "count": pair_count,
                    "source_task_count": 3,
                    "dropped_validation_overlap": 1,
                },
            }
        else:
            validation = {
                "version": validator,
                "valid": True,
                "errors": [],
                "warnings": [],
                "checked_pairs": pair_count,
                "pairs": pair_count,
                "items": item_count,
                "target_counts": targets,
                "no_new_or_changed_visible_facts": True,
            }
            distribution = {
                "schema_version": 1,
                "builder_version": builder,
                "generated": {"pairs": pair_count},
                "interpretation": {"atomic_attribute_changes": 0},
            }
            extra_summary = {"no_atomic_change": True}
        summary = {
            "schema_version": 1,
            "builder_version": builder,
            "validation_version": validator,
            "run_signature": run_signature,
            "generated_pairs": pair_count,
            "generated_items": item_count,
            "target_counts": targets,
            "label_source": label_source,
            "source_provenance": source_provenance,
            "validation": validation,
            **extra_summary,
        }
        if dataset_kind == "human_skeleton_ab":
            summary.update(
                {
                    "evidence_version": (
                        "label1_source_example_exact_subfacet_grounding_v2"
                    ),
                    "selection_version": (
                        "exact_scope_balanced_human_positive_pair_v1"
                    ),
                }
            )
        for filename, payload in (
            ("summary.json", summary),
            ("validation_report.json", validation),
            ("distribution_report.json", distribution),
        ):
            (source / filename).write_text(
                json.dumps(payload, sort_keys=True), encoding="utf-8"
            )

        manifest_filenames = {
            "items.parquet",
            "pairs.parquet",
            metadata_filename,
            "summary.json",
            "validation_report.json",
            "distribution_report.json",
        }
        files = {}
        for filename in sorted(manifest_filenames):
            path = source / filename
            raw = path.read_bytes()
            files[filename] = {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        manifest = {
            "schema_version": 1,
            "builder_version": builder,
            "run_signature": run_signature,
            "pairs": pair_count,
            "items": item_count,
            "targets": targets,
            "label_source": label_source,
            "source_provenance": source_provenance,
            "config": {"count": pair_count},
            "files": files,
        }
        if dataset_kind == "near_duplicate":
            manifest["no_atomic_change"] = True
        else:
            manifest["overlay_allowlist"] = overlay_allowlist
            manifest["source_task_count"] = 3
            manifest["validation_overlap_filter"] = validation_overlap_filter
            manifest["config"] = {
                "count": pair_count,
                "source_task_count": 3,
                "dropped_validation_overlap": 1,
            }
        (source / "build_manifest.json").write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )
        provenance_paths = [
            source / filename
            for filename in (
                "summary.json",
                "validation_report.json",
                "distribution_report.json",
                "build_manifest.json",
            )
        ]
        return provenance_paths, metadata_filename, label_source

    @staticmethod
    def refresh_manifest_file(source: Path, filename: str) -> None:
        manifest_path = source / "build_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw = (source / filename).read_bytes()
        manifest["files"][filename] = {
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True), encoding="utf-8"
        )

    def verify_hardened_fixture(
        self,
        source: Path,
        *,
        dataset_kind: str,
        provenance_paths: list[Path],
        metadata_filename: str,
    ):
        return self.launcher.verify_source(
            source,
            pair_count=2,
            dataset_kind=dataset_kind,
            provenance_paths=provenance_paths,
            metadata_filename=metadata_filename,
        )

    def test_all_dataset_kinds_are_validated_reindexed_and_fully_pinned(self) -> None:
        for dataset_kind in self.launcher.DATASET_KINDS:
            with self.subTest(dataset_kind=dataset_kind), tempfile.TemporaryDirectory(
                prefix=f"positive-{dataset_kind}-"
            ) as raw:
                root = Path(raw)
                source, provenance = self.make_source(root)
                if dataset_kind in self.launcher.HARDENED_DATASET_KINDS:
                    provenance_paths, metadata_filename, label_source = (
                        self.make_hardened_provenance(
                            source, dataset_kind=dataset_kind
                        )
                    )
                else:
                    provenance_paths = [provenance]
                    metadata_filename = "pair_generation_metadata.parquet"
                    label_source = f"{dataset_kind}_positive_v1"
                checked = self.launcher.verify_source(
                    source,
                    pair_count=2,
                    dataset_kind=dataset_kind,
                    provenance_paths=provenance_paths,
                    metadata_filename=metadata_filename,
                )
                stage = root / "stage"
                manifest = self.launcher.prepare_upload_payload(
                    checked,
                    stage_dir=stage,
                    owner="owner",
                    dataset_slug=f"positive-{dataset_kind.replace('_', '-')}",
                    artifact_tag=f"positive-{dataset_kind.replace('_', '-')}",
                    label_source=label_source,
                    dataset_kind=dataset_kind,
                )

                self.assertEqual(manifest["targets"], {"0": 0, "1": 2})
                self.assertEqual(manifest["pairs"], 2)
                self.assertEqual(manifest["items"], 4)
                self.assertTrue(manifest["is_private"])
                self.assertEqual(manifest["generation_kind"], dataset_kind)
                notes = self.launcher.build_notes(
                    dataset_kind=dataset_kind,
                    pair_count=2,
                    dataset_ref=str(manifest["dataset"]),
                    upload_manifest_sha256="d" * 64,
                    source_fingerprint=checked["source_fingerprint"],
                    extra_notes=None,
                    provenance_contract=checked.get("provenance_contract") or {},
                )
                self.assertIn(dataset_kind.replace("_", " "), notes)
                self.assertEqual(
                    manifest["source_provenance"]["source_fingerprint"],
                    checked["source_fingerprint"],
                )
                self.assertEqual(
                    manifest["source_provenance"]["verified_contract"],
                    checked["provenance_contract"],
                )
                if dataset_kind == "human_skeleton_ab":
                    self.assertIn("not a factual-novelty claim", notes)
                    self.assertIn("explicitly nonblocking", notes)
                    self.assertIn("2/4 cards", notes)
                    self.assertIn("excluded 1 of 3 source tasks", notes)
                    self.assertIn("zero overlapping cards", notes)
                    self.assertEqual(
                        checked["provenance_contract"]["overlay_allowlist"]
                        ["families"],
                        HUMAN_ALLOWLIST_FAMILIES,
                    )
                staged_pairs = pd.read_parquet(
                    stage
                    / f"generation_rule_pairs_positive-{dataset_kind.replace('_', '-')}.parquet"
                )
                staged_items = pd.read_parquet(
                    stage
                    / f"generation_rule_items_positive-{dataset_kind.replace('_', '-')}.parquet"
                )
                staged_metadata = pd.read_parquet(
                    stage
                    / (
                        "generation_rule_pair_metadata_positive-"
                        f"{dataset_kind.replace('_', '-')}.parquet"
                    )
                )
                self.assertTrue(staged_pairs["target"].eq(1).all())
                self.assertTrue(staged_items["id"].lt(-10**18).all())
                self.assertEqual(staged_items["id"].nunique(), 4)
                self.assertEqual(
                    set(staged_items["id"]),
                    set(staged_pairs["id1"]) | set(staged_pairs["id2"]),
                )
                self.assertEqual(
                    staged_metadata["ablation_source_id1"].tolist(), [101, 201]
                )
                self.assertEqual(
                    set(staged_pairs["label_source"]),
                    {label_source},
                )
                kaggle_metadata = json.loads(
                    (stage / "dataset-metadata.json").read_text(encoding="utf-8")
                )
                self.assertIs(kaggle_metadata["isPrivate"], True)

    def test_hardened_kinds_require_the_exact_provenance_bundle(self) -> None:
        for dataset_kind in self.launcher.HARDENED_DATASET_KINDS:
            with self.subTest(dataset_kind=dataset_kind), tempfile.TemporaryDirectory(
                prefix=f"positive-provenance-{dataset_kind}-"
            ) as raw:
                source, _ = self.make_source(Path(raw))
                provenance_paths, metadata_filename, _ = (
                    self.make_hardened_provenance(
                        source, dataset_kind=dataset_kind
                    )
                )
                with self.assertRaisesRegex(
                    RuntimeError, "kind-specific provenance documents differ"
                ):
                    self.verify_hardened_fixture(
                        source,
                        dataset_kind=dataset_kind,
                        provenance_paths=provenance_paths[:-1],
                        metadata_filename=metadata_filename,
                    )

    def test_human_skeleton_semantic_gates_and_signatures_fail_closed(self) -> None:
        mutations = ("non_target", "distribution_gate", "metadata_signature")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"positive-human-{mutation}-"
            ) as raw:
                source, _ = self.make_source(Path(raw))
                provenance_paths, metadata_filename, _ = (
                    self.make_hardened_provenance(
                        source, dataset_kind="human_skeleton_ab"
                    )
                )
                if mutation == "non_target":
                    validation_path = source / "validation_report.json"
                    validation = json.loads(
                        validation_path.read_text(encoding="utf-8")
                    )
                    validation["non_target_values_preserved"] = False
                    validation_path.write_text(
                        json.dumps(validation, sort_keys=True), encoding="utf-8"
                    )
                    summary_path = source / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["validation"] = validation
                    summary_path.write_text(
                        json.dumps(summary, sort_keys=True), encoding="utf-8"
                    )
                    self.refresh_manifest_file(source, "validation_report.json")
                    self.refresh_manifest_file(source, "summary.json")
                    error = "non-target values"
                elif mutation == "distribution_gate":
                    distribution_path = source / "distribution_report.json"
                    distribution = json.loads(
                        distribution_path.read_text(encoding="utf-8")
                    )
                    first_gate = next(iter(distribution["gates"]))
                    distribution["gates"][first_gate] = False
                    distribution_path.write_text(
                        json.dumps(distribution, sort_keys=True), encoding="utf-8"
                    )
                    summary_path = source / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["distribution_gates"] = distribution["gates"]
                    summary_path.write_text(
                        json.dumps(summary, sort_keys=True), encoding="utf-8"
                    )
                    self.refresh_manifest_file(source, "distribution_report.json")
                    self.refresh_manifest_file(source, "summary.json")
                    error = "distribution gates"
                else:
                    metadata_path = source / metadata_filename
                    metadata = pd.read_parquet(metadata_path)
                    metadata.loc[0, "run_signature"] = "b" * 64
                    metadata.to_parquet(metadata_path, index=False)
                    self.refresh_manifest_file(source, metadata_filename)
                    error = "metadata run_signature mismatch"
                with self.assertRaisesRegex(RuntimeError, error):
                    self.verify_hardened_fixture(
                        source,
                        dataset_kind="human_skeleton_ab",
                        provenance_paths=provenance_paths,
                        metadata_filename=metadata_filename,
                    )

    def test_human_overlay_allowlist_and_clone_diagnostics_fail_closed(self) -> None:
        self.assertEqual(
            self.launcher.HUMAN_SKELETON_OVERLAY_ALLOWLIST_FAMILIES,
            HUMAN_ALLOWLIST_FAMILIES,
        )
        mutations = {
            "family_definition": "family definitions are not pinned",
            "unsupported_metadata_family": "unsupported overlay families",
            "unsupported_overlay_key": "does not replay from exact key signature",
            "false_schema_proof": "schema overlay proof does not replay",
            "allowlist_counts": "task counts differ from metadata",
            "skeleton_counts": "construction-mode counts differ from metadata",
            "blocking_clone_diagnostics": "must be explicitly nonblocking",
        }
        for mutation, error in mutations.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"positive-human-contract-{mutation}-"
            ) as raw:
                source, _ = self.make_source(Path(raw))
                provenance_paths, metadata_filename, _ = (
                    self.make_hardened_provenance(
                        source, dataset_kind="human_skeleton_ab"
                    )
                )
                summary_path = source / "summary.json"
                distribution_path = source / "distribution_report.json"
                manifest_path = source / "build_manifest.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                distribution = json.loads(
                    distribution_path.read_text(encoding="utf-8")
                )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                metadata_path = source / metadata_filename
                metadata = pd.read_parquet(metadata_path)

                if mutation == "family_definition":
                    for payload in (summary, distribution, manifest):
                        payload["overlay_allowlist"]["families"][
                            "composition_ingredients"
                        ] = [["composition"]]
                elif mutation == "unsupported_metadata_family":
                    metadata.loc[0, "overlay_allowlist_family"] = (
                        "composition_ingredients"
                    )
                elif mutation == "unsupported_overlay_key":
                    metadata.loc[0, "required_attribute_key"] = "состав"
                    metadata.loc[0, "target_key_a"] = "состав"
                    metadata.loc[0, "target_key_b"] = "состав"
                elif mutation == "false_schema_proof":
                    metadata.loc[0, "schema_safe_for_overlay"] = False
                elif mutation == "allowlist_counts":
                    wrong_counts = {"warranty": 2}
                    for payload in (summary, distribution, manifest):
                        payload["overlay_allowlist"][
                            "task_counts_by_family"
                        ] = wrong_counts
                elif mutation == "skeleton_counts":
                    wrong_modes = {"overlay": 2, "source_pair_surface": 0}
                    summary["skeleton_distribution"][
                        "construction_modes"
                    ] = wrong_modes
                    distribution["skeletons"]["construction_modes"] = wrong_modes
                else:
                    summary["fact_clone_diagnostics"]["nonblocking"] = False
                    distribution["fact_clone_diagnostics"]["nonblocking"] = False

                metadata.to_parquet(metadata_path, index=False)
                summary_path.write_text(
                    json.dumps(summary, sort_keys=True), encoding="utf-8"
                )
                distribution_path.write_text(
                    json.dumps(distribution, sort_keys=True), encoding="utf-8"
                )
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True), encoding="utf-8"
                )
                for filename in (
                    metadata_filename,
                    "summary.json",
                    "distribution_report.json",
                ):
                    self.refresh_manifest_file(source, filename)
                if mutation in {"family_definition", "allowlist_counts"}:
                    refreshed_manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    refreshed_manifest["overlay_allowlist"] = manifest[
                        "overlay_allowlist"
                    ]
                    manifest_path.write_text(
                        json.dumps(refreshed_manifest, sort_keys=True),
                        encoding="utf-8",
                    )

                with self.assertRaisesRegex(RuntimeError, error):
                    self.verify_hardened_fixture(
                        source,
                        dataset_kind="human_skeleton_ab",
                        provenance_paths=provenance_paths,
                        metadata_filename=metadata_filename,
                    )

    def test_human_validation_overlap_filter_replays_and_fails_closed(self) -> None:
        mutations = {
            "cross_document_report": "validation-overlap reports differ",
            "duplicate_dropped_id": "composition IDs are not unique/exact",
            "dropped_id_still_emitted": "dropped composition set differs",
            "tampered_dropped_fact": "absent from frozen validation",
            "postfilter_card_overlap": "still overlap frozen validation facts",
            "validation_source_sha": "local SHA-256 differs from provenance",
            "validation_source_path": "must resolve to iid_validation_pairs.parquet",
        }
        for mutation, error in mutations.items():
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"positive-human-overlap-{mutation}-"
            ) as raw:
                source, _ = self.make_source(Path(raw))
                provenance_paths, metadata_filename, _ = (
                    self.make_hardened_provenance(
                        source, dataset_kind="human_skeleton_ab"
                    )
                )
                summary_path = source / "summary.json"
                validation_path = source / "validation_report.json"
                distribution_path = source / "distribution_report.json"
                manifest_path = source / "build_manifest.json"

                def synchronize_filter(report: dict[str, object]) -> None:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    validation = json.loads(
                        validation_path.read_text(encoding="utf-8")
                    )
                    distribution = json.loads(
                        distribution_path.read_text(encoding="utf-8")
                    )
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    validation["validation_overlap_filter"] = report
                    summary["validation_overlap_filter"] = report
                    summary["validation"] = validation
                    distribution["validation_overlap_filter"] = report
                    manifest["validation_overlap_filter"] = report
                    validation_path.write_text(
                        json.dumps(validation, sort_keys=True), encoding="utf-8"
                    )
                    summary_path.write_text(
                        json.dumps(summary, sort_keys=True), encoding="utf-8"
                    )
                    distribution_path.write_text(
                        json.dumps(distribution, sort_keys=True), encoding="utf-8"
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8"
                    )
                    for filename in (
                        "validation_report.json",
                        "summary.json",
                        "distribution_report.json",
                    ):
                        self.refresh_manifest_file(source, filename)

                if mutation == "cross_document_report":
                    distribution = json.loads(
                        distribution_path.read_text(encoding="utf-8")
                    )
                    distribution["validation_overlap_filter"][
                        "source_task_count"
                    ] = 4
                    distribution_path.write_text(
                        json.dumps(distribution, sort_keys=True), encoding="utf-8"
                    )
                    self.refresh_manifest_file(source, "distribution_report.json")
                elif mutation == "dropped_id_still_emitted":
                    metadata_path = source / metadata_filename
                    metadata = pd.read_parquet(metadata_path)
                    metadata.loc[0, "composition_index"] = 2
                    metadata.to_parquet(metadata_path, index=False)
                    self.refresh_manifest_file(source, metadata_filename)
                elif mutation == "postfilter_card_overlap":
                    items_path = source / "items.parquet"
                    items = pd.read_parquet(items_path)
                    pair_mask = items["id"].isin([101, 102])
                    items.loc[pair_mask, "category"] = "Тест"
                    items.loc[items["id"].eq(101), "name"] = "Контрольная утечка"
                    items.loc[items["id"].eq(101), "attributes"] = json.dumps(
                        {"Тип товара": "тест"}, ensure_ascii=False
                    )
                    items.to_parquet(items_path, index=False)
                    self.refresh_manifest_file(source, "items.parquet")
                elif mutation in {"validation_source_sha", "validation_source_path"}:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    if mutation == "validation_source_sha":
                        summary["source_provenance"]["validation_items"][
                            "sha256"
                        ] = "f" * 64
                    else:
                        summary["source_provenance"]["validation_iid_pairs"] = (
                            json.loads(
                                json.dumps(
                                    summary["source_provenance"][
                                        "validation_hard_pairs"
                                    ]
                                )
                            )
                        )
                    manifest["source_provenance"] = summary["source_provenance"]
                    summary_path.write_text(
                        json.dumps(summary, sort_keys=True), encoding="utf-8"
                    )
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8"
                    )
                    self.refresh_manifest_file(source, "summary.json")
                else:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    report = json.loads(
                        json.dumps(summary["validation_overlap_filter"])
                    )
                    if mutation == "duplicate_dropped_id":
                        report["dropped_task_ids"] = [2, 2]
                    else:
                        endpoint = report["dropped_pairs"][0][
                            "overlap_endpoints"
                        ][0]
                        endpoint["fact_tokens"].append("подмена")
                        endpoint["fact_key_sha256"] = (
                            self.launcher.human_skeleton_builder.sha256_json(
                                endpoint["fact_tokens"]
                            )
                        )
                    synchronize_filter(report)

                with self.assertRaisesRegex(RuntimeError, error):
                    self.verify_hardened_fixture(
                        source,
                        dataset_kind="human_skeleton_ab",
                        provenance_paths=provenance_paths,
                        metadata_filename=metadata_filename,
                    )

    def test_near_duplicate_semantics_and_manifest_hashes_fail_closed(self) -> None:
        for mutation in ("visible_facts", "manifest_hash"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                prefix=f"positive-near-{mutation}-"
            ) as raw:
                source, _ = self.make_source(Path(raw))
                provenance_paths, metadata_filename, _ = (
                    self.make_hardened_provenance(
                        source, dataset_kind="near_duplicate"
                    )
                )
                if mutation == "visible_facts":
                    validation_path = source / "validation_report.json"
                    validation = json.loads(
                        validation_path.read_text(encoding="utf-8")
                    )
                    validation["no_new_or_changed_visible_facts"] = False
                    validation_path.write_text(
                        json.dumps(validation, sort_keys=True), encoding="utf-8"
                    )
                    summary_path = source / "summary.json"
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                    summary["validation"] = validation
                    summary_path.write_text(
                        json.dumps(summary, sort_keys=True), encoding="utf-8"
                    )
                    self.refresh_manifest_file(source, "validation_report.json")
                    self.refresh_manifest_file(source, "summary.json")
                    error = "preserve visible facts"
                else:
                    manifest_path = source / "build_manifest.json"
                    manifest = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                    manifest["files"]["items.parquet"]["sha256"] = "f" * 64
                    manifest_path.write_text(
                        json.dumps(manifest, sort_keys=True), encoding="utf-8"
                    )
                    error = "SHA-256 differs"
                with self.assertRaisesRegex(RuntimeError, error):
                    self.verify_hardened_fixture(
                        source,
                        dataset_kind="near_duplicate",
                        provenance_paths=provenance_paths,
                        metadata_filename=metadata_filename,
                    )

    def test_hardened_upload_cannot_relabel_source(self) -> None:
        with tempfile.TemporaryDirectory(prefix="positive-relabel-") as raw:
            root = Path(raw)
            source, _ = self.make_source(root)
            provenance_paths, metadata_filename, _ = self.make_hardened_provenance(
                source, dataset_kind="near_duplicate"
            )
            checked = self.verify_hardened_fixture(
                source,
                dataset_kind="near_duplicate",
                provenance_paths=provenance_paths,
                metadata_filename=metadata_filename,
            )
            with self.assertRaisesRegex(ValueError, "pinned source provenance"):
                self.launcher.prepare_upload_payload(
                    checked,
                    stage_dir=root / "stage",
                    owner="owner",
                    dataset_slug="positive-near-relabel",
                    artifact_tag="positive-near-relabel",
                    label_source="wrong_positive_label",
                    dataset_kind="near_duplicate",
                )

    def test_rejects_negative_labels_and_category_agnostic_duplicate_cards(self) -> None:
        with tempfile.TemporaryDirectory(prefix="positive-reject-label-") as raw:
            root = Path(raw)
            source, provenance = self.make_source(root, target=0)
            with self.assertRaisesRegex(RuntimeError, "all target=1"):
                self.launcher.verify_source(
                    source,
                    pair_count=2,
                    dataset_kind="rehydrated_ab",
                    provenance_paths=[provenance],
                )
        with tempfile.TemporaryDirectory(prefix="positive-reject-card-") as raw:
            root = Path(raw)
            source, provenance = self.make_source(root, duplicate_card=True)
            with self.assertRaisesRegex(
                RuntimeError, "category-agnostic duplicate cards"
            ):
                self.launcher.verify_source(
                    source,
                    pair_count=2,
                    dataset_kind="near_duplicate",
                    provenance_paths=[provenance],
                )

    def test_rejects_secret_bearing_provenance(self) -> None:
        with tempfile.TemporaryDirectory(prefix="positive-secret-") as raw:
            root = Path(raw)
            source, provenance = self.make_source(root)
            provenance.write_text(
                json.dumps({"status": "complete", "api_key": "do-not-stage"}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "secret-like key"):
                self.launcher.verify_source(
                    source,
                    pair_count=2,
                    dataset_kind="rehydrated_ab",
                    provenance_paths=[provenance],
                )

    def test_generated_notebook_keeps_frozen_baseline_and_data_exps(self) -> None:
        with tempfile.TemporaryDirectory(prefix="positive-notebook-") as raw:
            output = Path(raw) / "positive.ipynb"
            self.launcher.generate_notebook(
                notebook=output,
                pair_count=2,
                artifact_tag="positive-test",
                experiment_label="minilm_5ep_positive_test_v1",
                dataset_ref="owner/positive-test",
                upload_manifest_sha256="d" * 64,
                label_source="positive_test_v1",
                notes="Test source owner/positive-test, " + "d" * 64,
            )
            notebook = nbformat.read(output, as_version=4)
            routing = next(
                cell
                for cell in notebook.cells
                if cell.cell_type == "code"
                and "experiment-routing"
                in cell.get("metadata", {}).get("tags", [])
            )
            data_hook = next(
                cell
                for cell in notebook.cells
                if cell.cell_type == "code"
                and "data-hook" in cell.get("metadata", {}).get("tags", [])
            )
            self.assertIn("EXPERIMENT_SHEET = 'data_exps'", routing.source)
            self.assertIn("expected_targets = {'0': 0, '1': 2}", data_hook.source)
            self.assertIn(
                self.launcher.frozen_notebook.SIGNIFICANCE_BASELINE_RUN_ID,
                "\n".join(cell.source for cell in notebook.cells),
            )

    def test_notebook_command_pins_all_four_datasets_and_waits(self) -> None:
        command = self.launcher.notebook_command(
            notebook=Path("experiment.ipynb"),
            env_file=Path(".env"),
            kernel_slug="positive-test",
            title="Positive test",
            dataset_ref="owner/positive-data",
        )
        dataset_values = [
            command[index + 1]
            for index, value in enumerate(command[:-1])
            if value == "--dataset"
        ]
        self.assertEqual(
            dataset_values,
            [
                self.launcher.VALIDATION_DATASET,
                self.launcher.CHECKPOINT_DATASET,
                self.launcher.SIGNIFICANCE_DATASET,
                "owner/positive-data",
            ],
        )
        self.assertIn("--no-env-sources", command)
        self.assertNotIn("--no-wait", command)
        self.assertNotIn("--no-google-sheets-credentials", command)

    def test_completion_requires_all_splits_significance_and_data_exps_sync(self) -> None:
        with tempfile.TemporaryDirectory(prefix="positive-completion-") as raw:
            output = Path(raw)
            model_dir = output / "minilm_5ep_team_data_loss_ablation"
            model_dir.mkdir()
            run_id = "a" * 32
            training_report = {
                "validation_splits": {
                    split: {"macro_average_precision": 0.5}
                    for split in self.launcher.EXPECTED_SPLITS
                }
            }
            completion = {
                "status": "complete",
                "run_id": run_id,
                "experiment": "minilm_5ep_positive_test_v1",
                "experiment_group": "data",
                "notes": "owner/data " + "b" * 64 + " " + "c" * 64,
                "train_data": {
                    "label_source_counts": {"positive_test_v1": 2}
                },
                "training_report": training_report,
            }
            sync = {
                "status": "synced",
                "run_id": run_id,
                "experiment_group": "data",
                "comparison_sheet": "data_exps",
            }
            split_result = {
                "baseline_macro_average_precision": 0.5,
                "candidate_macro_average_precision": 0.51,
                "delta_macro_average_precision": 0.01,
                "p_value": 0.1,
                "p_value_holm": 0.3,
                "ci95_low": -0.01,
                "ci95_high": 0.03,
            }
            comparison = {
                "status": "ready",
                "baseline_run_id": (
                    self.launcher.frozen_notebook.SIGNIFICANCE_BASELINE_RUN_ID
                ),
                "candidate_run_id": run_id,
                "splits": {
                    split: dict(split_result)
                    for split in self.launcher.EXPECTED_SPLITS
                },
            }
            for path, payload in (
                (output / "notebook_completed.json", completion),
                (output / "google_sheets_sync.json", sync),
                (output / "baseline_comparison.json", comparison),
                (model_dir / "training_report.json", training_report),
            ):
                path.write_text(json.dumps(payload), encoding="utf-8")
            result = self.launcher.verify_completion(
                output,
                experiment_label="minilm_5ep_positive_test_v1",
                dataset_ref="owner/data",
                upload_manifest_sha256="b" * 64,
                source_fingerprint="c" * 64,
                pair_count=2,
                label_source="positive_test_v1",
            )
            self.assertEqual(result["run_id"], run_id)

            sync["comparison_sheet"] = "sft_exps"
            (output / "google_sheets_sync.json").write_text(
                json.dumps(sync), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "data_exps"):
                self.launcher.verify_completion(
                    output,
                    experiment_label="minilm_5ep_positive_test_v1",
                    dataset_ref="owner/data",
                    upload_manifest_sha256="b" * 64,
                    source_fingerprint="c" * 64,
                    pair_count=2,
                    label_source="positive_test_v1",
                )


if __name__ == "__main__":
    unittest.main()
