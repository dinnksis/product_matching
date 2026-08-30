from __future__ import annotations

import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_sft_candidate_notebooks as candidate_builder
import create_bge_2ep_sft_notebooks as baseline_builder
import create_cross_encoder_training_notebook as cross_builder
import run_bge_2ep_sft_candidates as workflow


OWNER = "alexproger23"
BASELINE_RUN_ID = "a" * 32


@lru_cache(maxsize=1)
def current_baseline_entry() -> dict[str, object]:
    entries = baseline_builder.build_campaign(
        owner=OWNER,
        only={"baseline"},
        write=False,
    )
    if len(entries) != 1:
        raise AssertionError("Expected one dynamic baseline entry")
    return entries[0]


def baseline_context() -> dict[str, object]:
    entry = current_baseline_entry()
    binding = {
        "baseline_run_id": BASELINE_RUN_ID,
        "baseline_experiment": entry["experiment"],
        "campaign": baseline_builder.CAMPAIGN,
        "campaign_identity_sha256": entry["identity_sha256"],
        "source_sha256": entry["source_sha256"],
        "recipe_sha256": entry["recipe_sha256"],
        "executable_cells_sha256": entry["executable_cells_sha256"],
        "loss_hook_sha256": entry["loss_hook_sha256"],
        "checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
        "checkpoint_model_sha256": entry["checkpoint_model_sha256"],
        "validation_manifest_sha256": entry["validation_manifest_sha256"],
    }
    files = {
        "notebook_completed.json": {"bytes": 10, "sha256": "1" * 64},
        "iid_validation_predictions.parquet": {
            "bytes": 20,
            "sha256": "2" * 64,
        },
        "hard_validation_predictions.parquet": {
            "bytes": 30,
            "sha256": "3" * 64,
        },
    }
    manifest = {
        "schema_version": 1,
        "dataset": f"{OWNER}/{workflow.baseline_uploader.DATASET_SLUG}",
        "is_private": True,
        "binding": binding,
        "evaluated_splits": ["iid", "hard"],
        "ood": {
            "evaluated": False,
            "metric_sentinel": -1.0,
            "comparison": None,
            "prediction_file": None,
        },
        "files": files,
    }
    return {
        "dataset_ref": manifest["dataset"],
        "dataset_slug": workflow.baseline_uploader.DATASET_SLUG,
        "dataset_version": 7,
        "manifest_sha256": "4" * 64,
        "manifest_canonical_sha256": candidate_builder.canonical_sha256(manifest),
        "manifest": manifest,
        "binding": binding,
    }


def baseline_parent() -> dict[str, object]:
    return candidate_builder.baseline_parent_receipt(
        baseline_context(),
        current_baseline_entry(),
    )


def build_candidate(spec: dict[str, object], parent: dict[str, object]):
    entry = current_baseline_entry()
    config = cross_builder.load_training_config(baseline_builder.DEFAULT_CONFIG)
    validation = {
        "dataset": entry["validation_dataset"],
        "manifest_sha256": entry["validation_manifest_sha256"],
    }
    checkpoint = {
        "dataset": entry["checkpoint_dataset"],
        "manifest_sha256": entry["checkpoint_manifest_sha256"],
    }
    return candidate_builder.build_candidate_notebook(
        validation=validation,
        checkpoint=checkpoint,
        base_config=config,
        spec=spec,
        parent=parent,
        baseline_context=baseline_context(),
    )


def selected_candidate_parent(entry: dict[str, object]) -> dict[str, object]:
    return candidate_builder.validate_parent_receipt(
        {
            "run_id": "c" * 32,
            "experiment": entry["experiment"],
            "campaign_identity_sha256": entry["identity_sha256"],
            "source_sha256": entry["source_sha256"],
            "recipe_sha256": entry["recipe_sha256"],
            "checkpoint_manifest_sha256": entry["checkpoint_manifest_sha256"],
            "checkpoint_model_sha256": entry["checkpoint_model_sha256"],
            "validation_manifest_sha256": entry["validation_manifest_sha256"],
            "loss_hook_sha256": entry["loss_hook_sha256"],
            "config": entry["expected_config"],
        }
    )


def split_result(delta: float = 0.003) -> dict[str, object]:
    return {
        "examples": 10,
        "baseline_macro_average_precision": 0.5,
        "candidate_macro_average_precision": 0.5 + delta,
        "delta_macro_average_precision": delta,
        "p_value": 0.01,
        "p_value_holm": 0.02,
        "ci95_low": 0.001,
        "ci95_high": 0.005,
    }


def augmented_completion() -> dict[str, object]:
    baseline_run_id = BASELINE_RUN_ID
    return {
        "status": "complete",
        "run_id": "d" * 32,
        "experiment": "bge2_sft_oodtrain_e1_lr1e5_v1",
        "experiment_group": "sft",
        "training_report": {
            "validation_splits": {
                "iid": {"macro_average_precision": 0.503},
                "hard": {"macro_average_precision": 0.49},
                "ood": copy.deepcopy(baseline_builder.OOD_SENTINEL),
            },
            "args": {},
        },
        "baseline_comparison": {
            "status": "ready_ood_disabled",
            "baseline_run_id": baseline_run_id,
            "method": "paired_component_permutation",
            "splits": {
                "iid": split_result(),
                "hard": split_result(-0.001),
                "ood": workflow.comparator._ood_result(),
            },
        },
    }


class CandidateNotebookTest(unittest.TestCase):
    def test_baseline_context_requires_private_positive_version_and_no_ood(self) -> None:
        normalized = candidate_builder.validate_baseline_context(baseline_context())
        self.assertEqual(normalized["dataset_version"], 7)
        self.assertTrue(normalized["manifest"]["is_private"])
        self.assertIsNone(normalized["manifest"]["ood"]["comparison"])

        public = baseline_context()
        public["manifest"]["is_private"] = False
        with self.assertRaises(candidate_builder.CandidateBuildError):
            candidate_builder.validate_baseline_context(public)
        zero = baseline_context()
        zero["dataset_version"] = 0
        with self.assertRaises(candidate_builder.CandidateBuildError):
            candidate_builder.validate_baseline_context(zero)

    def test_lr_notebook_gates_before_training_and_binds_current_source(self) -> None:
        notebook, entry = build_candidate(
            candidate_builder.lr_variant_spec("lr1e5"),
            baseline_parent(),
        )
        tags = [tuple(cell.metadata.get("tags", [])) for cell in notebook.cells]
        self.assertLess(
            tags.index(("frozen", "baseline-dataset-gate")),
            tags.index(("frozen", "training")),
        )
        self.assertEqual(entry["expected_config"]["learning_rate"], 1e-5)
        self.assertEqual(entry["expected_config"]["epochs"], 1)
        self.assertEqual(entry["source_sha256"], baseline_context()["binding"]["source_sha256"])
        metadata = notebook.metadata["product_matching_training"]
        self.assertEqual(
            metadata["frozen_baseline_dataset"]["dataset_version"], 7
        )
        self.assertIsNone(metadata["ood_comparison"])

    def test_e2_inherits_selected_lr_parent_instead_of_baseline(self) -> None:
        _, lr_entry = build_candidate(
            candidate_builder.lr_variant_spec("lr1e5"),
            baseline_parent(),
        )
        selected = selected_candidate_parent(lr_entry)
        e2_spec = candidate_builder.e2_variant_spec(selected)
        _, e2_entry = build_candidate(e2_spec, selected)
        self.assertEqual(e2_entry["parent"]["run_id"], "c" * 32)
        self.assertNotEqual(e2_entry["parent"]["run_id"], BASELINE_RUN_ID)
        self.assertEqual(e2_entry["expected_config"]["learning_rate"], 1e-5)
        self.assertEqual(e2_entry["expected_config"]["epochs"], 2)
        inherited = dict(selected["config"])
        inherited["epochs"] = 2
        self.assertEqual(e2_entry["expected_config"], inherited)

        mismatched = dict(e2_spec)
        mismatched["learning_rate"] = 4e-5
        with self.assertRaises(candidate_builder.CandidateBuildError):
            build_candidate(mismatched, selected)


class CandidateControllerTest(unittest.TestCase):
    def test_default_is_network_free_plan_only(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            workflow.kaggle,
            "kaggle_command",
            side_effect=AssertionError("Kaggle CLI must not be resolved"),
        ), redirect_stdout(output):
            self.assertEqual(workflow.main([]), 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["mode"], "plan_only")
        self.assertFalse(payload["mutation"])
        self.assertEqual(payload["stages"][0]["candidates"], ["lr1e5", "lr4e5"])

    def test_candidate_sources_attach_exact_baseline_dataset(self) -> None:
        entry = {
            "validation_dataset": "owner/validation",
            "checkpoint_dataset": "owner/checkpoint",
            "baseline_context": baseline_context(),
        }
        self.assertEqual(
            workflow.expected_dataset_sources(entry),
            [
                "owner/validation",
                "owner/checkpoint",
                baseline_context()["dataset_ref"],
                workflow.baseline_launcher.CREDENTIALS_DATASET,
            ],
        )

    def test_remote_version_mismatch_blocks_before_manifest_or_mutation(self) -> None:
        authority = {"context": baseline_context()}
        with mock.patch.object(
            workflow.dataset_push,
            "dataset_status",
            return_value={"status": "ready", "current_version_number": 8},
        ), mock.patch.object(
            workflow.baseline_uploader,
            "verify_remote_dataset",
            side_effect=AssertionError("manifest check must not run after mismatch"),
        ), self.assertRaises(workflow.CandidateWorkflowError):
            workflow.verify_remote_baseline_dataset(["kaggle"], authority)

    def test_remote_baseline_gate_is_last_action_before_push(self) -> None:
        order: list[str] = []
        entry = {"kernel_slug": "candidate-slug"}
        with mock.patch.object(
            workflow,
            "validate_staged_kernel_metadata",
            side_effect=lambda _: order.append("local"),
        ), mock.patch.object(
            workflow.baseline_launcher,
            "confirm_remote_absence",
            side_effect=lambda *_: order.append("absence"),
        ), mock.patch.object(
            workflow,
            "verify_remote_baseline_dataset",
            side_effect=lambda *_: order.append("baseline_gate"),
        ), mock.patch.object(
            workflow.kaggle,
            "run_command",
            side_effect=lambda *_args, **_kwargs: (
                order.append("push") or SimpleNamespace(stdout="", returncode=0)
            ),
        ), mock.patch.object(
            workflow,
            "verify_remote_candidate_sources",
            side_effect=lambda *_args, **_kwargs: order.append("post_push_verify"),
        ):
            workflow.push_candidate_after_final_gates(
                ["kaggle"],
                kernel_ref="owner/candidate-slug",
                entry=entry,
                authority={},
                run_timeout=60,
            )
        self.assertEqual(
            order,
            ["local", "absence", "baseline_gate", "push", "post_push_verify"],
        )

    def test_staged_metadata_requires_all_four_exact_private_sources(self) -> None:
        context = baseline_context()
        entry = {
            "kernel_slug": "candidate-slug",
            "title": "candidate-slug",
            "validation_dataset": "owner/validation",
            "checkpoint_dataset": "owner/checkpoint",
            "baseline_context": context,
        }
        with tempfile.TemporaryDirectory() as raw:
            stage_root = Path(raw)
            stage = stage_root / entry["kernel_slug"]
            stage.mkdir()
            (stage / "notebook.ipynb").write_text("{}", encoding="utf-8")
            metadata = {
                "id": f"{OWNER}/candidate-slug",
                "title": "candidate-slug",
                "code_file": "notebook.ipynb",
                "language": "python",
                "kernel_type": "notebook",
                "is_private": True,
                "enable_gpu": True,
                "enable_tpu": False,
                "machine_shape": "NvidiaTeslaT4",
                "dataset_sources": workflow.expected_dataset_sources(entry),
                "competition_sources": [],
                "kernel_sources": [],
                "model_sources": [],
            }
            (stage / "kernel-metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with mock.patch.object(workflow.kaggle, "STAGE_ROOT", stage_root), mock.patch.object(
                workflow.candidate_builder, "load_candidate_notebook"
            ):
                self.assertEqual(
                    workflow.validate_staged_kernel_metadata(entry)["dataset_sources"],
                    metadata["dataset_sources"],
                )
                metadata["dataset_sources"] = metadata["dataset_sources"][:-1]
                (stage / "kernel-metadata.json").write_text(
                    json.dumps(metadata), encoding="utf-8"
                )
                with self.assertRaises(workflow.CandidateWorkflowError):
                    workflow.validate_staged_kernel_metadata(entry)

    def test_candidate_output_requires_exact_gate_parent_and_sheets_validation(self) -> None:
        context = baseline_context()
        parent = baseline_parent()
        expected_gate = {
            "status": "passed",
            "dataset_ref": context["dataset_ref"],
            "dataset_version": context["dataset_version"],
            "manifest_sha256": context["manifest_sha256"],
            "baseline_run_id": context["binding"]["baseline_run_id"],
            "baseline_identity_sha256": context["binding"]["campaign_identity_sha256"],
            "baseline_source_sha256": context["binding"]["source_sha256"],
            "ood_predictions": False,
        }
        expected_parent = {
            "run_id": parent["run_id"],
            "experiment": parent["experiment"],
            "campaign_identity_sha256": parent["campaign_identity_sha256"],
            "source_sha256": parent["source_sha256"],
            "recipe_sha256": parent["recipe_sha256"],
            "config": parent["config"],
        }
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            (directory / candidate_builder.BASELINE_GATE_FILENAME).write_text(
                json.dumps(expected_gate), encoding="utf-8"
            )
            completion = {
                "frozen_baseline_dataset": expected_gate,
                "stage_parent": expected_parent,
                "candidate_generator_sha256": "9" * 64,
            }
            (directory / "notebook_completed.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            entry = {
                "baseline_context": context,
                "parent": parent,
                "candidate_generator_sha256": "9" * 64,
            }
            with mock.patch.object(
                workflow.baseline_launcher,
                "validate_run_output",
                return_value={"run_id": "d" * 32},
            ) as validator:
                result = workflow.validate_candidate_output(directory, entry=entry)
            validator.assert_called_once()
            self.assertEqual(result["baseline_dataset_gate"], expected_gate)
            completion["stage_parent"]["run_id"] = "e" * 32
            (directory / "notebook_completed.json").write_text(
                json.dumps(completion), encoding="utf-8"
            )
            with mock.patch.object(
                workflow.baseline_launcher,
                "validate_run_output",
                return_value={},
            ), self.assertRaises(workflow.CandidateWorkflowError):
                workflow.validate_candidate_output(directory, entry=entry)

    def test_ood_metric_is_minus_one_but_comparison_fields_are_blank(self) -> None:
        completion = augmented_completion()
        projection = workflow.validate_augmented_completion(
            completion,
            expected_baseline_run_id=BASELINE_RUN_ID,
        )
        self.assertEqual(projection["ood_macro_ap"], -1.0)
        for field in (
            "ood_delta",
            "ood_p_value",
            "ood_p_holm",
            "ood_ci95_low",
            "ood_ci95_high",
        ):
            self.assertEqual(projection[field], "")

    def test_epoch_selection_uses_only_strict_iid_margin_and_selected_parent(self) -> None:
        _, lr_entry = build_candidate(
            candidate_builder.lr_variant_spec("lr1e5"),
            baseline_parent(),
        )
        parent = selected_candidate_parent(lr_entry)
        raw_e2_completion = augmented_completion()
        raw_e2_completion["experiment"] = "bge2_sft_oodtrain_e2_lr1e5_v1"
        raw_e2_completion["run_id"] = "e" * 32
        raw_e2_completion["frozen_recipe_sha256"] = "f" * 64
        raw_e2_completion.pop("baseline_comparison")
        baseline = {
            "run_id": BASELINE_RUN_ID,
            "experiment": current_baseline_entry()["experiment"],
            "manifest_sha256": baseline_context()["manifest_sha256"],
        }
        anchor = {
            "run_id": parent["run_id"],
            "experiment": parent["experiment"],
        }
        e2 = {
            "run_id": raw_e2_completion["run_id"],
            "experiment": raw_e2_completion["experiment"],
            "completion": raw_e2_completion,
        }
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            for name in (
                "selected-e1",
                "e2-output",
                "baseline-stage",
                "baseline-output",
            ):
                (root / name).mkdir()
            lr_receipt_path = root / "lr_selection_receipt.json"
            lr_receipt_path.write_text("{}\n", encoding="utf-8")
            lr_receipt = {
                "selected_parent": parent,
                "selected_directory": root / "selected-e1",
                "_receipt_path": lr_receipt_path,
            }
            authority = {
                "stage_dir": root / "baseline-stage",
                "source_dir": root / "baseline-output",
                "context": baseline_context(),
            }
            iid = split_result(0.002)
            hard = split_result(0.2)
            with mock.patch.object(workflow, "validate_candidate_output"), mock.patch.object(
                workflow.comparator,
                "load_frozen_baseline",
                return_value=baseline,
            ), mock.patch.object(
                workflow.comparator,
                "load_candidate",
                side_effect=[anchor, e2],
            ), mock.patch.object(
                workflow,
                "_paired_epoch_split",
                side_effect=[iid, hard],
            ):
                result = workflow.summarize_epoch_stage(
                    authority,
                    lr_receipt,
                    {"experiment": raw_e2_completion["experiment"]},
                    root / "e2-output",
                    output_dir=root / "comparison",
                    sync_sheets=False,
                    permutations=10,
                    bootstrap_resamples=10,
                )
            selection = result["summary"]["selection"]
            self.assertEqual(selection["selected_run_id"], parent["run_id"])
            self.assertEqual(selection["selected_epoch"], 1)
            self.assertFalse(result["summary"]["hard_used_for_selection"])
            self.assertFalse(result["summary"]["ood_used_for_selection"])
            self.assertIsNone(
                result["comparison"]["splits"]["ood"]["delta_macro_average_precision"]
            )
            self.assertEqual(result["receipt"]["parent"], parent)

    def test_terminal_failure_never_resubmits(self) -> None:
        entry = {"kernel_slug": "candidate-slug"}
        with mock.patch.object(workflow, "_local_output", return_value=None), mock.patch.object(
            workflow.baseline_launcher,
            "remote_kernel_status",
            return_value="error",
        ), mock.patch.object(
            workflow.kaggle,
            "run_command",
            return_value=SimpleNamespace(returncode=0, stdout=""),
        ), mock.patch.object(
            workflow,
            "push_candidate_after_final_gates",
        ) as push, mock.patch.object(
            workflow,
            "download_candidate_output",
        ) as download, self.assertRaises(workflow.CandidateWorkflowError):
            workflow.execute_candidate(
                cli=["kaggle"],
                env_file=ROOT / ".env",
                authority={},
                entry=entry,
                poll_interval=5,
                wait_timeout=60,
                run_timeout=60,
                full_download=False,
            )
        push.assert_not_called()
        download.assert_not_called()

    def test_comparison_sync_marker_is_exact(self) -> None:
        completion = augmented_completion()
        marker = {
            "status": "synced_comparison",
            "run_id": completion["run_id"],
            "experiment_group": "sft",
            "comparison_sheet": "sft_exps",
            "spreadsheet_id": workflow.shared.EXPERIMENT_SPREADSHEET_ID,
            "completion_canonical_sha256": candidate_builder.canonical_sha256(
                completion
            ),
        }
        workflow.validate_comparison_sync_marker(marker, completion=completion)
        marker["comparison_sheet"] = "wrong"
        with self.assertRaises(workflow.CandidateWorkflowError):
            workflow.validate_comparison_sync_marker(marker, completion=completion)

    def test_e2_loader_rejects_minimal_or_unbound_family_summary(self) -> None:
        parent = baseline_parent()
        authority = {
            "baseline_parent": parent,
            "context": baseline_context(),
            "source_dir": Path("/tmp/not-used-baseline-output"),
        }
        planned = [spec["experiment"] for spec in candidate_builder.LR_SPECS]
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            summary_path = directory / "family_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "selection": {
                            "selected_with_practical_tie_break": parent[
                                "experiment"
                            ]
                        }
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            receipt = {
                "schema_version": 1,
                "status": "complete",
                "campaign": baseline_builder.CAMPAIGN,
                "stage": "lr_log_line",
                "family_name": workflow.LR_FAMILY_NAME,
                "planned_experiments": planned,
                "execution_order": planned,
                "primary_split": "iid",
                "diagnostic_splits": ["hard"],
                "practical_tie_margin": workflow.comparator.PRACTICAL_TIE_MARGIN,
                "ood": {"macro_average_precision": -1.0, "comparison": None},
                "frozen_baseline_dataset": workflow._baseline_dataset_receipt(
                    authority
                ),
                "family_summary_path": str(summary_path),
                "family_summary_sha256": workflow.comparator.sha256_file(
                    summary_path
                ),
                "selected_source": "baseline",
                "selected_directory": str(authority["source_dir"]),
                "selected_parent": parent,
                "comparison_sheets_synced": True,
                "comparison_sync_markers": {},
                "boundary_expansion": "deferred",
            }
            receipt_path = directory / workflow.LR_RECEIPT_FILENAME
            receipt_path.write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                workflow.CandidateWorkflowError,
                "family summary protocol differs",
            ):
                workflow.load_lr_selection_receipt(
                    receipt_path,
                    authority=authority,
                )


if __name__ == "__main__":
    unittest.main()
