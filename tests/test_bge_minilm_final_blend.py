from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import evaluate_bge_minilm_final_blend as evaluator


TARGET = [0, 0, 1, 1, 0, 0, 1, 1]
CATEGORIES = ["a"] * 4 + ["b"] * 4
IID_BGE = [0.5103, 0.3354, 0.0294, 0.6763, 0.0572, 0.0485, 0.2652, 0.0911]
IID_MINILM = [0.7261, 0.0151, 0.6456, 0.3392, 0.9723, 0.5757, 0.6646, 0.8524]
HARD_BGE = [0.3987, 0.8660, 0.3333, 0.6232, 0.8428, 0.1274, 0.1775, 0.7237]
HARD_MINILM = [0.1332, 0.0267, 0.1968, 0.3969, 0.6986, 0.6962, 0.2200, 0.2300]
BGE_SLUG = "pm-b2-lbce-123456789abc-s17-l1"
BGE_IDENTITY = "1" * 64
BGE_RECIPE = "2" * 64
PARENT = {
    "run_id": "b" * 32,
    "experiment": "selected_e1_or_e2",
    "campaign_identity_sha256": "3" * 64,
    "source_sha256": "4" * 64,
    "recipe_sha256": "5" * 64,
    "checkpoint_manifest_sha256": "6" * 64,
    "checkpoint_model_sha256": "7" * 64,
    "validation_manifest_sha256": "8" * 64,
    "loss_hook_sha256": evaluator.LOSS_HOOK_SHA256[evaluator.PLAIN_BCE],
    "config": {"epochs": 2, "seed": 42},
}


def prediction_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id1": [1, 3, 5, 7, 11, 13, 15, 17],
            "id2": [2, 4, 6, 8, 12, 14, 16, 18],
            "target": TARGET,
            "category_1": CATEGORIES,
            "score": np.asarray(scores, dtype=np.float32),
        }
    )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_parquets(
    directory: Path,
    *,
    iid: pd.DataFrame,
    hard: pd.DataFrame,
    nested: bool = False,
) -> dict[str, Path]:
    output = directory / "model" if nested else directory
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "iid": output / "iid_validation_predictions.parquet",
        "hard": output / "hard_validation_predictions.parquet",
    }
    iid.to_parquet(paths["iid"], index=False)
    hard.to_parquet(paths["hard"], index=False)
    return paths


def write_bge_completion(directory: Path, frames: dict[str, pd.DataFrame]) -> Path:
    validation: dict[str, object] = {}
    for split, frame in frames.items():
        metrics = evaluator._score_metrics(
            frame, frame["score"].to_numpy(dtype=np.float64)
        )
        validation[split] = {
            "examples": len(frame),
            "macro_average_precision": metrics["macro_average_precision"],
            "overall_average_precision": metrics["overall_average_precision"],
        }
    validation["ood"] = {
        "evaluated": False,
        "macro_average_precision": -1.0,
        "overall_average_precision": -1.0,
        "predictions_file": None,
    }
    payload = {
        "status": "complete",
        "run_id": "a" * 32,
        "experiment": evaluator.SEED17_EXPERIMENT[evaluator.PLAIN_BCE],
        "campaign_identity_sha256": BGE_IDENTITY,
        "frozen_recipe_sha256": BGE_RECIPE,
        "loss_hook_sha256": evaluator.LOSS_HOOK_SHA256[evaluator.PLAIN_BCE],
        "loss_variant": evaluator.PLAIN_BCE,
        "loss_confirmation": {
            "workflow": evaluator.WORKFLOW,
            "stage": "seed_confirmation",
            "seed": 17,
            "loss_variant": evaluator.PLAIN_BCE,
            "parent": {"run_id": PARENT["run_id"]},
            "primary_split": "iid",
            "diagnostic_splits": ["hard"],
            "ood_macro_average_precision": -1.0,
            "ood_comparison": None,
            "fresh_start": True,
            "checkpoint_resume": False,
        },
        "training_report": {
            "evaluated_validation_splits": ["iid", "hard"],
            "validation_splits": validation,
        },
    }
    path = directory / "notebook_completed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def synthetic_contract(minilm_paths: dict[str, Path]) -> dict[str, object]:
    contract = copy.deepcopy(evaluator.EXPECTED_CONTRACT)
    contract["splits"]["expected_category_count"] = 2
    for split, path in minilm_paths.items():
        contract["frozen_minilm"]["predictions"][split] = {
            "filename": path.name,
            "rows": 8,
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return contract


def prediction_binding(paths: dict[str, Path]) -> dict[str, dict[str, object]]:
    return {
        split: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for split, path in paths.items()
    }


def write_receipt_chain(
    root: Path, bge_paths: dict[str, Path], contract: dict
) -> tuple[dict[str, Path], dict]:
    receipt_root = root / "receipts"
    lr_dir = receipt_root / "lr"
    epoch_dir = receipt_root / "e2"
    screen_dir = receipt_root / "screen"
    final_dir = receipt_root / "confirm"
    for directory in (lr_dir, epoch_dir, screen_dir, final_dir):
        directory.mkdir(parents=True, exist_ok=True)
    baseline = {"dataset_ref": "owner/baseline", "dataset_version": 1}
    lr = {
        "schema_version": 1,
        "status": "complete",
        "campaign": evaluator.CAMPAIGN,
        "stage": "lr_log_line",
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "comparison_sheets_synced": True,
        "ood": {"comparison": None, "macro_average_precision": -1.0},
        "selected_parent": copy.deepcopy(PARENT),
        "selected_source": "candidate",
        "frozen_baseline_dataset": baseline,
    }
    lr_path = lr_dir / contract["bge_authority"]["lr_receipt_filename"]
    lr_path.write_text(json.dumps(lr), encoding="utf-8")
    epoch = {
        "schema_version": 1,
        "status": "complete",
        "campaign": evaluator.CAMPAIGN,
        "stage": "epoch_line",
        "comparison_sheets_synced": True,
        "epoch_3": "deferred",
        "frozen_baseline_dataset": baseline,
        "lr_selection_receipt_sha256": file_sha256(lr_path),
        "parent": copy.deepcopy(PARENT),
        "candidate_run_id": "d" * 32,
        "candidate_experiment": "unused_e2",
        "selection": {
            "selected_experiment": PARENT["experiment"],
            "selected_run_id": PARENT["run_id"],
            "selected_epoch": 1,
            "rule": "select e2 iff paired IID delta > 0.002",
        },
    }
    epoch_path = epoch_dir / contract["bge_authority"]["epoch_receipt_filename"]
    epoch_path.write_text(json.dumps(epoch), encoding="utf-8")
    binding = prediction_binding(bge_paths)
    screen = {
        "schema_version": 1,
        "status": "complete",
        "workflow": evaluator.WORKFLOW,
        "campaign": evaluator.CAMPAIGN,
        "stage": "loss_screen",
        "family_name": "test_loss_screen",
        "frozen_baseline_dataset": baseline,
        "lr_selection_receipt_sha256": file_sha256(lr_path),
        "epoch_selection_receipt_sha256": file_sha256(epoch_path),
        "anchor": {
            "parent": copy.deepcopy(PARENT),
            "directory": str((root / "anchor").resolve()),
            "prediction_binding": binding,
            "loss_variant": evaluator.PLAIN_BCE,
            "reused_existing_run": True,
        },
        "challenger": {
            "run_id": "c" * 32,
            "loss_variant": evaluator.SQRT_BALANCED_BCE,
            "prediction_binding": binding,
        },
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "iid_delta": 0.001,
        "acceptance_threshold": 0.002,
        "threshold_relation": "strictly_greater_than",
        "seed42_winner": evaluator.PLAIN_BCE,
        "challenger_accepted_for_seed17": False,
        "comparison_path": str((screen_dir / "baseline_comparison.json").resolve()),
        "comparison_sha256": "9" * 64,
        "completion_with_comparison_path": str(
            (screen_dir / "completion_with_comparison.json").resolve()
        ),
        "completion_with_comparison_sha256": "a" * 64,
        "comparison_sheets_synced": True,
        "comparison_sync_marker": {"sha256": "b" * 64},
        "kernel_budget": {},
    }
    screen_path = screen_dir / contract["bge_authority"]["screen_receipt_filename"]
    screen_path.write_text(json.dumps(screen), encoding="utf-8")
    seed17 = {
        "bce_run_id": "a" * 32,
        "bce_experiment": evaluator.SEED17_EXPERIMENT[evaluator.PLAIN_BCE],
        "bce_kernel_slug": BGE_SLUG,
        "bce_identity_sha256": BGE_IDENTITY,
        "bce_recipe_sha256": BGE_RECIPE,
        "bce_loss_hook_sha256": evaluator.LOSS_HOOK_SHA256[evaluator.PLAIN_BCE],
        "bce_parent_run_id": PARENT["run_id"],
        "bce_prediction_binding": binding,
        "challenger_run_id": None,
        "challenger_kernel_slug": None,
        "challenger_identity_sha256": None,
        "challenger_recipe_sha256": None,
        "challenger_loss_hook_sha256": None,
        "challenger_parent_run_id": None,
        "challenger_prediction_binding": None,
        "iid_delta": None,
        "comparison_artifacts": None,
    }
    final = {
        "schema_version": 1,
        "status": "complete",
        "workflow": evaluator.WORKFLOW,
        "campaign": evaluator.CAMPAIGN,
        "stage": "seed_confirmation",
        "loss_screen_receipt_sha256": file_sha256(screen_path),
        "branch": "matched_bce_seed17_only",
        "execution_order": [BGE_SLUG],
        "seed42": {
            "anchor_run_id": PARENT["run_id"],
            "challenger_run_id": "c" * 32,
            "iid_delta": 0.001,
            "screen_threshold": 0.002,
            "challenger_passed": False,
        },
        "seed17": seed17,
        "final_gate": {
            "seed42_delta_strictly_positive": True,
            "seed17_delta_strictly_positive": None,
            "mean_iid_delta": None,
            "required_mean_iid_delta": 0.002,
            "challenger_accepted": False,
        },
        "selected_loss_variant": evaluator.PLAIN_BCE,
        "selected_recipe": copy.deepcopy(PARENT["config"]),
        "selected_recipe_sha256": PARENT["recipe_sha256"],
        "selected_loss_hook_sha256": evaluator.LOSS_HOOK_SHA256[evaluator.PLAIN_BCE],
        "primary_split": "iid",
        "diagnostic_splits": ["hard"],
        "hard_used_for_selection": False,
        "ood": {"macro_average_precision": -1.0, "comparison": None},
        "seed17_comparison_sheets_synced": None,
        "seed17_comparison_sync_marker": None,
        "kernel_budget": {},
    }
    final_path = final_dir / contract["bge_authority"]["final_receipt_filename"]
    final_path.write_text(json.dumps(final), encoding="utf-8")
    paths = {"lr": lr_path, "epoch": epoch_path, "screen": screen_path, "final": final_path}
    authority = evaluator.load_final_bge_authority(
        final_receipt_path=final_path,
        screen_receipt_path=screen_path,
        lr_receipt_path=lr_path,
        epoch_receipt_path=epoch_path,
        contract=contract,
    )
    return paths, authority


def build_fixture(root: Path) -> tuple:
    bge_dir = root / BGE_SLUG
    mini_dir = root / "minilm"
    iid_bge = prediction_frame(IID_BGE)
    hard_bge = prediction_frame(HARD_BGE)
    iid_mini = prediction_frame(IID_MINILM)
    hard_mini = prediction_frame(HARD_MINILM)
    bge_paths = write_parquets(
        bge_dir, iid=iid_bge, hard=hard_bge, nested=True
    )
    mini_paths = write_parquets(mini_dir, iid=iid_mini, hard=hard_mini)
    write_bge_completion(bge_dir, {"iid": iid_bge, "hard": hard_bge})
    contract = synthetic_contract(mini_paths)
    receipt_paths, authority = write_receipt_chain(root, bge_paths, contract)
    return bge_dir, mini_dir, bge_paths, mini_paths, contract, receipt_paths, authority


def evaluate_fixture(
    bge_dir: Path,
    mini_dir: Path,
    bge_paths: dict[str, Path],
    contract: dict,
    authority: dict,
) -> dict:
    return evaluator.evaluate_final_blend(
        bge_dir=bge_dir,
        minilm_dir=mini_dir,
        bge_authority=authority,
        contract=contract,
        contract_sha256="f" * 64,
        enforce_frozen_contract=False,
    )


def reload_authority(receipts: dict[str, Path], contract: dict) -> dict:
    return evaluator.load_final_bge_authority(
        final_receipt_path=receipts["final"],
        screen_receipt_path=receipts["screen"],
        lr_receipt_path=receipts["lr"],
        epoch_receipt_path=receipts["epoch"],
        contract=contract,
    )


class FinalBlendEvaluatorTest(unittest.TestCase):
    def test_repository_contract_is_exactly_frozen(self) -> None:
        contract, digest = evaluator.load_frozen_contract()
        self.assertEqual(contract, evaluator.EXPECTED_CONTRACT)
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertEqual(contract["blend"]["bge_weights"], [0.6, 0.7])
        self.assertEqual(contract["blend"]["space"], "logit")
        self.assertFalse(contract["selection"]["hard_used_for_selection"])
        self.assertEqual(contract["ood"]["macro_average_precision"], -1.0)
        self.assertIsNone(contract["ood"]["comparison"])

    def test_success_strict_binding_metrics_deltas_and_iid_only_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bge_dir, mini_dir, bge_paths, _, contract, _, authority = build_fixture(root)
            # The selected MiniLM run has a historical OOD artifact.  The BGE
            # campaign trained on those categories, so this evaluator must not
            # inspect or validate MiniLM OOD at all.
            (mini_dir / "ood_validation_predictions.parquet").write_bytes(
                b"intentionally not a parquet"
            )
            before = {
                path: file_sha256(path)
                for path in [*bge_dir.rglob("*"), *mini_dir.rglob("*")]
                if path.is_file()
            }
            report = evaluate_fixture(bge_dir, mini_dir, bge_paths, contract, authority)
            after = {path: file_sha256(path) for path in before}

        self.assertEqual(before, after)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["method"]["evaluated_bge_weights"], [0.6, 0.7])
        self.assertEqual(
            set(report["splits"]["iid"]["models"]),
            {"bge_only", "minilm_only", "logit_bge_0p6", "logit_bge_0p7"},
        )
        self.assertEqual(report["selection"]["blend"]["selected"], "logit_bge_0p7")
        hard_models = report["splits"]["hard"]["models"]
        self.assertGreater(
            hard_models["logit_bge_0p6"]["macro_average_precision"],
            hard_models["logit_bge_0p7"]["macro_average_precision"],
        )
        self.assertFalse(report["selection"]["hard_used_for_selection"])
        self.assertIn("vs_bge", report["splits"]["iid"]["blend_deltas"]["logit_bge_0p7"])
        self.assertEqual(report["ood"]["macro_average_precision"], -1.0)
        self.assertFalse(report["claims"]["ood_claim"])
        self.assertFalse(report["claims"]["hidden_test_gain_claim"])

    def test_identical_models_use_frozen_tie_rules(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                bge_dir,
                mini_dir,
                bge_paths,
                mini_paths,
                contract,
                _,
                authority,
            ) = build_fixture(root)
            for split, path in mini_paths.items():
                frame = pd.read_parquet(bge_paths[split])
                frame.to_parquet(path, index=False)
            contract = synthetic_contract(mini_paths)
            report = evaluate_fixture(bge_dir, mini_dir, bge_paths, contract, authority)
        self.assertTrue(report["selection"]["blend"]["tie_applied"])
        self.assertEqual(report["selection"]["blend"]["selected"], "logit_bge_0p7")
        self.assertTrue(report["selection"]["final"]["tie_applied"])
        self.assertEqual(report["selection"]["final"]["selected"], "minilm_only")
        self.assertFalse(report["selection"]["final"]["recommend_blend"])

    def test_row_target_and_category_mismatches_fail_closed(self) -> None:
        mutations = {
            "id1": lambda frame: frame.__setitem__("id1", [99, *frame["id1"].tolist()[1:]]),
            "target": lambda frame: frame.__setitem__(
                "target", [1, *frame["target"].tolist()[1:]]
            ),
            "category_1": lambda frame: frame.__setitem__(
                "category_1", ["b", *frame["category_1"].tolist()[1:]]
            ),
        }
        for column, mutate in mutations.items():
            with self.subTest(column=column), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (
                    bge_dir,
                    mini_dir,
                    bge_paths,
                    mini_paths,
                    _,
                    _,
                    authority,
                ) = build_fixture(root)
                frame = pd.read_parquet(mini_paths["iid"])
                mutate(frame)
                frame.to_parquet(mini_paths["iid"], index=False)
                contract = synthetic_contract(mini_paths)
                with self.assertRaisesRegex(evaluator.FinalBlendError, "strict row binding"):
                    evaluate_fixture(
                        bge_dir, mini_dir, bge_paths, contract, authority
                    )

    def test_frozen_minilm_and_final_receipt_bge_hashes_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bge_dir, mini_dir, bge_paths, _, contract, _, authority = build_fixture(root)
            bad_minilm = copy.deepcopy(contract)
            bad_minilm["frozen_minilm"]["predictions"]["iid"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(evaluator.FinalBlendError, "MiniLM iid.*frozen e3"):
                evaluate_fixture(
                    bge_dir, mini_dir, bge_paths, bad_minilm, authority
                )
            bad_authority = copy.deepcopy(authority)
            bad_authority["prediction_binding"]["iid"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(evaluator.FinalBlendError, "BGE iid.*final receipt"):
                evaluator.evaluate_final_blend(
                    bge_dir=bge_dir,
                    minilm_dir=mini_dir,
                    bge_authority=bad_authority,
                    contract=contract,
                    contract_sha256="f" * 64,
                    enforce_frozen_contract=False,
                )

    def test_nonfinal_completed_bge_directories_are_rejected(self) -> None:
        for slug in (
            "pm-b2-base-de25c35eabf4-s42-v1",
            "pm-b2-lsqrt-abcdef123456-s42-l1",
        ):
            with self.subTest(slug=slug), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (
                    _,
                    mini_dir,
                    _,
                    _,
                    contract,
                    _,
                    authority,
                ) = build_fixture(root)
                nonfinal = root / slug
                iid = prediction_frame(IID_BGE)
                hard = prediction_frame(HARD_BGE)
                write_parquets(nonfinal, iid=iid, hard=hard, nested=True)
                write_bge_completion(nonfinal, {"iid": iid, "hard": hard})
                with self.assertRaisesRegex(
                    evaluator.FinalBlendError, "final selected kernel slug"
                ):
                    evaluator.evaluate_final_blend(
                        bge_dir=nonfinal,
                        minilm_dir=mini_dir,
                        bge_authority=authority,
                        contract=contract,
                        contract_sha256="f" * 64,
                        enforce_frozen_contract=False,
                    )

    def test_receipt_chain_run_path_and_hash_mismatches_fail_closed(self) -> None:
        mutations = {
            "screen_hash": lambda final: final.__setitem__(
                "loss_screen_receipt_sha256", "0" * 64
            ),
            "selected_run": lambda final: final["seed17"].__setitem__(
                "bce_run_id", "e" * 32
            ),
            "iid_path": lambda final: final["seed17"]["bce_prediction_binding"][
                "iid"
            ].__setitem__(
                "path",
                final["seed17"]["bce_prediction_binding"]["hard"]["path"],
            ),
            "hard_hash": lambda final: final["seed17"]["bce_prediction_binding"][
                "hard"
            ].__setitem__("sha256", "0" * 64),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                (
                    bge_dir,
                    mini_dir,
                    _,
                    _,
                    contract,
                    receipts,
                    _,
                ) = build_fixture(root)
                final = json.loads(receipts["final"].read_text())
                mutate(final)
                receipts["final"].write_text(json.dumps(final), encoding="utf-8")
                if name == "screen_hash":
                    with self.assertRaisesRegex(
                        evaluator.FinalBlendError, "loss_screen_receipt_sha256"
                    ):
                        reload_authority(receipts, contract)
                    continue
                authority = reload_authority(receipts, contract)
                expected = {
                    "selected_run": "run_id",
                    "iid_path": "iid path",
                    "hard_hash": "hard prediction SHA",
                }[name]
                with self.assertRaisesRegex(evaluator.FinalBlendError, expected):
                    evaluator.evaluate_final_blend(
                        bge_dir=bge_dir,
                        minilm_dir=mini_dir,
                        bge_authority=authority,
                        contract=contract,
                        contract_sha256="f" * 64,
                        enforce_frozen_contract=False,
                    )

    def test_two_seed_challenger_final_branch_is_bound_to_selected_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                bge_dir,
                mini_dir,
                bge_paths,
                _,
                contract,
                receipts,
                _,
            ) = build_fixture(root)
            bce_shadow_paths = write_parquets(
                root / "bce-shadow",
                iid=pd.read_parquet(bge_paths["iid"]),
                hard=pd.read_parquet(bge_paths["hard"]),
            )
            sqrt_slug = "pm-b2-lsqrt-fedcba987654-s17-l1"
            sqrt_dir = bge_dir.with_name(sqrt_slug)
            bge_dir.rename(sqrt_dir)
            sqrt_paths = {
                split: sqrt_dir / "model" / f"{split}_validation_predictions.parquet"
                for split in evaluator.SPLITS
            }
            completion_path = sqrt_dir / "notebook_completed.json"
            completion = json.loads(completion_path.read_text())
            completion.update(
                {
                    "run_id": "e" * 32,
                    "experiment": evaluator.SEED17_EXPERIMENT[
                        evaluator.SQRT_BALANCED_BCE
                    ],
                    "campaign_identity_sha256": "c" * 64,
                    "frozen_recipe_sha256": "d" * 64,
                    "loss_hook_sha256": evaluator.LOSS_HOOK_SHA256[
                        evaluator.SQRT_BALANCED_BCE
                    ],
                    "loss_variant": evaluator.SQRT_BALANCED_BCE,
                }
            )
            completion["loss_confirmation"]["loss_variant"] = (
                evaluator.SQRT_BALANCED_BCE
            )
            completion["loss_confirmation"]["parent"] = {"run_id": "a" * 32}
            completion_path.write_text(json.dumps(completion), encoding="utf-8")

            screen = json.loads(receipts["screen"].read_text())
            screen.update(
                {
                    "iid_delta": 0.003,
                    "seed42_winner": evaluator.SQRT_BALANCED_BCE,
                    "challenger_accepted_for_seed17": True,
                }
            )
            receipts["screen"].write_text(json.dumps(screen), encoding="utf-8")
            final = json.loads(receipts["final"].read_text())
            final["loss_screen_receipt_sha256"] = file_sha256(receipts["screen"])
            final["branch"] = "matched_bce_and_challenger_seed17"
            final["execution_order"] = [BGE_SLUG, sqrt_slug]
            final["seed42"].update({"iid_delta": 0.003, "challenger_passed": True})
            final["seed17"].update(
                {
                    "bce_prediction_binding": prediction_binding(bce_shadow_paths),
                    "challenger_run_id": "e" * 32,
                    "challenger_kernel_slug": sqrt_slug,
                    "challenger_identity_sha256": "c" * 64,
                    "challenger_recipe_sha256": "d" * 64,
                    "challenger_loss_hook_sha256": evaluator.LOSS_HOOK_SHA256[
                        evaluator.SQRT_BALANCED_BCE
                    ],
                    "challenger_parent_run_id": "a" * 32,
                    "challenger_prediction_binding": prediction_binding(sqrt_paths),
                    "iid_delta": 0.003,
                    "comparison_artifacts": {"comparison_sha256": "f" * 64},
                }
            )
            final["final_gate"] = {
                "seed42_delta_strictly_positive": True,
                "seed17_delta_strictly_positive": True,
                "mean_iid_delta": 0.003,
                "required_mean_iid_delta": 0.002,
                "challenger_accepted": True,
            }
            final["selected_loss_variant"] = evaluator.SQRT_BALANCED_BCE
            final["selected_loss_hook_sha256"] = evaluator.LOSS_HOOK_SHA256[
                evaluator.SQRT_BALANCED_BCE
            ]
            final["seed17_comparison_sheets_synced"] = True
            final["seed17_comparison_sync_marker"] = {"sha256": "f" * 64}
            receipts["final"].write_text(json.dumps(final), encoding="utf-8")

            authority = reload_authority(receipts, contract)
            report = evaluator.evaluate_final_blend(
                bge_dir=sqrt_dir,
                minilm_dir=mini_dir,
                bge_authority=authority,
                contract=contract,
                contract_sha256="f" * 64,
                enforce_frozen_contract=False,
            )
        self.assertEqual(
            report["inputs"]["bge"]["selected_loss_variant"],
            evaluator.SQRT_BALANCED_BCE,
        )
        self.assertEqual(report["inputs"]["bge"]["run_id"], "e" * 32)
    def test_bge_completion_and_ood_contract_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bge_dir, mini_dir, bge_paths, _, contract, _, authority = build_fixture(root)
            (bge_dir / "ood_shadow_validation_predictions.parquet").write_bytes(
                b"forbidden"
            )
            with self.assertRaisesRegex(evaluator.FinalBlendError, "forbidden OOD"):
                evaluate_fixture(bge_dir, mini_dir, bge_paths, contract, authority)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bge_dir, mini_dir, bge_paths, _, contract, _, authority = build_fixture(root)
            completion_path = bge_dir / "notebook_completed.json"
            completion = json.loads(completion_path.read_text())
            completion["training_report"]["validation_splits"]["iid"][
                "macro_average_precision"
            ] += 0.01
            completion_path.write_text(json.dumps(completion), encoding="utf-8")
            with self.assertRaisesRegex(evaluator.FinalBlendError, "differs from parquet"):
                evaluate_fixture(bge_dir, mini_dir, bge_paths, contract, authority)

    def test_boundary_scores_and_weight_expansion_are_rejected(self) -> None:
        with self.assertRaisesRegex(evaluator.FinalBlendError, "strictly inside"):
            evaluator.logit_blend(
                np.asarray([0.0, 0.5]),
                np.asarray([0.2, 0.8]),
                bge_weight=0.6,
            )
        with self.assertRaisesRegex(evaluator.FinalBlendError, "Only frozen"):
            evaluator.logit_blend(
                np.asarray([0.2, 0.5]),
                np.asarray([0.2, 0.8]),
                bge_weight=0.65,
            )

    def test_report_rendering_and_outputs_never_overwrite_inputs_or_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bge_dir, mini_dir, bge_paths, _, contract, _, authority = build_fixture(root)
            report = evaluate_fixture(
                bge_dir, mini_dir, bge_paths, contract, authority
            )
            markdown = evaluator.render_markdown(report)
            self.assertIn("hard is diagnostic", markdown)
            self.assertIn("sentinel is exactly `-1`", markdown)

            output = root / "reports" / "report.json"
            prepared = evaluator._prepare_output(output, inputs=(bge_dir, mini_dir))
            evaluator._write_new_text(prepared, "first")
            with self.assertRaisesRegex(evaluator.FinalBlendError, "overwrite"):
                evaluator._prepare_output(output, inputs=(bge_dir, mini_dir))
            with self.assertRaisesRegex(evaluator.FinalBlendError, "overwrite"):
                evaluator._write_new_text(prepared, "second")
            with self.assertRaisesRegex(evaluator.FinalBlendError, "input directory"):
                evaluator._prepare_output(
                    bge_dir / "report.json", inputs=(bge_dir, mini_dir)
                )

    def test_cli_has_no_weight_or_external_mutation_switch(self) -> None:
        args = evaluator.parse_args(
            [
                "--bge-dir",
                "/tmp/bge-final",
                "--final-receipt",
                "/tmp/loss_confirmation_receipt.json",
                "--screen-receipt",
                "/tmp/loss_screen_receipt.json",
                "--lr-receipt",
                "/tmp/lr_selection_receipt.json",
                "--epoch-receipt",
                "/tmp/epoch_selection_receipt.json",
            ]
        )
        self.assertFalse(hasattr(args, "weight"))
        self.assertFalse(hasattr(args, "upload"))
        self.assertFalse(hasattr(args, "execute"))
        self.assertFalse(hasattr(args, "sync_sheets"))


if __name__ == "__main__":
    unittest.main()
