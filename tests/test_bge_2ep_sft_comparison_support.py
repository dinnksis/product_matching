from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts import create_bge_2ep_sft_notebooks as builder
from scripts import push_bge_2ep_sft_baseline_dataset as uploader
from scripts import summarize_bge_2ep_sft_comparisons as summarizer
from src.experiment_significance import holm_adjust, macro_average_precision
from src.google_sheets_logger import COMPARISON_HEADERS, build_comparison_row


HASHES = {
    "validation_manifest_sha256": "1" * 64,
    "initial_checkpoint_manifest_sha256": "2" * 64,
    "initial_checkpoint_model_sha256": "3" * 64,
    "code_bundle_sha256": "4" * 64,
    "frozen_recipe_sha256": "5" * 64,
    "campaign_identity_sha256": "6" * 64,
    "executable_cells_sha256": "7" * 64,
    "loss_hook_sha256": "8" * 64,
}
BASELINE_EXPERIMENT = "bge2_test_baseline"
BASELINE_RUN_ID = "a" * 32
ASSERTIONS = unittest.TestCase()


def prediction_frame(scores: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id1": [1, 3, 5, 7, 11, 13, 15, 17],
            "id2": [2, 4, 6, 8, 12, 14, 16, 18],
            "target": [0, 0, 1, 1, 0, 0, 1, 1],
            "category_1": ["a"] * 4 + ["b"] * 4,
            "score": scores,
            "product_text_1": ["must not be staged"] * 8,
        }
    )


def macro_ap(frame: pd.DataFrame) -> float:
    return macro_average_precision(
        frame["target"].to_numpy(),
        frame["score"].to_numpy(),
        frame["category_1"].to_numpy(),
    )


def report(iid: pd.DataFrame, hard: pd.DataFrame) -> dict[str, object]:
    return {
        "evaluated_validation_splits": ["iid", "hard"],
        "ood_evaluation_policy": "disabled_train_contaminated",
        "validation_splits": {
            "iid": {"macro_average_precision": macro_ap(iid)},
            "hard": {"macro_average_precision": macro_ap(hard)},
            "ood": copy.deepcopy(builder.OOD_SENTINEL),
        },
    }


def baseline_entry() -> dict[str, object]:
    return {
        "experiment": BASELINE_EXPERIMENT,
        "role": "baseline",
        "kernel_slug": "bge2-test-baseline-kernel",
        "checkpoint_dataset": "owner/checkpoint",
        "validation_dataset": "owner/validation",
        "validation_manifest_sha256": HASHES["validation_manifest_sha256"],
        "checkpoint_manifest_sha256": HASHES[
            "initial_checkpoint_manifest_sha256"
        ],
        "checkpoint_model_sha256": HASHES["initial_checkpoint_model_sha256"],
        "source_sha256": HASHES["code_bundle_sha256"],
        "recipe_sha256": HASHES["frozen_recipe_sha256"],
        "identity_sha256": HASHES["campaign_identity_sha256"],
        "executable_cells_sha256": HASHES["executable_cells_sha256"],
        "loss_hook_sha256": HASHES["loss_hook_sha256"],
    }


def completion(
    *,
    experiment: str,
    run_id: str,
    role: str,
    iid: pd.DataFrame,
    hard: pd.DataFrame,
    identity_digit: str = "6",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "complete",
        "run_id": run_id,
        "experiment": experiment,
        "experiment_group": "sft",
        "campaign": builder.CAMPAIGN,
        "role": role,
        "model": "owner/checkpoint",
        "dataset_ref": "owner/validation",
        "initial_checkpoint_ref": "owner/checkpoint",
        "ood_evaluation_policy": "disabled_train_contaminated",
        "training_report": report(iid, hard),
        **HASHES,
    }
    payload["campaign_identity_sha256"] = identity_digit * 64
    return payload


def write_run(
    directory: Path,
    *,
    completion_payload: dict[str, object],
    iid: pd.DataFrame,
    hard: pd.DataFrame,
    nested: bool = False,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "notebook_completed.json").write_text(
        json.dumps(completion_payload, ensure_ascii=False), encoding="utf-8"
    )
    artifact_dir = directory / "model-output" if nested else directory
    artifact_dir.mkdir(parents=True, exist_ok=True)
    iid.to_parquet(artifact_dir / "iid_validation_predictions.parquet", index=False)
    hard.to_parquet(artifact_dir / "hard_validation_predictions.parquet", index=False)


def fake_source_validator(source_dir: Path, *, entry: object) -> dict[str, object]:
    del entry
    return json.loads(
        (source_dir / "notebook_completed.json").read_text(encoding="utf-8")
    )


def build_frozen_baseline(root: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    iid = prediction_frame([0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5])
    hard = prediction_frame([0.75, 0.65, 0.55, 0.45, 0.75, 0.65, 0.55, 0.45])
    source = root / "source-baseline"
    write_run(
        source,
        completion_payload=completion(
            experiment=BASELINE_EXPERIMENT,
            run_id=BASELINE_RUN_ID,
            role="baseline",
            iid=iid,
            hard=hard,
        ),
        iid=iid,
        hard=hard,
        nested=True,
    )
    (source / "training.log").write_text("not staged", encoding="utf-8")
    stage = root / "frozen-baseline"
    with mock.patch.dict(
        uploader.EXPECTED_ROWS, {"iid": 8, "hard": 8}, clear=True
    ):
        uploader.build_payload(
            source,
            stage,
            "owner",
            entry=baseline_entry(),
            source_validator=fake_source_validator,
        )
    return stage, iid, hard


def _check_uploader_defaults_to_no_network_mode() -> None:
    args = uploader.parse_args([])
    assert args.upload is False
    assert args.dry_run is False
    with ASSERTIONS.assertRaises(SystemExit), redirect_stderr(io.StringIO()):
        uploader.parse_args(["--dry-run", "--upload"])


def _check_default_main_never_resolves_kaggle_cli() -> None:
    manifest = {
        "dataset": "owner/product-matching-bge-2ep-sft-baseline-v1",
        "binding": {
            "baseline_run_id": BASELINE_RUN_ID,
            "campaign_identity_sha256": "6" * 64,
        },
        "files": {"notebook_completed.json": {"bytes": 123}},
    }
    with tempfile.TemporaryDirectory() as temporary:
        stage = Path(temporary) / "stage"
        with (
            mock.patch.dict(os.environ, {"KAGGLE_USERNAME": "owner"}),
            mock.patch.object(uploader.kaggle, "load_dotenv"),
            mock.patch.object(
                uploader, "expected_baseline_entry", return_value=baseline_entry()
            ),
            mock.patch.object(
                uploader, "default_source_dir", return_value=Path(temporary) / "source"
            ),
            mock.patch.object(uploader, "build_payload", return_value=manifest),
            mock.patch.object(
                uploader, "verify_payload_for_upload", return_value="f" * 64
            ),
            mock.patch.object(uploader.kaggle, "kaggle_command") as kaggle_command,
            redirect_stdout(io.StringIO()),
        ):
            result = uploader.main(["--stage-dir", str(stage)])
    assert result == 0
    kaggle_command.assert_not_called()


def _check_uploader_stages_exact_completion_and_only_compact_iid_hard() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        stage, _, _ = build_frozen_baseline(root)
        source = root / "source-baseline"
        manifest = json.loads(
            (stage / uploader.MANIFEST_FILENAME).read_text(encoding="utf-8")
        )

        assert manifest["binding"]["baseline_run_id"] == BASELINE_RUN_ID
        assert manifest["binding"]["campaign_identity_sha256"] == "6" * 64
        assert manifest["binding"]["source_sha256"] == "4" * 64
        assert manifest["binding"]["recipe_sha256"] == "5" * 64
        assert manifest["binding"]["executable_cells_sha256"] == "7" * 64
        assert manifest["binding"]["loss_hook_sha256"] == "8" * 64
        assert manifest["binding"]["checkpoint_manifest_sha256"] == "2" * 64
        assert manifest["ood"]["metric_sentinel"] == -1.0
        assert manifest["ood"]["comparison"] is None
        assert (stage / "notebook_completed.json").read_bytes() == (
            source / "notebook_completed.json"
        ).read_bytes()
        assert {path.name for path in stage.iterdir()} == {
            "dataset-metadata.json",
            "notebook_completed.json",
            "iid_validation_predictions.parquet",
            "hard_validation_predictions.parquet",
            uploader.MANIFEST_FILENAME,
        }
        assert not list(stage.rglob("*ood*"))
        assert not list(stage.rglob("model.safetensors*"))
        for filename in uploader.PREDICTION_FILENAMES.values():
            staged = pd.read_parquet(stage / filename)
            assert list(staged.columns) == list(uploader.COMPACT_COLUMNS)
            assert "product_text_1" not in staged
        assert uploader.verify_payload_for_upload(stage, manifest) == uploader.sha256_file(
            stage / uploader.MANIFEST_FILENAME
        )


def _check_uploader_rejects_overlap_ood_and_postbuild_tamper() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        stage, iid, hard = build_frozen_baseline(root)
        source = root / "source-baseline"
        with ASSERTIONS.assertRaisesRegex(uploader.BaselineDatasetError, "disjoint"):
            uploader.build_payload(
                source,
                source,
                "owner",
                entry=baseline_entry(),
                source_validator=fake_source_validator,
            )
        iid.to_parquet(source / "ood_validation_predictions.parquet", index=False)
        with ASSERTIONS.assertRaisesRegex(
            uploader.BaselineDatasetError, "forbidden OOD"
        ):
            uploader.build_payload(
                source,
                root / "another-stage",
                "owner",
                entry=baseline_entry(),
                source_validator=fake_source_validator,
            )
        (source / "ood_validation_predictions.parquet").unlink()
        manifest = json.loads(
            (stage / uploader.MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        path = stage / "iid_validation_predictions.parquet"
        payload = path.read_bytes()
        path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
        with ASSERTIONS.assertRaisesRegex(uploader.BaselineDatasetError, "drifted"):
            uploader.verify_payload_for_upload(stage, manifest)


def _check_complete_family_uses_iid_primary_holm_and_never_compares_ood() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frozen, _, _ = build_frozen_baseline(root)
        strong_iid = prediction_frame([0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.9, 0.8])
        strong_hard = prediction_frame([0.2, 0.1, 0.8, 0.9, 0.2, 0.1, 0.8, 0.9])
        tie_iid = prediction_frame([0.8, 0.7, 0.6, 0.5, 0.8, 0.7, 0.6, 0.5])
        tie_hard = prediction_frame([0.75, 0.65, 0.55, 0.45, 0.75, 0.65, 0.55, 0.45])
        candidate_a = root / "candidate-a"
        candidate_b = root / "candidate-b"
        write_run(
            candidate_a,
            completion_payload=completion(
                experiment="candidate_a",
                run_id="b" * 32,
                role="candidate",
                iid=strong_iid,
                hard=strong_hard,
                identity_digit="9",
            ),
            iid=strong_iid,
            hard=strong_hard,
            nested=True,
        )
        write_run(
            candidate_b,
            completion_payload=completion(
                experiment="candidate_b",
                run_id="c" * 32,
                role="candidate",
                iid=tie_iid,
                hard=tie_hard,
                identity_digit="d",
            ),
            iid=tie_iid,
            hard=tie_hard,
        )

        with mock.patch.dict(
            summarizer.EXPECTED_ROWS, {"iid": 8, "hard": 8}, clear=True
        ):
            summary = summarizer.summarize_candidate_family(
                frozen,
                [candidate_b, candidate_a],
                planned_experiments=["candidate_a", "candidate_b"],
                family_name="lr_log_line_v1",
                permutations=39,
                bootstrap_resamples=39,
                seed=7,
            )

        assert summary["primary_split"] == "iid"
        assert summary["diagnostic_splits"] == ["hard"]
        assert summary["selection"]["selected_with_practical_tie_break"] == "candidate_a"
        assert summary["selection"]["hard_used_for_selection"] is False
        assert summary["selection"]["ood_used_for_selection"] is False
        raw_iid = {
            name: summary["candidate_outputs"][name]["comparison"]["splits"]["iid"][
                "p_value"
            ]
            for name in ("candidate_a", "candidate_b")
        }
        expected_adjusted = holm_adjust(raw_iid)
        for name in ("candidate_a", "candidate_b"):
            comparison = summary["candidate_outputs"][name]["comparison"]
            assert comparison["splits"]["iid"]["p_value_holm"] == expected_adjusted[name]
            assert comparison["splits"]["iid"]["holm_family_size"] == 2
            assert comparison["splits"]["hard"]["holm_family_size"] == 2
            assert comparison["splits"]["ood"] == {
                "evaluated": False,
                "status": "disabled_train_contaminated",
                "reason": "former frozen OOD pairs are part of BGE supervised training",
                "examples": 0,
                "delta_macro_average_precision": None,
                "p_value": None,
                "p_value_holm": None,
                "ci95_low": None,
                "ci95_high": None,
            }
            augmented = summary["candidate_outputs"][name]["augmented_completion"]
            projection = dict(
                zip(COMPARISON_HEADERS, build_comparison_row(augmented), strict=True)
            )
            assert projection["baseline_run_id"] == BASELINE_RUN_ID
            assert projection["ood_macro_ap"] == -1.0
            assert projection["ood_delta"] == ""
            assert projection["ood_p_value"] == ""
            assert projection["comparison_status"] == "ready_ood_disabled"

        outputs = summarizer.materialize_summary(summary, root / "reports")
        assert Path(outputs["summary"]).is_file()
        assert Path(outputs["candidates"]["candidate_a"]["comparison"]).is_file()
        assert Path(outputs["candidates"]["candidate_b"]["completion"]).is_file()


def _check_incomplete_planned_holm_family_and_candidate_ood_are_rejected() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        frozen, _, _ = build_frozen_baseline(root)
        iid = prediction_frame([0.1, 0.2, 0.9, 0.8, 0.1, 0.2, 0.9, 0.8])
        hard = prediction_frame([0.2, 0.1, 0.8, 0.9, 0.2, 0.1, 0.8, 0.9])
        candidate = root / "candidate"
        write_run(
            candidate,
            completion_payload=completion(
                experiment="candidate_a",
                run_id="b" * 32,
                role="candidate",
                iid=iid,
                hard=hard,
                identity_digit="9",
            ),
            iid=iid,
            hard=hard,
        )
        with (
            mock.patch.dict(
                summarizer.EXPECTED_ROWS, {"iid": 8, "hard": 8}, clear=True
            ),
            ASSERTIONS.assertRaisesRegex(
                summarizer.BgeComparisonError, "family must be complete"
            ),
        ):
            summarizer.summarize_candidate_family(
                frozen,
                [candidate],
                planned_experiments=["candidate_a", "candidate_b"],
                family_name="lr_log_line_v1",
                permutations=9,
                bootstrap_resamples=9,
            )

        iid.to_parquet(candidate / "ood_validation_predictions.parquet", index=False)
        with ASSERTIONS.assertRaisesRegex(
            summarizer.BgeComparisonError, "forbidden OOD"
        ):
            summarizer.load_candidate(candidate, baseline=summarizer.load_frozen_baseline(frozen))


def _check_manifest_hash_is_bound_to_exact_completion_bytes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        stage, _, _ = build_frozen_baseline(Path(temporary))
        manifest = read_manifest = json.loads(
            (stage / uploader.MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        completion_bytes = (stage / "notebook_completed.json").read_bytes()
        assert manifest["files"]["notebook_completed.json"]["sha256"] == hashlib.sha256(
            completion_bytes
        ).hexdigest()
        assert read_manifest["files"]["notebook_completed.json"][
            "exact_source_copy"
        ] is True


class BgeSftComparisonSupportTest(unittest.TestCase):
    def test_uploader_defaults_to_no_network_mode(self) -> None:
        _check_uploader_defaults_to_no_network_mode()

    def test_uploader_stages_exact_completion_and_only_compact_iid_hard(self) -> None:
        _check_uploader_stages_exact_completion_and_only_compact_iid_hard()

    def test_default_main_never_resolves_kaggle_cli(self) -> None:
        _check_default_main_never_resolves_kaggle_cli()

    def test_uploader_rejects_overlap_ood_and_postbuild_tamper(self) -> None:
        _check_uploader_rejects_overlap_ood_and_postbuild_tamper()

    def test_complete_family_uses_iid_primary_holm_and_never_compares_ood(self) -> None:
        _check_complete_family_uses_iid_primary_holm_and_never_compares_ood()

    def test_incomplete_planned_holm_family_and_candidate_ood_are_rejected(self) -> None:
        _check_incomplete_planned_holm_family_and_candidate_ood_are_rejected()

    def test_manifest_hash_is_bound_to_exact_completion_bytes(self) -> None:
        _check_manifest_hash_is_bound_to_exact_completion_bytes()


if __name__ == "__main__":
    unittest.main()
