from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from scripts import sync_local_experiment_to_google_sheet as local_sync
from scripts.sync_local_experiment_to_google_sheet import (
    build_local_completion,
    load_optional_dotenv,
    load_training_report,
    service_account_json,
)
from src.google_sheets_logger import (
    EXPERIMENT_HEADERS,
    SheetsLoggerError,
    build_experiment_row,
)


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "run_human_reranker_baseline.sh"


def sample_report() -> dict[str, object]:
    return {
        "original_training_examples": 306_669,
        "validation_splits": {
            "iid": {
                "macro_average_precision": 0.71,
                "overall_average_precision": 0.72,
            },
            "hard": {
                "macro_average_precision": 0.31,
                "overall_average_precision": 0.32,
            },
            "ood": {
                "macro_average_precision": 0.61,
                "overall_average_precision": 0.62,
            },
        },
        "args": {
            "model": "model/from-report",
            "epochs": 1,
            "batch_size": 192,
            "gradient_accumulation": 1,
            "learning_rate": 1e-5,
            "max_length": 384,
            "seed": 42,
        },
    }


class LocalSyncTest(unittest.TestCase):
    def test_report_requires_all_three_protocols(self) -> None:
        report = sample_report()
        del report["validation_splits"]["ood"]  # type: ignore[index]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(SheetsLoggerError, "ood"):
                load_training_report(path)

    def test_report_rejects_non_finite_metrics(self) -> None:
        report = sample_report()
        report["validation_splits"]["hard"][  # type: ignore[index]
            "macro_average_precision"
        ] = float("nan")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training_report.json"
            path.write_text(json.dumps(report), encoding="utf-8")

            with self.assertRaisesRegex(SheetsLoggerError, "non-finite"):
                load_training_report(path)

    def test_local_completion_builds_compact_three_protocol_row(self) -> None:
        completion = build_local_completion(
            sample_report(),
            experiment="server-baseline",
            model=None,
            dataset_ref="owner/frozen-data",
            run_id="server-run-1",
            started_at_utc="2026-08-16T10:00:00Z",
            completed_at_utc="2026-08-16T10:10:00Z",
        )

        values = dict(zip(EXPERIMENT_HEADERS, build_experiment_row(completion)))

        self.assertEqual(values["model"], "model/from-report")
        self.assertEqual(values["iid_macro_ap"], 0.71)
        self.assertEqual(values["hard_macro_ap"], 0.31)
        self.assertEqual(values["ood_macro_ap"], 0.61)
        self.assertEqual(values["train_pairs"], 306_669)
        self.assertNotIn("validation_seconds", values)

    def test_explicit_key_is_loaded_without_printing_credential(self) -> None:
        credential = json.dumps(
            {
                "type": "service_account",
                "client_email": "writer@example.iam.gserviceaccount.com",
                "private_key": "test-private-key",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "key.json"
            path.write_text(credential, encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(service_account_json(path), credential)

    def test_optional_dotenv_populates_local_sync_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "GOOGLE_SERVICE_ACCOUNT_JSON_PATH=/secure/key.json\n"
                "EXPERIMENT_SPREADSHEET_ID=test-sheet\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_optional_dotenv(path)
                self.assertEqual(
                    os.environ["GOOGLE_SERVICE_ACCOUNT_JSON_PATH"],
                    "/secure/key.json",
                )
                self.assertEqual(
                    os.environ["EXPERIMENT_SPREADSHEET_ID"], "test-sheet"
                )

    def test_main_writes_completion_and_invokes_compact_sync(self) -> None:
        credential = {
            "type": "service_account",
            "client_email": "writer@example.iam.gserviceaccount.com",
            "private_key": "test-private-key",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report_path = root / "training_report.json"
            report_path.write_text(json.dumps(sample_report()), encoding="utf-8")
            key_path = root / "key.json"
            key_path.write_text(json.dumps(credential), encoding="utf-8")
            missing_env = root / "missing.env"
            arguments = [
                "sync_local_experiment_to_google_sheet.py",
                "--report",
                str(report_path),
                "--experiment",
                "server-baseline",
                "--run-id",
                "server-run-1",
                "--key",
                str(key_path),
                "--env-file",
                str(missing_env),
            ]
            sync_result = {
                "run_id": "server-run-1",
                "experiment_action": "appended",
            }
            with (
                patch.dict(os.environ, {}, clear=True),
                patch.object(sys, "argv", arguments),
                patch.object(
                    local_sync,
                    "sync_experiment",
                    return_value=sync_result,
                ) as sync,
                redirect_stdout(StringIO()),
            ):
                local_sync.main()

            completion = json.loads(
                (root / "server_run_completed.json").read_text(encoding="utf-8")
            )
            result = json.loads(
                (root / "google_sheets_sync.json").read_text(encoding="utf-8")
            )
            self.assertEqual(result["status"], "synced")
            self.assertEqual(completion["run_id"], "server-run-1")
            self.assertEqual(
                set(completion["training_report"]["validation_splits"]),
                {"iid", "hard", "ood"},
            )
            self.assertEqual(
                sync.call_args.kwargs["spreadsheet_id"],
                local_sync.DEFAULT_EXPERIMENT_SPREADSHEET_ID,
            )


class LauncherDryRunTest(unittest.TestCase):
    profiles = {
        "gte": "Alibaba-NLP/gte-multilingual-reranker-base",
        "jina-v3.5": "jinaai/jina-reranker-v3.5",
        "jina-v2": "jinaai/jina-reranker-v2-base-multilingual",
        "bge-v2-m3": "BAAI/bge-reranker-v2-m3",
        "qwen-0.6b": "Qwen/Qwen3-Reranker-0.6B",
        "qwen-4b": "Qwen/Qwen3-Reranker-4B",
        "rumodernbert": "deepvk/RuModernBERT-base",
    }

    def dry_run(self, profile: str) -> str:
        environment = os.environ.copy()
        environment.update(
            {
                "DRY_RUN": "1",
                "PYTHON_BIN": "/fake/python",
                "TOKEN_CACHE_DIR": "/tmp/product-matching-test-cache",
                "PREPARED_DIR": "/tmp/product-matching-test-data",
            }
        )
        if profile != "all":
            environment.update(
                {
                    "OUTPUT_DIR": "/tmp/product-matching-test-output",
                    "RUN_ID": "test-run-id",
                }
            )
        else:
            for name in ("OUTPUT_DIR", "RUN_ID", "EXPERIMENT_NAME"):
                environment.pop(name, None)
        result = subprocess.run(
            ["bash", str(LAUNCHER), profile],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout

    def test_all_profiles_train_and_sync_three_splits(self) -> None:
        for profile, model in self.profiles.items():
            with self.subTest(profile=profile):
                output = self.dry_run(profile)
                self.assertIn(model, output)
                self.assertEqual(output.count("--validation-split"), 3)
                self.assertIn("sync_local_experiment_to_google_sheet.py", output)
                self.assertIn("experiments_v2", output)

    def test_backend_specific_profiles_are_not_silently_interchanged(self) -> None:
        gte = self.dry_run("gte")
        self.assertIn("--attention-implementation eager", gte)
        self.assertIn("--learning-rate 2e-5", gte)
        self.assertIn("--max-grad-norm 0.5", gte)
        self.assertIn("--model-backend jina_lbnl", self.dry_run("jina-v3.5"))
        self.assertIn("--training-mode full", self.dry_run("qwen-0.6b"))
        self.assertIn("--training-mode lora", self.dry_run("qwen-4b"))
        self.assertIn("random_head", self.dry_run("rumodernbert"))

    def test_all_runs_every_profile_sequentially(self) -> None:
        output = self.dry_run("all")

        self.assertEqual(output.count("Training profile:"), len(self.profiles))
        self.assertEqual(
            output.count("Post-training experiments_v2 sync:"), len(self.profiles)
        )


if __name__ == "__main__":
    unittest.main()
