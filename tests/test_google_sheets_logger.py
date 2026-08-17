from __future__ import annotations

import json
import io
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, call, patch

from src.google_sheets_logger import (
    CATEGORY_HEADERS,
    EXPERIMENT_HEADERS,
    SheetsApiError,
    SheetsLoggerError,
    SheetsRestClient,
    _upsert_categories,
    _upsert_experiment,
    build_category_rows,
    build_experiment_row,
    column_letter,
    ensure_tables,
    kaggle_service_account_json,
    safe_error_message,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_cross_encoder_training_notebook as cross_builder
import create_mxbai_training_notebook as mxbai_builder
import create_qwen_training_notebook as qwen_builder


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        payload: object | None = None,
        *,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text
        self.content = b"" if payload is None else json.dumps(payload).encode("utf-8")

    def json(self) -> object:
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


def sample_completion() -> dict[str, object]:
    return {
        "run_id": "run-20260813-001",
        "completed_at_utc": "2026-08-13T10:15:30Z",
        "status": "complete",
        "experiment": "mxbai-balanced",
        "dataset_ref": "owner/product-matching-training",
        "kaggle_kernel_ref": "owner/product-matching-mxbai-training",
        "code_bundle_sha256": "abc123",
        "training_wall_seconds": 120.5,
        "training_report": {
            "training_seconds": 100.25,
            "validation_seconds": 12.5,
            "total_pipeline_seconds": 119.75,
            "training_examples": 1024,
            "original_training_examples": 900,
            "validation_examples": 128,
            "validation_positive_examples": 32,
            "validation_positive_rate": 0.25,
            "macro_average_precision": 0.8125,
            "overall_average_precision": 0.845,
            "validation_splits": {
                "iid": {
                    "macro_average_precision": 0.8125,
                    "overall_average_precision": 0.845,
                },
                "hard": {
                    "macro_average_precision": 0.4321,
                    "overall_average_precision": 0.4567,
                },
                "ood": {
                    "macro_average_precision": 0.6789,
                    "overall_average_precision": 0.7012,
                },
            },
            "examples_per_second": 10.214,
            "padding_efficiency": 0.91,
            "peak_vram_gib_by_rank": [10.25, 11.75],
            "mean_score_order_gap": float("nan"),
            "per_category_average_precision": {
                "Телефоны": 0.93,
                "Аксессуары": 0.71,
                "Пустая": float("inf"),
            },
            "args": {
                "model": "mixedbread-ai/mxbai-rerank-xsmall-v1",
                "epochs": 1,
                "batch_size": 64,
                "eval_batch_size": 128,
                "gradient_accumulation": 2,
                "learning_rate": 1e-5,
                "weight_decay": 0.01,
                "warmup_ratio": 0.05,
                "max_length": 384,
                "attention_implementation": "eager",
                "sampling": "none",
                "train_subset": "all",
                "loss_weighting": "none",
                "lexical_hard_negative_strength": 0.0,
                "symmetric_validation": True,
                "label_smoothing": 0.02,
                "seed": 42,
            },
        },
    }


def sample_service_account_json() -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "example-project",
            "private_key": "-----BEGIN PRIVATE KEY-----\ntest-only\n-----END PRIVATE KEY-----\n",
            "client_email": "writer@example-project.iam.gserviceaccount.com",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    )


def sheet_metadata(*, experiments_columns: int = 50) -> dict[str, object]:
    return {
        "sheets": [
            {
                "properties": {
                    "sheetId": 101,
                    "title": "experiments_v2",
                    "gridProperties": {"columnCount": experiments_columns},
                }
            },
            {
                "properties": {
                    "sheetId": 202,
                    "title": "category_metrics",
                    "gridProperties": {"columnCount": 50},
                }
            },
        ]
    }


class RowBuildingTest(unittest.TestCase):
    def test_experiment_row_contains_three_protocol_scores_and_core_config(self) -> None:
        completion = sample_completion()
        row = build_experiment_row(
            completion,
            synced_at_utc="2026-08-13T10:16:00Z",
        )
        values = dict(zip(EXPERIMENT_HEADERS, row, strict=True))

        self.assertEqual(values["run_id"], "run-20260813-001")
        self.assertEqual(
            values["model"], "mixedbread-ai/mxbai-rerank-xsmall-v1"
        )
        self.assertEqual(values["iid_macro_ap"], 0.8125)
        self.assertEqual(values["hard_macro_ap"], 0.4321)
        self.assertEqual(values["ood_macro_ap"], 0.6789)
        self.assertEqual(values["iid_overall_ap"], 0.845)
        self.assertEqual(values["hard_overall_ap"], 0.4567)
        self.assertEqual(values["ood_overall_ap"], 0.7012)
        self.assertEqual(values["train_pairs"], 900)
        self.assertEqual(values["batch_size"], 64)
        self.assertNotIn("validation_seconds", EXPERIMENT_HEADERS)
        self.assertNotIn("report_json", EXPERIMENT_HEADERS)

    def test_category_rows_are_sorted_and_non_finite_scores_are_blank(self) -> None:
        rows = build_category_rows(sample_completion())

        self.assertEqual(
            [row[4] for row in rows],
            ["Аксессуары", "Пустая", "Телефоны"],
        )
        self.assertTrue(all(row[0] == "run-20260813-001" for row in rows))
        self.assertEqual(rows[1][5], "")
        self.assertEqual(rows[2][5], 0.93)

    def test_empty_run_id_is_rejected_before_building_rows(self) -> None:
        completion = sample_completion()
        completion["run_id"] = "  "

        with self.assertRaisesRegex(SheetsLoggerError, "run_id"):
            build_experiment_row(completion)
        with self.assertRaisesRegex(SheetsLoggerError, "run_id"):
            build_category_rows(completion)


class CredentialLoadingTest(unittest.TestCase):
    @patch("src.google_sheets_logger.kaggle_secret")
    def test_kaggle_secret_is_preferred_over_dataset(self, secret: Mock) -> None:
        expected = sample_service_account_json()
        secret.return_value = expected

        with tempfile.TemporaryDirectory() as directory:
            actual = kaggle_service_account_json(input_root=Path(directory))

        self.assertEqual(actual, expected)
        secret.assert_called_once_with("GOOGLE_SERVICE_ACCOUNT_JSON")

    @patch("src.google_sheets_logger.kaggle_secret")
    def test_private_dataset_is_used_when_secret_is_unavailable(
        self,
        secret: Mock,
    ) -> None:
        secret.side_effect = SheetsLoggerError("secret unavailable")
        expected = sample_service_account_json()

        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = (
                Path(directory) / "ecom-matching-google-sheets-credentials"
            )
            dataset_dir.mkdir()
            (dataset_dir / "google-service-account.json").write_text(
                expected,
                encoding="utf-8",
            )

            actual = kaggle_service_account_json(input_root=Path(directory))

        self.assertEqual(actual, expected)

    @patch("src.google_sheets_logger.kaggle_secret")
    def test_expanded_private_dataset_layout_is_supported(
        self,
        secret: Mock,
    ) -> None:
        secret.side_effect = SheetsLoggerError("secret unavailable")
        expected = sample_service_account_json()

        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = (
                Path(directory)
                / "ecom-matching-google-sheets-credentials"
                / "expanded-version"
            )
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "google-service-account.json").write_text(
                expected,
                encoding="utf-8",
            )

            actual = kaggle_service_account_json(input_root=Path(directory))

        self.assertEqual(actual, expected)

    @patch("src.google_sheets_logger.kaggle_secret")
    def test_private_dataset_uses_new_kaggle_owner_layout(
        self,
        secret: Mock,
    ) -> None:
        secret.side_effect = SheetsLoggerError("secret unavailable")
        expected = sample_service_account_json()

        with tempfile.TemporaryDirectory() as directory:
            dataset_dir = (
                Path(directory)
                / "datasets"
                / "alexproger23"
                / "ecom-matching-google-sheets-credentials"
            )
            dataset_dir.mkdir(parents=True)
            (dataset_dir / "google-service-account.json").write_text(
                expected,
                encoding="utf-8",
            )

            actual = kaggle_service_account_json(input_root=Path(directory))

        self.assertEqual(actual, expected)

    @patch("src.google_sheets_logger.kaggle_secret")
    def test_missing_credentials_error_does_not_leak_secret_error(
        self,
        secret: Mock,
    ) -> None:
        secret.side_effect = RuntimeError("sensitive credential material")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SheetsLoggerError) as raised:
                kaggle_service_account_json(input_root=Path(directory))

        self.assertNotIn("sensitive credential material", str(raised.exception))
        self.assertIn("ecom-matching-google-sheets-credentials", str(raised.exception))


class RestClientTest(unittest.TestCase):
    def test_ranges_are_url_encoded_and_updates_use_raw_values(self) -> None:
        request = Mock(return_value=FakeResponse(200, {"updatedRows": 1}))
        client = SheetsRestClient(
            spreadsheet_id="sheet/id with space",
            access_token="access-token",
            request=request,
            sleep=Mock(),
        )

        client.update_values("'odd sheet'!A1:B1", [["=1+1", "текст"]])

        request.assert_called_once_with(
            "PUT",
            "https://sheets.googleapis.com/v4/spreadsheets/"
            "sheet%2Fid%20with%20space/values/%27odd%20sheet%27%21A1%3AB1",
            headers={
                "Authorization": "Bearer access-token",
                "Content-Type": "application/json; charset=utf-8",
            },
            params={"valueInputOption": "RAW"},
            json={
                "majorDimension": "ROWS",
                "values": [["=1+1", "текст"]],
            },
            timeout=30.0,
        )

    def test_transient_responses_retry_with_bounded_backoff(self) -> None:
        request = Mock(
            side_effect=[
                FakeResponse(429, {"error": {"message": "quota"}}),
                FakeResponse(503, {"error": {"message": "unavailable"}}),
                FakeResponse(200, {"values": [["ok"]]}),
            ]
        )
        sleep = Mock()
        client = SheetsRestClient(
            spreadsheet_id="sheet-id",
            access_token="token",
            request=request,
            sleep=sleep,
        )

        values = client.get_values("'experiments'!A1:A")

        self.assertEqual(values, [["ok"]])
        self.assertEqual(request.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(0.5), call(1.0)])

    def test_non_transient_error_is_not_retried_and_can_be_sanitized(self) -> None:
        private_key = (
            "-----BEGIN PRIVATE KEY-----\nvery-secret-material\n"
            "-----END PRIVATE KEY-----"
        )
        request = Mock(
            return_value=FakeResponse(
                403,
                {
                    "error": {
                        "message": f"denied Bearer token-value {private_key}"
                    }
                },
            )
        )
        sleep = Mock()
        client = SheetsRestClient(
            spreadsheet_id="sheet-id",
            access_token="token-value",
            request=request,
            sleep=sleep,
        )

        with self.assertRaises(SheetsApiError) as raised:
            client.get_values("'experiments'!A1:A")

        rendered = safe_error_message(raised.exception)
        self.assertEqual(request.call_count, 1)
        sleep.assert_not_called()
        self.assertIn("HTTP 403", rendered)
        self.assertIn("Bearer [redacted]", rendered)
        self.assertIn("[private key redacted]", rendered)
        self.assertNotIn("token-value", rendered)
        self.assertNotIn("very-secret-material", rendered)

    def test_network_exception_does_not_copy_exception_text_into_error(self) -> None:
        request = Mock(side_effect=OSError("failed with super-secret-token"))
        sleep = Mock()
        client = SheetsRestClient(
            spreadsheet_id="sheet-id",
            access_token="token",
            request=request,
            sleep=sleep,
            max_attempts=1,
        )

        with self.assertRaises(SheetsApiError) as raised:
            client.metadata()

        self.assertIn("OSError", str(raised.exception))
        self.assertNotIn("super-secret-token", str(raised.exception))


class TableSetupTest(unittest.TestCase):
    def test_missing_tables_are_created_and_headers_are_initialized(self) -> None:
        client = Mock()
        client.metadata.side_effect = [{"sheets": []}, sheet_metadata()]
        client.get_values.return_value = []

        sheet_ids = ensure_tables(client)

        self.assertEqual(sheet_ids, {"experiments_v2": 101})
        creation_requests = client.batch_update_spreadsheet.call_args_list[0].args[0]
        self.assertEqual(
            [request["addSheet"]["properties"]["title"] for request in creation_requests],
            ["experiments_v2"],
        )
        self.assertEqual(
            client.update_values.call_args_list[0],
            call("'experiments_v2'!A1:U1", [EXPERIMENT_HEADERS]),
        )

    def test_existing_prefix_headers_are_extended_without_shifting_columns(self) -> None:
        client = Mock()
        client.metadata.return_value = sheet_metadata()
        client.get_values.return_value = [list(EXPERIMENT_HEADERS[:5])]

        ensure_tables(client)

        self.assertEqual(
            client.update_values.call_args_list,
            [
                call("'experiments_v2'!A1:U1", [EXPERIMENT_HEADERS]),
            ],
        )

    def test_incompatible_header_aborts_before_any_data_write(self) -> None:
        client = Mock()
        client.metadata.return_value = sheet_metadata()
        client.get_values.return_value = [["wrong_run_id"]]

        with self.assertRaisesRegex(SheetsLoggerError, "incompatible header"):
            ensure_tables(client)

        client.update_values.assert_not_called()
        client.batch_update_spreadsheet.assert_not_called()

    def test_compact_experiment_header_stays_below_z(self) -> None:
        self.assertLessEqual(len(EXPERIMENT_HEADERS), 26)
        self.assertEqual(column_letter(26), "Z")
        self.assertEqual(column_letter(27), "AA")
        self.assertEqual(column_letter(len(EXPERIMENT_HEADERS)), "U")


class UpsertTest(unittest.TestCase):
    def test_new_experiment_is_appended_and_second_sync_updates_same_row(self) -> None:
        row = build_experiment_row(
            sample_completion(), synced_at_utc="2026-08-13T10:16:00Z"
        )
        client = Mock()
        client.max_attempts = 4
        client.sleep = Mock()
        client.get_values.side_effect = [[], [row]]

        first = _upsert_experiment(client, row)
        second = _upsert_experiment(client, row)

        self.assertEqual(first, "appended")
        self.assertEqual(second, "updated")
        client.append_values_once.assert_called_once_with(
            "'experiments_v2'!A:U", [row]
        )
        client.update_values.assert_called_once_with(
            "'experiments_v2'!A2:U2", [row]
        )

    def test_uncertain_append_is_not_repeated_when_run_id_was_committed(self) -> None:
        row = build_experiment_row(
            sample_completion(), synced_at_utc="2026-08-13T10:16:00Z"
        )
        client = Mock()
        client.max_attempts = 4
        client.sleep = Mock()
        client.get_values.side_effect = [[], [[row[0]]]]
        client.append_values_once.side_effect = SheetsApiError(
            "connection dropped after commit", transient=True
        )

        action = _upsert_experiment(client, row)

        self.assertEqual(action, "appended")
        self.assertEqual(client.append_values_once.call_count, 1)
        client.sleep.assert_not_called()

    def test_duplicate_experiment_run_ids_are_rejected(self) -> None:
        row = build_experiment_row(
            sample_completion(), synced_at_utc="2026-08-13T10:16:00Z"
        )
        client = Mock()
        client.get_values.return_value = [[row[0]], [row[0]]]

        with self.assertRaisesRegex(SheetsLoggerError, "duplicate run_id"):
            _upsert_experiment(client, row)

        client.update_values.assert_not_called()
        client.append_values_once.assert_not_called()


    def test_category_rows_append_then_update_without_duplicate_append(self) -> None:
        rows = build_category_rows(sample_completion())
        client = Mock()
        client.max_attempts = 4
        client.sleep = Mock()
        client.get_values.side_effect = [[], rows]

        first = _upsert_categories(client, 202, rows, "run-20260813-001")
        second = _upsert_categories(client, 202, rows, "run-20260813-001")

        self.assertEqual(first, "appended")
        self.assertEqual(second, "updated")
        client.append_values_once.assert_called_once_with(
            "'category_metrics'!A:F", rows
        )
        client.batch_update_values.assert_called_once()
        updates = client.batch_update_values.call_args.args[0]
        self.assertEqual(
            [range_name for range_name, _ in updates],
            [
                "'category_metrics'!A2:F2",
                "'category_metrics'!A3:F3",
                "'category_metrics'!A4:F4",
            ],
        )

    def test_changed_category_set_replaces_old_run_rows_bottom_up(self) -> None:
        completion = sample_completion()
        incoming = build_category_rows(completion)[:2]
        existing = build_category_rows(completion)
        client = Mock()
        client.max_attempts = 4
        client.sleep = Mock()
        client.get_values.return_value = existing

        action = _upsert_categories(
            client,
            202,
            incoming,
            "run-20260813-001",
        )

        self.assertEqual(action, "replaced")
        deletion_requests = client.batch_update_spreadsheet.call_args.args[0]
        self.assertEqual(
            [
                request["deleteDimension"]["range"]["startIndex"]
                for request in deletion_requests
            ],
            [3, 2, 1],
        )
        client.append_values_once.assert_called_once_with(
            "'category_metrics'!A:F", incoming
        )

    def test_duplicate_category_key_is_rejected_without_mutation(self) -> None:
        rows = build_category_rows(sample_completion())
        client = Mock()
        client.get_values.return_value = [rows[0], rows[0]]

        with self.assertRaisesRegex(SheetsLoggerError, "duplicate key"):
            _upsert_categories(client, 202, rows, "run-20260813-001")

        client.batch_update_values.assert_not_called()
        client.batch_update_spreadsheet.assert_not_called()
        client.append_values_once.assert_not_called()


class NotebookIntegrationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "dataset": "owner/training-data",
            "code_bundle": {
                "sha256": "a" * 64,
                "source": {"schema_version": 1, "files": {}},
            },
        }
        self.config = {
            "model": "model/name",
            "epochs": 1,
            "batch_size": 8,
            "max_length": 128,
            "learning_rate": 1e-5,
        }

    def _assert_tracking_contract(self, notebook: object) -> None:
        cells = notebook.cells
        code_cells = [cell for cell in cells if cell.cell_type == "code"]
        for index, cell in enumerate(code_cells):
            compile(cell.source, f"generated-cell-{index}", "exec")
        sources = "\n".join(cell.source for cell in cells)
        run_cell_index = next(
            index for index, cell in enumerate(cells)
            if "EXPERIMENT_RUN_ID" in cell.source
            and "RUN_STARTED_PATH" in cell.source
        )
        train_cell_index = next(
            index for index, cell in enumerate(cells)
            if "torch.distributed.run" in cell.source
        )
        self.assertLess(run_cell_index, train_cell_index)
        self.assertIn("sync_from_kaggle_credentials", code_cells[-1].source)
        self.assertIn("sheets_sync_pending.json", code_cells[-1].source)
        self.assertIn("except Exception as error", code_cells[-1].source)
        self.assertIn("GOOGLE_SERVICE_ACCOUNT_JSON", sources)
        self.assertNotIn('"private_key_id":', sources)
        self.assertNotIn('"client_email":', sources)

    def test_cross_and_qwen_notebooks_have_non_fatal_tracking(self) -> None:
        self._assert_tracking_contract(
            cross_builder.build_notebook(self.manifest, self.config)
        )
        self._assert_tracking_contract(qwen_builder.build_notebook(self.manifest))
        self.assertNotIn(
            Path("src/google_sheets_logger.py"),
            qwen_builder.BUNDLE_FILES,
        )
        self.assertIn(
            "embedded_logger_path.write_text",
            qwen_builder.build_notebook(self.manifest).cells[-1].source,
        )

    def test_mxbai_adaptation_survives_tracking_cell_insertion(self) -> None:
        notebook = mxbai_builder.build_notebook(self.manifest, self.config)
        self._assert_tracking_contract(notebook)
        sources = "\n".join(cell.source for cell in notebook.cells)
        self.assertIn("mxbai_balanced_items.parquet", sources)
        self.assertIn("prepared_sources", sources)
        self.assertIn("TRAIN_CONFIG =", sources)

    def test_embedded_cell_turns_missing_secret_into_pending_artifact(self) -> None:
        notebook = cross_builder.build_notebook(self.manifest, self.config)
        fake_secrets = types.ModuleType("kaggle_secrets")

        class FakeSecretsClient:
            def get_secret(self, name: str) -> str:
                self.last_name = name
                return "not valid service account JSON"

        fake_secrets.UserSecretsClient = FakeSecretsClient
        fake_google = types.ModuleType("google")
        fake_google_auth = types.ModuleType("google.auth")
        fake_google.auth = fake_google_auth
        fake_requests = types.ModuleType("requests")

        with tempfile.TemporaryDirectory() as directory:
            namespace = {
                "WORKING_ROOT": Path(directory),
                "completion": sample_completion(),
                "json": json,
                "subprocess": subprocess,
                "sys": sys,
            }
            with patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.auth": fake_google_auth,
                    "kaggle_secrets": fake_secrets,
                    "requests": fake_requests,
                },
            ):
                with redirect_stdout(io.StringIO()):
                    exec(notebook.cells[-1].source, namespace)

            result = json.loads(
                (Path(directory) / "google_sheets_sync.json").read_text(
                    encoding="utf-8"
                )
            )
            pending = (Path(directory) / "sheets_sync_pending.json").read_text(
                encoding="utf-8"
            )
            self.assertEqual(result["status"], "pending")
            self.assertEqual(result["run_id"], "run-20260813-001")
            self.assertNotIn("not valid service account JSON", pending)


if __name__ == "__main__":
    unittest.main()
