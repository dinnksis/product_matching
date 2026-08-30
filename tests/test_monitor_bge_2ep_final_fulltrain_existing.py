from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import nbformat


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import create_bge_2ep_final_fulltrain_notebook as builder
import monitor_bge_2ep_final_fulltrain_existing as monitor


class ExistingFinalExportMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.entry = monitor.load_exact_b501_entry()
        cls.local_notebook = nbformat.read(cls.entry["notebook"], as_version=4)

    def test_new_monitor_files_are_identity_excluded_and_tuple_is_unchanged(self) -> None:
        ledger = {path.as_posix() for path in builder.SOURCE_LEDGER_FILES}
        self.assertNotIn(monitor.THIS_SOURCE.as_posix(), ledger)
        self.assertNotIn(monitor.THIS_TEST.as_posix(), ledger)
        self.assertEqual(self.entry["kernel_slug"], monitor.EXPECTED_KERNEL_SLUG)
        self.assertEqual(
            self.entry["identity_sha256"], monitor.EXPECTED_IDENTITY_SHA256
        )
        self.assertEqual(self.entry["source_sha256"], monitor.EXPECTED_SOURCE_SHA256)
        self.assertEqual(
            self.entry["executable_cells_sha256"],
            monitor.EXPECTED_EXECUTABLE_CELLS_SHA256,
        )
        self.assertEqual(
            self.entry["notebook_sha256"], monitor.EXPECTED_NOTEBOOK_SHA256
        )

    def test_exact_attempt_requires_only_b501_and_rejects_push_pending(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "attempts.json"
            payload = {
                "schema_version": 1,
                "campaign": builder.CAMPAIGN,
                "policy": "one_push_per_identity_no_automatic_resubmit_v1",
                "attempts": [
                    {
                        "kernel_slug": self.entry["kernel_slug"],
                        "identity_sha256": self.entry["identity_sha256"],
                        "status": "running",
                        "reserved_at_unix": 1.0,
                        "updated_at_unix": 2.0,
                    }
                ],
            }
            path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                monitor.require_exact_attempt(self.entry, path=path)["status"],
                "running",
            )
            payload["attempts"][0]["status"] = "push_pending"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(monitor.ExistingFinalExportMonitorError):
                monitor.require_exact_attempt(self.entry, path=path)
            payload["attempts"][0]["status"] = "running"
            payload["attempts"].append(dict(payload["attempts"][0]))
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(monitor.ExistingFinalExportMonitorError):
                monitor.require_exact_attempt(self.entry, path=path)

    def _write_pull(self, destination: Path, *, metadata: dict | None = None) -> None:
        slug = self.entry["kernel_slug"]
        (destination / f"{slug}.ipynb").write_text(
            nbformat.writes(self.local_notebook), encoding="utf-8"
        )
        (destination / "kernel-metadata.json").write_text(
            json.dumps(metadata or monitor.expected_remote_metadata(self.entry)),
            encoding="utf-8",
        )

    def test_pull_accepts_only_normalized_filename_and_validates_notebook(self) -> None:
        authority = {"current_version_number": 1, "id_no": 132_549_639}

        def fake_run(command: list[str], *, check: bool = True):
            destination = Path(command[command.index("-p") + 1])
            self._write_pull(destination)
            return SimpleNamespace(returncode=0, stdout="")

        with mock.patch.object(
            monitor, "read_remote_v1_authority", return_value=authority
        ) as version_gate, mock.patch.object(
            monitor.kaggle, "run_command", side_effect=fake_run
        ) as run:
            result = monitor.pull_and_validate_remote_v1(
                ["kaggle"], mock.sentinel.api, owner=monitor.EXPECTED_OWNER,
                entry=self.entry,
            )
        self.assertEqual(version_gate.call_count, 2)
        command = run.call_args.args[0]
        self.assertEqual(command[1:3], ["kernels", "pull"])
        self.assertEqual(
            command[3], f"{monitor.EXPECTED_OWNER}/{self.entry['kernel_slug']}"
        )
        self.assertNotIn("/1", command[3])
        self.assertEqual(
            result["notebook"]["identity_sha256"],
            monitor.EXPECTED_IDENTITY_SHA256,
        )

    def test_pull_retries_transport_once_but_not_content_validation(self) -> None:
        authority = {"current_version_number": 1, "id_no": 132_549_639}
        calls = 0

        def fake_run(command: list[str], *, check: bool = True):
            nonlocal calls
            calls += 1
            if calls == 1:
                return SimpleNamespace(returncode=1, stdout="proxy error")
            destination = Path(command[command.index("-p") + 1])
            self._write_pull(destination)
            return SimpleNamespace(returncode=0, stdout="")

        with mock.patch.object(
            monitor, "read_remote_v1_authority", return_value=authority
        ) as version_gate, mock.patch.object(
            monitor.kaggle, "run_command", side_effect=fake_run
        ) as run, mock.patch.object(monitor, "_transient_backoff") as backoff:
            monitor.pull_and_validate_remote_v1(
                ["kaggle"], mock.sentinel.api, owner=monitor.EXPECTED_OWNER,
                entry=self.entry,
            )
        self.assertEqual(run.call_count, 2)
        self.assertEqual(version_gate.call_count, 4)
        backoff.assert_called_once_with(1)

    def test_pull_rejects_old_local_code_filename(self) -> None:
        metadata = monitor.expected_remote_metadata(self.entry)
        metadata["code_file"] = "notebook.ipynb"

        def fake_run(command: list[str], *, check: bool = True):
            destination = Path(command[command.index("-p") + 1])
            self._write_pull(destination, metadata=metadata)
            return SimpleNamespace(returncode=0, stdout="")

        with mock.patch.object(
            monitor, "read_remote_v1_authority", return_value={"version": 1}
        ), mock.patch.object(monitor.kaggle, "run_command", side_effect=fake_run):
            with self.assertRaisesRegex(
                monitor.ExistingFinalExportMonitorError, "metadata differs"
            ):
                monitor.pull_and_validate_remote_v1(
                    ["kaggle"], mock.sentinel.api,
                    owner=monitor.EXPECTED_OWNER, entry=self.entry,
                )

    def test_pulled_executable_tamper_is_rejected(self) -> None:
        tampered = copy.deepcopy(self.local_notebook)
        next(cell for cell in tampered.cells if cell.cell_type == "code").source += (
            "\n# tampered"
        )
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            path = root / f"{self.entry['kernel_slug']}.ipynb"
            path.write_text(nbformat.writes(tampered), encoding="utf-8")
            with self.assertRaises(builder.FinalExportConfigError):
                monitor.validate_pulled_notebook(path, root=root, entry=self.entry)

    def test_status_is_bracketed_by_unchanged_v1_authority(self) -> None:
        first = {"current_version_number": 1, "id_no": 132_549_639}
        result = SimpleNamespace(returncode=0, stdout='status "running"')
        with mock.patch.object(
            monitor, "read_remote_v1_authority", side_effect=[first, dict(first)]
        ) as gate, mock.patch.object(
            monitor.kaggle, "run_command", return_value=result
        ) as run:
            self.assertEqual(
                monitor.read_exact_v1_status(
                    ["kaggle"], mock.sentinel.api,
                    owner=monitor.EXPECTED_OWNER, entry=self.entry,
                ),
                "running",
            )
        self.assertEqual(gate.call_count, 2)
        self.assertEqual(run.call_args.args[0][1:3], ["kernels", "status"])

        with mock.patch.object(
            monitor,
            "read_remote_v1_authority",
            side_effect=[first, {"current_version_number": 2}],
        ), mock.patch.object(monitor.kaggle, "run_command", return_value=result):
            with self.assertRaisesRegex(
                monitor.ExistingFinalExportMonitorError, "authority changed"
            ):
                monitor.read_exact_v1_status(
                    ["kaggle"], mock.sentinel.api,
                    owner=monitor.EXPECTED_OWNER, entry=self.entry,
                )

    def test_status_transient_once_rebrackets_v1_and_returns_running(self) -> None:
        authority = {"current_version_number": 1, "id_no": 132_549_639}
        results = [
            SimpleNamespace(returncode=1, stdout="proxy error"),
            SimpleNamespace(returncode=0, stdout='status "running"'),
        ]
        with mock.patch.object(
            monitor, "read_remote_v1_authority", return_value=authority
        ) as gate, mock.patch.object(
            monitor.kaggle, "run_command", side_effect=results
        ) as run, mock.patch.object(monitor, "_transient_backoff") as backoff:
            status = monitor.read_exact_v1_status(
                ["kaggle"], mock.sentinel.api,
                owner=monitor.EXPECTED_OWNER, entry=self.entry,
            )
        self.assertEqual(status, "running")
        self.assertEqual(run.call_count, 2)
        self.assertEqual(gate.call_count, 4)
        backoff.assert_called_once_with(1)

    def test_get_kernel_authority_requires_exact_current_version_one(self) -> None:
        expected_pull = monitor.expected_remote_metadata(self.entry)
        metadata = SimpleNamespace(
            id=expected_pull["id_no"],
            ref=expected_pull["id"],
            title=expected_pull["title"],
            slug=self.entry["kernel_slug"],
            current_version_number=1,
            is_private=True,
            enable_gpu=True,
            enable_tpu=False,
            enable_internet=True,
            dataset_data_sources=expected_pull["dataset_sources"],
            kernel_data_sources=[],
            competition_data_sources=[],
            model_data_sources=[],
            docker_image=expected_pull["docker_image"],
            machine_shape=expected_pull["machine_shape"],
        )

        class FakeContext:
            def __enter__(self):
                endpoint = SimpleNamespace(
                    get_kernel=mock.Mock(
                        return_value=SimpleNamespace(metadata=metadata, blob="ignored")
                    )
                )
                return SimpleNamespace(
                    kernels=SimpleNamespace(kernels_api_client=endpoint)
                )

            def __exit__(self, *args):
                return False

        api = SimpleNamespace(build_kaggle_client=lambda: FakeContext())
        authority = monitor.read_remote_v1_authority(
            api, owner=monitor.EXPECTED_OWNER, entry=self.entry
        )
        self.assertEqual(authority["current_version_number"], 1)
        metadata.current_version_number = 2
        with self.assertRaisesRegex(
            monitor.ExistingFinalExportMonitorError, "authority differs"
        ):
            monitor.read_remote_v1_authority(
                api, owner=monitor.EXPECTED_OWNER, entry=self.entry
            )

    def test_get_kernel_transport_retries_without_retrying_semantic_mismatch(self) -> None:
        expected_pull = monitor.expected_remote_metadata(self.entry)
        metadata = SimpleNamespace(
            id=expected_pull["id_no"], ref=expected_pull["id"],
            title=expected_pull["title"], slug=self.entry["kernel_slug"],
            current_version_number=1, is_private=True, enable_gpu=True,
            enable_tpu=False, enable_internet=True,
            dataset_data_sources=expected_pull["dataset_sources"],
            kernel_data_sources=[], competition_data_sources=[],
            model_data_sources=[], docker_image=expected_pull["docker_image"],
            machine_shape=expected_pull["machine_shape"],
        )
        response = SimpleNamespace(metadata=metadata, blob="ignored")
        get_kernel = mock.Mock(side_effect=[OSError("proxy"), response])

        class FakeContext:
            def __enter__(self):
                endpoint = SimpleNamespace(get_kernel=get_kernel)
                return SimpleNamespace(
                    kernels=SimpleNamespace(kernels_api_client=endpoint)
                )

            def __exit__(self, *args):
                return False

        api = SimpleNamespace(build_kaggle_client=lambda: FakeContext())
        with mock.patch.object(monitor, "_transient_backoff") as backoff:
            monitor.read_remote_v1_authority(
                api, owner=monitor.EXPECTED_OWNER, entry=self.entry
            )
        self.assertEqual(get_kernel.call_count, 2)
        backoff.assert_called_once_with(1)

        metadata.current_version_number = 2
        get_kernel.reset_mock(side_effect=True)
        get_kernel.side_effect = None
        get_kernel.return_value = response
        with mock.patch.object(monitor, "_transient_backoff") as no_backoff:
            with self.assertRaisesRegex(
                monitor.ExistingFinalExportMonitorError, "authority differs"
            ):
                monitor.read_remote_v1_authority(
                    api, owner=monitor.EXPECTED_OWNER, entry=self.entry
                )
        self.assertEqual(get_kernel.call_count, 1)
        no_backoff.assert_not_called()

    def test_default_plan_declares_no_remote_or_mutating_action(self) -> None:
        attempt = {
            "kernel_slug": self.entry["kernel_slug"],
            "identity_sha256": self.entry["identity_sha256"],
            "status": "running",
            "reserved_at_unix": 1.0,
        }
        plan = monitor.plan_payload(self.entry, attempt)
        self.assertEqual(plan["default_remote_calls"], [])
        self.assertEqual(plan["remote_mutations"], [])
        self.assertFalse(plan["resubmit"])
        source = Path(monitor.__file__).read_text(encoding="utf-8")
        self.assertNotIn('"kernels", "push"', source)
        self.assertNotIn("reserve_single_attempt(", source)
        self.assertNotIn("execute_once(", source)
        self.assertNotIn("stage_locally(", source)

    def test_terminal_success_strictly_downloads_then_marks_complete(self) -> None:
        destination = ROOT / "artifacts/kaggle" / self.entry["kernel_slug"]
        attempt = {
            "kernel_slug": self.entry["kernel_slug"],
            "identity_sha256": self.entry["identity_sha256"],
            "status": "running",
            "reserved_at_unix": 1.0,
        }
        authority = {"current_version_number": 1}
        with mock.patch.object(
            monitor, "require_exact_attempt", return_value=attempt
        ), mock.patch.object(
            monitor, "pull_and_validate_remote_v1"
        ) as pull, mock.patch.object(
            monitor, "read_exact_v1_status", return_value="complete"
        ), mock.patch.object(
            monitor, "read_remote_v1_authority", return_value=authority
        ), mock.patch.object(
            monitor.frozen, "download_and_validate", return_value=destination
        ) as download, mock.patch.object(
            monitor.frozen, "validate_artifact_tree"
        ) as validate, mock.patch.object(
            monitor, "update_exact_attempt"
        ) as update:
            actual = monitor.wait_download_and_validate(
                ["kaggle"], mock.sentinel.api,
                owner=monitor.EXPECTED_OWNER, entry=self.entry,
            )
        self.assertEqual(actual, destination)
        self.assertEqual(pull.call_count, 2)
        download.assert_called_once_with(
            ["kaggle"], owner=monitor.EXPECTED_OWNER, entry=self.entry
        )
        validate.assert_called_once_with(destination, entry=self.entry)
        update.assert_called_once_with(self.entry, "complete")

    def test_terminal_failure_marks_failed_and_never_downloads(self) -> None:
        attempt = {
            "kernel_slug": self.entry["kernel_slug"],
            "identity_sha256": self.entry["identity_sha256"],
            "status": "running",
            "reserved_at_unix": 1.0,
        }
        with mock.patch.object(
            monitor, "require_exact_attempt", return_value=attempt
        ), mock.patch.object(
            monitor, "pull_and_validate_remote_v1"
        ), mock.patch.object(
            monitor, "read_exact_v1_status", return_value="failed"
        ), mock.patch.object(
            monitor, "log_terminal_failure"
        ) as logs, mock.patch.object(
            monitor, "update_exact_attempt"
        ) as update, mock.patch.object(
            monitor.frozen, "download_and_validate"
        ) as download:
            with self.assertRaisesRegex(
                monitor.ExistingFinalExportMonitorError, "no resubmit"
            ):
                monitor.wait_download_and_validate(
                    ["kaggle"], mock.sentinel.api,
                    owner=monitor.EXPECTED_OWNER, entry=self.entry,
                )
        update.assert_called_once_with(self.entry, "failed")
        logs.assert_called_once()
        download.assert_not_called()

    def test_download_transport_failure_preserves_retryable_attempt(self) -> None:
        destination = ROOT / "artifacts/kaggle" / self.entry["kernel_slug"]
        attempt = {
            "kernel_slug": self.entry["kernel_slug"],
            "identity_sha256": self.entry["identity_sha256"],
            "status": "running",
            "reserved_at_unix": 1.0,
        }
        authority = {"current_version_number": 1}
        transport_error = monitor.frozen.FinalExportWorkflowError(
            "transient output download failure"
        )
        with mock.patch.object(
            monitor, "require_exact_attempt", return_value=attempt
        ), mock.patch.object(
            monitor, "pull_and_validate_remote_v1"
        ), mock.patch.object(
            monitor, "read_exact_v1_status", return_value="complete"
        ), mock.patch.object(
            monitor, "read_remote_v1_authority", return_value=authority
        ), mock.patch.object(
            monitor.frozen,
            "download_and_validate",
            side_effect=[transport_error, destination],
        ) as download, mock.patch.object(
            monitor.frozen, "validate_artifact_tree"
        ) as validate, mock.patch.object(
            monitor, "update_exact_attempt"
        ) as update:
            with self.assertRaisesRegex(
                monitor.frozen.FinalExportWorkflowError,
                "transient output download failure",
            ):
                monitor.wait_download_and_validate(
                    ["kaggle"], mock.sentinel.api,
                    owner=monitor.EXPECTED_OWNER, entry=self.entry,
                )
            update.assert_not_called()

            # A second invocation retries the same already-existing v1 and can
            # complete without any push, reservation or new identity.
            actual = monitor.wait_download_and_validate(
                ["kaggle"], mock.sentinel.api,
                owner=monitor.EXPECTED_OWNER, entry=self.entry,
            )

        self.assertEqual(actual, destination)
        self.assertEqual(download.call_count, 2)
        validate.assert_called_once_with(destination, entry=self.entry)
        update.assert_called_once_with(self.entry, "complete")

    def test_persistent_status_transport_failure_keeps_ledger_retryable(self) -> None:
        attempt = {
            "kernel_slug": self.entry["kernel_slug"],
            "identity_sha256": self.entry["identity_sha256"],
            "status": "running",
            "reserved_at_unix": 1.0,
        }
        authority = {"current_version_number": 1}
        failed_status = SimpleNamespace(returncode=1, stdout="proxy error")
        with mock.patch.object(
            monitor, "require_exact_attempt", return_value=attempt
        ), mock.patch.object(
            monitor, "pull_and_validate_remote_v1"
        ), mock.patch.object(
            monitor, "read_remote_v1_authority", return_value=authority
        ) as gate, mock.patch.object(
            monitor.kaggle, "run_command", return_value=failed_status
        ) as run, mock.patch.object(
            monitor, "_transient_backoff"
        ) as backoff, mock.patch.object(
            monitor, "update_exact_attempt"
        ) as update:
            with self.assertRaisesRegex(
                monitor.ExistingFinalExportMonitorError, "bounded retries"
            ):
                monitor.wait_download_and_validate(
                    ["kaggle"], mock.sentinel.api,
                    owner=monitor.EXPECTED_OWNER, entry=self.entry,
                )
        self.assertEqual(run.call_count, monitor.TRANSIENT_ATTEMPTS)
        self.assertEqual(gate.call_count, 2 * monitor.TRANSIENT_ATTEMPTS)
        self.assertEqual(backoff.call_count, monitor.TRANSIENT_ATTEMPTS - 1)
        update.assert_not_called()


if __name__ == "__main__":
    unittest.main()
