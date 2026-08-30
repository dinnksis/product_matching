from __future__ import annotations

import inspect
import json
import os
import stat
import struct
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import download_minilm_5ep_sft_selected_checkpoint as downloader
import materialize_minilm_5ep_sft_loss_confirmation as adaptive


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


@contextmanager
def resolved_temporary_directory():
    with tempfile.TemporaryDirectory() as raw_directory:
        # macOS exposes /var as a symlink to /private/var.  The production
        # validator deliberately rejects such aliases, so tests use the exact
        # canonical temporary path.
        yield Path(raw_directory).resolve(strict=True)


class SelectedCheckpointDownloaderTest(unittest.TestCase):
    def _summary(self, lock_path: Path) -> tuple[dict, dict, dict]:
        plan = {"campaign": "minilm_5ep_sft_hparam_search_v1"}
        lock = {
            "schema_version": 2,
            "kind": adaptive.LOCK_KIND,
            "mode": "confirmation",
            "effective_stage": downloader.CONFIRMATION_STAGE,
            "execution_status": "runnable",
            "lock_payload_sha256": SHA_A,
        }
        selected = "recipe_123"
        summary = {
            "schema_version": 2,
            "campaign": plan["campaign"],
            "execution_status": "pending",
            "execution_lock_sha256s": [SHA_A],
            "execution_receipt_sha256s": [],
            "execution_campaign_lock_sha256s": [SHA_A],
            "effective_stage": downloader.CONFIRMATION_STAGE,
            "mode": "confirmation",
            "budget": {},
            "adaptive_closure": [
                {
                    "schema_version": 2,
                    "kind": adaptive.LOCK_KIND,
                    "mode": "confirmation",
                    "effective_stage": downloader.CONFIRMATION_STAGE,
                    "execution_status": "runnable",
                    "lock_payload_sha256": SHA_A,
                    "lock_path": str(lock_path),
                }
            ],
            "stages": {
                downloader.CONFIRMATION_STAGE: {
                    "complete": False,
                    "runs_complete": True,
                    "decision_status": "runtime_gate_pending",
                    "branch_status": "runnable",
                    "expected_new_runs": 6,
                    "completed_new_runs": 6,
                    "needs_boundary_extension": False,
                }
            },
            "hypothesis_families": [],
            "confirmation": {
                "comparison": "matched",
                "acceptance": {},
                "groups": [
                    {
                        "recipe_group_id": selected,
                        "recipe_family_sha256": SHA_B,
                        "complete": True,
                        "accepted": True,
                    }
                ],
                "practical_shortlist_recipe_group_ids": [selected],
                "selected_recipe_group_id": None,
                "selection_before_runtime_gate": selected,
                "runtime_gate": {
                    "required": True,
                    "status": "pending",
                    "selected_recipe_group_id": selected,
                    "checked_recipe_family_sha256": None,
                },
                "decision_status": "runtime_gate_pending",
            },
            "runs": [],
        }
        summary["summary_payload_sha256"] = adaptive.canonical_sha256(summary)
        return plan, lock, summary

    def test_summary_validation_requires_exact_pre_runtime_selection(self) -> None:
        with resolved_temporary_directory() as directory:
            lock_path = directory / "confirmation.lock.json"
            lock_path.write_text("{}\n", encoding="utf-8")
            plan, lock, summary = self._summary(lock_path)
            self.assertEqual(
                downloader.validate_confirmation_summary(
                    summary,
                    summary_path=directory / "summary.json",
                    lock=lock,
                    lock_path=lock_path,
                    plan=plan,
                ),
                "recipe_123",
            )
            summary["confirmation"]["selection_before_runtime_gate"] = "forged"
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "payload SHA differs"
            ):
                downloader.validate_confirmation_summary(
                    summary,
                    summary_path=directory / "summary.json",
                    lock=lock,
                    lock_path=lock_path,
                    plan=plan,
                )

    def _origin_and_entry(self) -> tuple[dict, dict]:
        notes = adaptive.canonical_json_dumps({"stage": "loss"})
        origin = {
            "experiment": "selected_experiment",
            "kernel_slug": "selected-kernel",
            "recipe_sha256": SHA_A,
            "code_bundle_sha256": SHA_B,
            "loss_variant": "bce",
            "loss_hook_sha256": SHA_C,
            "resolved_config": {"seed": 42},
            "completion_notes": notes,
            "completion_notes_sha256": adaptive.text_sha256(notes),
        }
        entry = {
            "experiment": origin["experiment"],
            "kernel_slug": origin["kernel_slug"],
            "recipe_sha256": origin["recipe_sha256"],
            "source_sha256": origin["code_bundle_sha256"],
            "loss_variant": origin["loss_variant"],
            "loss_hook_sha256": origin["loss_hook_sha256"],
            "expected_config": origin["resolved_config"],
            "expected_notes": notes,
            "role": "candidate",
        }
        return origin, entry

    def test_source_dispatch_is_unique_and_lock_bound(self) -> None:
        origin, entry = self._origin_and_entry()
        source_lock = {
            "schema_version": 2,
            "execution_status": "runnable",
            "lock_payload_sha256": SHA_D,
            "resolved_stage": {"variants": [{"experiment": origin["experiment"]}]},
        }
        with resolved_temporary_directory() as directory:
            source_path = directory / "source.lock.json"
            source_path.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                downloader.launcher, "campaign_variants", side_effect=[[], [entry]]
            ), mock.patch.object(
                downloader,
                "_load_lock_closure",
                return_value=[(source_lock, source_path)],
            ):
                selected_entry, kind, path, lock_sha = downloader._source_dispatch(
                    plan={},
                    base_config={},
                    confirmation_lock={},
                    confirmation_lock_path=source_path,
                    origin=origin,
                )
        self.assertEqual(selected_entry, entry)
        self.assertEqual(kind, "campaign_lock_variant")
        self.assertEqual(path, source_path)
        self.assertEqual(lock_sha, SHA_D)

    def test_summary_selection_is_recomputed_from_bound_artifacts(self) -> None:
        projection = {
            "comparison": "matched",
            "acceptance": {"minimum": 0.002},
            "groups": [{"recipe_group_id": "recipe_123"}],
            "practical_shortlist_recipe_group_ids": ["recipe_123"],
            "selection_before_runtime_gate": "recipe_123",
        }
        rows = [{"experiment": "selected", "completed": True}]
        summary = {"confirmation": dict(projection), "runs": rows}
        with resolved_temporary_directory() as directory:
            lock_path = directory / "lock.json"
            manifest_path = directory / "trusted.json"
            artifacts_dir = directory / "artifacts"
            lock_path.write_text("{}", encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")
            artifacts_dir.mkdir()
            with mock.patch.object(
                downloader.adaptive,
                "trusted_provenance_manifest_path",
                return_value=manifest_path,
            ), mock.patch.object(
                downloader.adaptive,
                "load_trusted_provenance",
                return_value={"artifacts_dir": str(artifacts_dir)},
            ), mock.patch.object(
                downloader.summarizer, "adaptive_run_rows", return_value=rows
            ), mock.patch.object(
                downloader.summarizer,
                "adaptive_confirmation_projection",
                return_value=projection,
            ):
                downloader._verify_summary_selection_from_artifacts(
                    plan={}, lock={}, lock_path=lock_path, summary=summary
                )
                summary["confirmation"]["selection_before_runtime_gate"] = "forged"
                with self.assertRaisesRegex(
                    downloader.SelectedCheckpointError,
                    "differs from bound run artifacts",
                ):
                    downloader._verify_summary_selection_from_artifacts(
                        plan={}, lock={}, lock_path=lock_path, summary=summary
                    )

    def _selected(
        self,
        destination: Path,
        *,
        materialization_root: Path | None = None,
    ) -> downloader.SelectedCheckpoint:
        if materialization_root is None:
            materialization_root = (
                destination.parent / "selected_checkpoints" / destination.name
            )
        return downloader.SelectedCheckpoint(
            campaign="minilm_5ep_sft_hparam_search_v1",
            confirmation_lock_path=str(destination.parent / "confirmation.lock.json"),
            confirmation_lock_payload_sha256=SHA_A,
            confirmation_summary_path=str(destination.parent / "summary.json"),
            confirmation_summary_payload_sha256=SHA_B,
            selected_recipe_group_id="recipe_123",
            recipe_family_sha256=SHA_C,
            origin_id="origin_123",
            experiment="selected_experiment",
            run_id="run-123",
            kernel_slug=destination.name,
            kaggle_kernel_ref=(
                f"{downloader.CAMPAIGN_KAGGLE_OWNER}/{destination.name}"
            ),
            seed=42,
            recipe_sha256=SHA_D,
            loss_variant="bce",
            loss_hook_sha256=SHA_E,
            code_bundle_sha256=SHA_B,
            source_authority_kind="campaign_lock_variant",
            source_lock_path=str(destination.parent / "source.lock.json"),
            source_lock_payload_sha256=SHA_D,
            bound_output_directory=str(destination),
            materialization_root=str(materialization_root),
        )

    def _write_completion(
        self,
        destination: Path,
        selected: downloader.SelectedCheckpoint,
        *,
        recorded_ref: str | None = None,
    ) -> dict:
        completion = {
            "status": "complete",
            "run_id": selected.run_id,
            "kaggle_kernel_ref": (
                selected.kaggle_kernel_ref if recorded_ref is None else recorded_ref
            ),
        }
        (destination / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        return {
            "completion_sha256": downloader.secure_file_sha256(
                destination / "notebook_completed.json", label="test completion"
            ),
            "completion_artifact_path": str(destination / "notebook_completed.json"),
            "kernel_slug": selected.kernel_slug,
        }

    def _fake_download_validation(
        self,
        directory: Path,
        selected: downloader.SelectedCheckpoint,
    ) -> dict:
        snapshot = downloader.snapshot_regular_tree(
            directory, label="fake validated output"
        )
        hashes = downloader.hash_captured_regular_files(
            directory, snapshot, label="fake validated output"
        )
        return {
            "status": "validated",
            "run_id": selected.run_id,
            "experiment": selected.experiment,
            "model_dir": str(directory / "model"),
            "required_files": {
                name: hashes[f"model/{name}"]
                for name in downloader.REQUIRED_MODEL_FILES
            },
            "iid_replay": {
                "pairs": downloader.IID_EXPECTED_ROWS,
                "orientations": downloader.IID_EXPECTED_ROWS * 2,
                "max_length": downloader.INFERENCE_MAX_LENGTH,
                "device": "cpu_fp32",
                "saved_score_backend": "t4_fp16_autocast",
                "absolute_tolerance": downloader.SCORE_ABSOLUTE_TOLERANCE,
                "max_absolute_difference": 0.002,
                "fallback_pairs": 0,
            },
            "output_tree": {
                "sha256": downloader.output_tree_sha256(hashes),
                "files": hashes,
            },
        }

    def test_fixed_owner_accepts_legacy_empty_or_exact_ref_only(self) -> None:
        with resolved_temporary_directory() as directory:
            destination = directory / "selected-kernel"
            destination.mkdir()
            selected = self._selected(destination)
            origin = self._write_completion(
                destination, selected, recorded_ref=""
            )
            with mock.patch.object(
                downloader,
                "secure_read_bytes_and_sha256",
                wraps=downloader.secure_read_bytes_and_sha256,
            ) as single_read:
                self.assertEqual(
                    downloader._lock_bound_kernel_ref(
                        origin=origin,
                        completion_path=destination / "notebook_completed.json",
                    ),
                    f"{downloader.CAMPAIGN_KAGGLE_OWNER}/selected-kernel",
                )
            single_read.assert_called_once()
            payload = json.loads(
                (destination / "notebook_completed.json").read_text(encoding="utf-8")
            )
            payload["kaggle_kernel_ref"] = selected.kaggle_kernel_ref
            (destination / "notebook_completed.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            origin["completion_sha256"] = downloader.secure_file_sha256(
                destination / "notebook_completed.json", label="exact completion"
            )
            self.assertEqual(
                downloader._lock_bound_kernel_ref(
                    origin=origin,
                    completion_path=destination / "notebook_completed.json",
                ),
                selected.kaggle_kernel_ref,
            )
            payload["kaggle_kernel_ref"] = "attacker/selected-kernel"
            (destination / "notebook_completed.json").write_text(
                json.dumps(payload), encoding="utf-8"
            )
            origin["completion_sha256"] = downloader.secure_file_sha256(
                destination / "notebook_completed.json",
                label="contradictory completion",
            )
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "contradicts the fixed"
            ):
                downloader._lock_bound_kernel_ref(
                    origin=origin,
                    completion_path=destination / "notebook_completed.json",
                )
        self.assertNotIn(
            "username", inspect.signature(downloader.download_selected_checkpoint).parameters
        )
        self.assertFalse(hasattr(downloader.parse_args([]), "username"))

    def test_default_contract_is_plan_only_and_does_not_claim_remote_pinning(self) -> None:
        with resolved_temporary_directory() as directory:
            selected = self._selected(directory / "selected-kernel")
            payload = downloader.plan_payload(selected)
        self.assertEqual(payload["status"], "plan_only")
        self.assertEqual(payload["network_actions"], 0)
        self.assertEqual(
            payload["download_contract"]["fixed_kaggle_owner"],
            downloader.CAMPAIGN_KAGGLE_OWNER,
        )
        self.assertFalse(
            payload["download_contract"]["remote_kernel_metadata_prepinned"]
        )

    def test_manifest_model_path_must_be_one_normalized_relative_directory(self) -> None:
        self.assertEqual(
            downloader._one_level_relative_directory(
                "model", label="test model path"
            ),
            Path("model"),
        )
        for unsafe in (".", "..", "nested/model", "/tmp/model", " model "):
            with self.subTest(unsafe=unsafe), self.assertRaisesRegex(
                downloader.SelectedCheckpointError,
                "one normalized relative directory name",
            ):
                downloader._one_level_relative_directory(
                    unsafe, label="test model path"
                )

    def test_download_materializes_content_address_without_touching_slim_output(self) -> None:
        with resolved_temporary_directory() as directory:
            bound_output = directory / "selected-kernel"
            bound_output.mkdir()
            selected_root = directory / "selected_checkpoints"
            selected = self._selected(
                bound_output,
                materialization_root=selected_root / bound_output.name,
            )
            origin = self._write_completion(
                bound_output, selected, recorded_ref=""
            )
            (bound_output / "old.txt").write_text("old", encoding="utf-8")
            calls: list[list[str]] = []

            def runner(command: list[str]) -> SimpleNamespace:
                calls.append(command)
                if command[1:3] == ["kernels", "status"]:
                    self.assertEqual(command[3], selected.kaggle_kernel_ref)
                    return SimpleNamespace(
                        returncode=0, stdout="Kernel status: complete\n"
                    )
                self.assertEqual(command[1:3], ["kernels", "output"])
                self.assertEqual(command[3], selected.kaggle_kernel_ref)
                staging = Path(command[command.index("-p") + 1])
                self._write_model_tree(staging / "model")
                return SimpleNamespace(returncode=0, stdout="downloaded")

            def validate(directory: Path, **_: object) -> dict:
                return self._fake_download_validation(directory, selected)

            with mock.patch.object(
                downloader, "SELECTED_CHECKPOINTS_ROOT", selected_root
            ), mock.patch.object(
                downloader, "validate_downloaded_checkpoint", side_effect=validate
            ):
                result = downloader.download_selected_checkpoint(
                    selected=selected,
                    origin=origin,
                    entry={},
                    cli=["kaggle"],
                    command_runner=runner,
                )
                no_remote = mock.Mock(
                    side_effect=AssertionError("existing materialization used network")
                )
                second = downloader.download_selected_checkpoint(
                    selected=selected,
                    origin=origin,
                    entry={},
                    cli=["kaggle"],
                    command_runner=no_remote,
                )
            self.assertEqual(
                calls,
                [
                    ["kaggle", "kernels", "status", selected.kaggle_kernel_ref],
                    [
                        "kaggle",
                        "kernels",
                        "output",
                        selected.kaggle_kernel_ref,
                        "-p",
                        calls[1][5],
                        "--force",
                        "--page-size",
                        "200",
                    ],
                ],
            )
            destination = Path(result["destination"])
            self.assertTrue((destination / "model" / "model.safetensors").is_file())
            self.assertEqual(
                (bound_output / "old.txt").read_text(encoding="utf-8"), "old"
            )
            self.assertEqual(result["kernel_ref"], selected.kaggle_kernel_ref)
            self.assertEqual(result["model_dir"], str(destination / "model"))
            self.assertEqual(
                second["status"], "already_materialized_and_revalidated"
            )
            self.assertEqual(second["remote_operations"], [])
            self.assertEqual(stat.S_IMODE(os.lstat(destination).st_mode), 0o555)
            manifest = Path(result["manifest"])
            self.assertEqual(stat.S_IMODE(os.lstat(manifest).st_mode), 0o444)
            self.assertEqual(
                stat.S_IMODE(os.lstat(manifest.parent).st_mode), 0o555
            )
            self.assertEqual(
                set(
                    downloader._load_canonical_manifest(manifest)[
                        "required_model_files"
                    ]
                ),
                set(downloader.REQUIRED_MODEL_FILES),
            )
            recorded = downloader._load_canonical_manifest(manifest)
            self.assertEqual(recorded["selected_recipe_group_id"], "recipe_123")
            self.assertEqual(recorded["origin"]["run_id"], selected.run_id)
            self.assertEqual(
                recorded["confirmation_lock"]["payload_sha256"], SHA_A
            )
            self.assertFalse(
                recorded["kaggle_authority"]["remote_kernel_metadata_prepinned"]
            )
            for name, metadata in recorded["required_model_files"].items():
                self.assertEqual(
                    downloader.secure_file_sha256(
                        destination / "model" / name,
                        label=f"installed {name}",
                    ),
                    metadata["sha256"],
                )
            os.chmod(manifest, 0o644)
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "mutable/hard-linked"
            ):
                downloader._write_or_validate_manifest(manifest, recorded)
            os.chmod(manifest, 0o444)
            self.assertFalse(
                any(word in {"push", "submit", "version"} for call in calls for word in call)
            )

    def test_download_copy_failure_leaves_no_final_tree_and_retry_commits(self) -> None:
        with resolved_temporary_directory() as directory:
            bound_output = directory / "selected-kernel"
            bound_output.mkdir()
            selected_root = directory / "selected_checkpoints"
            selected = self._selected(
                bound_output,
                materialization_root=selected_root / bound_output.name,
            )
            origin = self._write_completion(
                bound_output, selected, recorded_ref=""
            )

            def runner(command: list[str]) -> SimpleNamespace:
                if command[1:3] == ["kernels", "status"]:
                    return SimpleNamespace(
                        returncode=0, stdout="Kernel status: complete\n"
                    )
                staging = Path(command[command.index("-p") + 1])
                self._write_model_tree(staging / "model")
                return SimpleNamespace(returncode=0, stdout="downloaded")

            real_copy = downloader._copy_validated_tree_once
            injected = False

            def copy_then_fail_once(**kwargs: object) -> None:
                nonlocal injected
                real_copy(**kwargs)
                if not injected:
                    injected = True
                    raise OSError("injected post-copy failure")

            with mock.patch.object(
                downloader, "SELECTED_CHECKPOINTS_ROOT", selected_root
            ), mock.patch.object(
                downloader,
                "validate_downloaded_checkpoint",
                side_effect=lambda path, **_: self._fake_download_validation(
                    path, selected
                ),
            ):
                with mock.patch.object(
                    downloader,
                    "_copy_validated_tree_once",
                    side_effect=copy_then_fail_once,
                ), self.assertRaisesRegex(OSError, "injected post-copy"):
                    downloader.download_selected_checkpoint(
                        selected=selected,
                        origin=origin,
                        entry={},
                        cli=["kaggle"],
                        command_runner=runner,
                    )

                materialization_root = Path(selected.materialization_root)
                self.assertTrue(materialization_root.is_dir())
                self.assertEqual(list(materialization_root.iterdir()), [])
                self.assertFalse(
                    any(".install-" in path.name for path in directory.iterdir())
                )
                result = downloader.download_selected_checkpoint(
                    selected=selected,
                    origin=origin,
                    entry={},
                    cli=["kaggle"],
                    command_runner=runner,
                )
            self.assertEqual(result["status"], "downloaded_validated_and_materialized")
            self.assertTrue(Path(result["manifest"]).is_file())

    def test_noncomplete_remote_never_downloads(self) -> None:
        with resolved_temporary_directory() as directory:
            destination = directory / "selected-kernel"
            destination.mkdir()
            selected_root = directory / "selected_checkpoints"
            selected = self._selected(
                destination,
                materialization_root=selected_root / destination.name,
            )
            origin = self._write_completion(destination, selected, recorded_ref="")
            runner = mock.Mock(
                return_value=SimpleNamespace(
                    returncode=0, stdout="Kernel status: running\n"
                )
            )
            with mock.patch.object(
                downloader, "SELECTED_CHECKPOINTS_ROOT", selected_root
            ), self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "not terminal COMPLETE"
            ):
                downloader.download_selected_checkpoint(
                    selected=selected,
                    origin=origin,
                    entry={},
                    cli=["kaggle"],
                    command_runner=runner,
                )
            runner.assert_called_once_with(
                ["kaggle", "kernels", "status", selected.kaggle_kernel_ref]
            )

    def test_intermediate_output_parent_symlink_is_rejected_before_remote_call(self) -> None:
        with resolved_temporary_directory() as directory:
            real_parent = directory / "real-parent"
            real_parent.mkdir()
            destination = real_parent / "selected-kernel"
            destination.mkdir()
            linked_parent = directory / "linked-parent"
            linked_parent.symlink_to(real_parent, target_is_directory=True)
            selected = self._selected(linked_parent / destination.name)
            runner = mock.Mock()
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "symlink component"
            ):
                downloader.download_selected_checkpoint(
                    selected=selected,
                    origin={},
                    entry={},
                    cli=["kaggle"],
                    command_runner=runner,
                )
            runner.assert_not_called()

    def test_intermediate_materialization_symlink_is_rejected_before_remote_call(self) -> None:
        with resolved_temporary_directory() as directory:
            bound_output = directory / "selected-kernel"
            bound_output.mkdir()
            real_store = directory / "real-store"
            real_store.mkdir()
            linked_store = directory / "linked-store"
            linked_store.symlink_to(real_store, target_is_directory=True)
            selected = self._selected(
                bound_output,
                materialization_root=linked_store / bound_output.name,
            )
            origin = self._write_completion(
                bound_output, selected, recorded_ref=""
            )
            runner = mock.Mock()
            with mock.patch.object(
                downloader, "SELECTED_CHECKPOINTS_ROOT", linked_store
            ), self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "symlink component"
            ):
                downloader.download_selected_checkpoint(
                    selected=selected,
                    origin=origin,
                    entry={},
                    cli=["kaggle"],
                    command_runner=runner,
                )
            runner.assert_not_called()

    def test_content_address_copy_never_replaces_existing_directory(self) -> None:
        with resolved_temporary_directory() as directory:
            parent = directory / "content"
            staging = directory / "staging"
            parent.mkdir()
            staging.mkdir()
            (staging / "new.txt").write_text("new", encoding="utf-8")
            snapshot = downloader.snapshot_regular_tree(staging, label="staging")
            hashes = downloader.hash_captured_regular_files(
                staging, snapshot, label="staging"
            )
            destination = parent / downloader.output_tree_sha256(hashes)
            destination.mkdir()
            (destination / "old.txt").write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "refusing replacement"
            ):
                downloader._copy_validated_tree_once(
                    source=staging,
                    destination=destination,
                    source_snapshot=snapshot,
                    expected_hashes=hashes,
                )
            self.assertEqual((destination / "old.txt").read_text(encoding="utf-8"), "old")
            self.assertEqual((staging / "new.txt").read_text(encoding="utf-8"), "new")

    def test_private_content_copy_failure_is_cleaned_and_retry_succeeds(self) -> None:
        with resolved_temporary_directory() as directory:
            parent = directory / "content"
            staging = directory / "staging"
            parent.mkdir()
            staging.mkdir()
            (staging / "payload.bin").write_bytes(b"complete validated bytes")
            snapshot = downloader.snapshot_regular_tree(staging, label="staging")
            hashes = downloader.hash_captured_regular_files(
                staging, snapshot, label="staging"
            )
            destination = parent / downloader.output_tree_sha256(hashes)
            real_write = os.write
            injected = False

            def fail_after_partial_write(descriptor: int, payload: bytes) -> int:
                nonlocal injected
                if not injected:
                    injected = True
                    real_write(descriptor, payload[: max(1, len(payload) // 2)])
                    raise OSError("injected private-tree copy failure")
                return real_write(descriptor, payload)

            with mock.patch.object(
                downloader.os, "write", side_effect=fail_after_partial_write
            ), self.assertRaisesRegex(OSError, "injected private-tree"):
                downloader._copy_validated_tree_once(
                    source=staging,
                    destination=destination,
                    source_snapshot=snapshot,
                    expected_hashes=hashes,
                )
            self.assertFalse(destination.exists())

            downloader._copy_validated_tree_once(
                source=staging,
                destination=destination,
                source_snapshot=snapshot,
                expected_hashes=hashes,
            )
            self.assertEqual(
                (destination / "payload.bin").read_bytes(),
                b"complete validated bytes",
            )

    def test_atomic_content_publish_never_replaces_existing_tree(self) -> None:
        with resolved_temporary_directory() as directory:
            source_parent = directory / "source"
            destination_parent = directory / "destination"
            source_parent.mkdir()
            destination_parent.mkdir()
            (source_parent / "tree").mkdir()
            (destination_parent / "tree").mkdir()
            (source_parent / "tree" / "new").write_text("new", encoding="utf-8")
            (destination_parent / "tree" / "old").write_text(
                "old", encoding="utf-8"
            )
            source_descriptor = downloader._open_directory_fd(
                source_parent, label="source parent"
            )
            destination_descriptor = downloader._open_directory_fd(
                destination_parent, label="destination parent"
            )
            try:
                self.assertFalse(
                    downloader._atomic_rename_no_replace_at(
                        source_parent_descriptor=source_descriptor,
                        source_name="tree",
                        destination_parent_descriptor=destination_descriptor,
                        destination_name="tree",
                    )
                )
            finally:
                os.close(source_descriptor)
                os.close(destination_descriptor)
            self.assertEqual(
                (source_parent / "tree" / "new").read_text(encoding="utf-8"),
                "new",
            )
            self.assertEqual(
                (destination_parent / "tree" / "old").read_text(
                    encoding="utf-8"
                ),
                "old",
            )

    def test_manifest_path_swap_after_fd_read_is_rejected(self) -> None:
        with resolved_temporary_directory() as directory:
            root = directory / "materialization"
            root.mkdir()
            unhashed = {"output_tree": {"sha256": SHA_A}}
            payload = {
                **unhashed,
                "manifest_payload_sha256": adaptive.canonical_sha256(unhashed),
            }
            manifest = root / f"{SHA_A}{downloader.MANIFEST_SUFFIX}"
            downloader._write_or_validate_manifest(manifest, payload)
            attacker = directory / "attacker.json"
            attacker.write_bytes(b"different but mode-correct bytes\n")
            os.chmod(attacker, 0o444)
            real_fd_read = downloader._read_regular_fd
            swapped = False

            def read_then_swap(
                descriptor: int, *, label: str
            ) -> tuple[bytes, tuple[int, ...]]:
                nonlocal swapped
                observed = real_fd_read(descriptor, label=label)
                if label == "selected-checkpoint manifest" and not swapped:
                    swapped = True
                    os.chmod(root, 0o755)
                    os.replace(attacker, manifest)
                    os.chmod(root, 0o555)
                return observed

            try:
                with mock.patch.object(
                    downloader, "_read_regular_fd", side_effect=read_then_swap
                ), self.assertRaisesRegex(
                    downloader.SelectedCheckpointError, "changed after its first fd read"
                ):
                    downloader._write_or_validate_manifest(manifest, payload)
            finally:
                os.chmod(root, 0o755)
            self.assertTrue(swapped)
            self.assertEqual(
                manifest.read_bytes(), b"different but mode-correct bytes\n"
            )

    def test_manifest_same_inode_overwrite_after_fd_read_is_rejected(self) -> None:
        with resolved_temporary_directory() as directory:
            root = directory / "materialization"
            root.mkdir()
            unhashed = {"output_tree": {"sha256": SHA_A}}
            payload = {
                **unhashed,
                "manifest_payload_sha256": adaptive.canonical_sha256(unhashed),
            }
            manifest = root / f"{SHA_A}{downloader.MANIFEST_SUFFIX}"
            downloader._write_or_validate_manifest(manifest, payload)
            expected = downloader._canonical_manifest_bytes(payload)
            altered = b"[" + expected[1:]
            self.assertEqual(len(altered), len(expected))
            original_inode = os.lstat(manifest).st_ino
            real_fd_read = downloader._read_regular_fd
            overwritten = False

            def read_then_overwrite_same_inode(
                descriptor: int, *, label: str
            ) -> tuple[bytes, tuple[int, ...]]:
                nonlocal overwritten
                observed = real_fd_read(descriptor, label=label)
                if label == "selected-checkpoint manifest" and not overwritten:
                    overwritten = True
                    os.chmod(manifest, 0o644)
                    attack_descriptor = os.open(manifest, os.O_WRONLY)
                    try:
                        os.lseek(attack_descriptor, 0, os.SEEK_SET)
                        offset = 0
                        while offset < len(altered):
                            offset += os.write(
                                attack_descriptor, altered[offset:]
                            )
                        os.ftruncate(attack_descriptor, len(altered))
                        os.fsync(attack_descriptor)
                    finally:
                        os.close(attack_descriptor)
                    os.chmod(manifest, 0o444)
                return observed

            try:
                with mock.patch.object(
                    downloader,
                    "_read_regular_fd",
                    side_effect=read_then_overwrite_same_inode,
                ), self.assertRaisesRegex(
                    downloader.SelectedCheckpointError,
                    "changed after its first fd read",
                ):
                    downloader._write_or_validate_manifest(manifest, payload)
            finally:
                os.chmod(root, 0o755)
            self.assertTrue(overwritten)
            self.assertEqual(os.lstat(manifest).st_ino, original_inode)
            self.assertEqual(manifest.read_bytes(), altered)

    def test_partial_manifest_write_is_cleaned_and_retry_succeeds(self) -> None:
        with resolved_temporary_directory() as directory:
            root = directory / "materialization"
            root.mkdir()
            unhashed = {"output_tree": {"sha256": SHA_A}}
            payload = {
                **unhashed,
                "manifest_payload_sha256": adaptive.canonical_sha256(unhashed),
            }
            manifest = root / f"{SHA_A}{downloader.MANIFEST_SUFFIX}"
            real_write = os.write
            injected = False

            def fail_after_partial_write(descriptor: int, data: bytes) -> int:
                nonlocal injected
                if not injected:
                    injected = True
                    real_write(descriptor, data[: max(1, len(data) // 2)])
                    raise OSError("injected manifest write failure")
                return real_write(descriptor, data)

            with mock.patch.object(
                downloader.os, "write", side_effect=fail_after_partial_write
            ), self.assertRaisesRegex(OSError, "injected manifest"):
                downloader._write_or_validate_manifest(manifest, payload)
            self.assertFalse(manifest.exists())
            self.assertEqual(list(root.iterdir()), [])

            downloader._write_or_validate_manifest(manifest, payload)
            self.assertEqual(stat.S_IMODE(os.lstat(root).st_mode), 0o555)
            self.assertEqual(stat.S_IMODE(os.lstat(manifest).st_mode), 0o444)
            os.chmod(root, 0o755)

    def test_hardlinks_are_rejected_and_content_is_rehashed(self) -> None:
        with resolved_temporary_directory() as directory:
            original = directory / "original.bin"
            linked = directory / "linked.bin"
            original.write_bytes(b"first")
            os.link(original, linked)
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "hard-linked file"
            ):
                downloader.snapshot_regular_tree(directory, label="hardlink tree")

        with resolved_temporary_directory() as directory:
            payload = directory / "payload.bin"
            payload.write_bytes(b"first")
            first_snapshot = downloader.snapshot_regular_tree(
                directory, label="first tree"
            )
            first_hashes = downloader.hash_captured_regular_files(
                directory, first_snapshot, label="first tree"
            )
            self.assertEqual(len(first_snapshot["payload.bin"]), 7)
            payload.write_bytes(b"other")
            second_snapshot = downloader.snapshot_regular_tree(
                directory, label="second tree"
            )
            second_hashes = downloader.hash_captured_regular_files(
                directory, second_snapshot, label="second tree"
            )
            self.assertNotEqual(first_hashes, second_hashes)

    def test_bound_iid_is_hash_checked_from_the_same_bytes_that_are_parsed(self) -> None:
        with resolved_temporary_directory() as directory:
            artifact = directory / "bound.parquet"
            artifact.write_bytes(b"not a parquet")
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "SHA differs from its authority"
            ):
                downloader._read_parquet_secure(
                    artifact,
                    label="bound IID predictions",
                    expected_sha256=SHA_A,
                )

    def _write_model_tree(self, model: Path) -> None:
        import numpy as np
        from safetensors.numpy import save_file

        model.mkdir()
        save_file({"weight": np.zeros((1,), dtype=np.float32)}, model / "model.safetensors")
        payloads = {
            "config.json": {
                "model_type": "xlm-roberta",
                "architectures": ["XLMRobertaForSequenceClassification"],
            },
            "tokenizer.json": {"version": "1.0"},
            "tokenizer_config.json": {"model_max_length": 512},
            "special_tokens_map.json": {"unk_token": "<unk>"},
            "training_config.json": {
                "max_length": downloader.INFERENCE_MAX_LENGTH,
                "symmetric_validation": True,
            },
            "training_report.json": {"status": "complete"},
        }
        for name, payload in payloads.items():
            (model / name).write_text(json.dumps(payload), encoding="utf-8")
        for name in (
            "iid_validation_predictions.parquet",
            "hard_validation_predictions.parquet",
            "ood_validation_predictions.parquet",
        ):
            (model / name).write_bytes(name.encode("utf-8"))

    def _download_fixture(
        self, directory: Path
    ) -> tuple[Path, downloader.SelectedCheckpoint, dict]:
        bound = directory / "selected-kernel"
        downloaded = directory / "downloaded"
        bound.mkdir()
        downloaded.mkdir()
        selected = self._selected(bound)
        completion_payload = {
            "status": "complete",
            "run_id": selected.run_id,
            "kaggle_kernel_ref": selected.kaggle_kernel_ref,
        }
        for root in (bound, downloaded):
            (root / "notebook_completed.json").write_text(
                json.dumps(completion_payload), encoding="utf-8"
            )
        for name in downloader.REQUIRED_ROOT_FILES - {"notebook_completed.json"}:
            payload = "{}" if name.endswith(".json") else "value\n"
            (downloaded / name).write_text(payload, encoding="utf-8")
        self._write_model_tree(downloaded / "model")
        (bound / "model").mkdir()
        for name in ("training_config.json", "iid_validation_predictions.parquet"):
            (bound / "model" / name).write_bytes(
                (downloaded / "model" / name).read_bytes()
            )
        origin = {
            "kernel_slug": selected.kernel_slug,
            "completion_artifact_path": str(bound / "notebook_completed.json"),
            "completion_sha256": downloader.secure_file_sha256(
                bound / "notebook_completed.json", label="bound completion"
            ),
            "training_config_artifact_path": str(
                bound / "model" / "training_config.json"
            ),
            "training_config_artifact_sha256": downloader.secure_file_sha256(
                bound / "model" / "training_config.json", label="bound config"
            ),
            "iid_predictions_artifact_path": str(
                bound / "model" / "iid_validation_predictions.parquet"
            ),
            "iid_predictions_relative_path": "model/iid_validation_predictions.parquet",
            "iid_predictions_sha256": downloader.secure_file_sha256(
                bound / "model" / "iid_validation_predictions.parquet",
                label="bound IID",
            ),
        }
        return downloaded, selected, origin

    def _validate_fixture(self, directory: Path) -> dict:
        downloaded, selected, origin = self._download_fixture(directory)
        with mock.patch.object(
            downloader.launcher,
            "validate_run_output",
            return_value={"run_id": selected.run_id},
        ), mock.patch.object(
            downloader,
            "verify_full_iid_replay",
            return_value={"pairs": downloader.IID_EXPECTED_ROWS, "fallback_pairs": 0},
        ):
            return downloader.validate_downloaded_checkpoint(
                downloaded,
                selected=selected,
                origin=origin,
                entry={},
            )

    def test_exact_model_tree_is_required(self) -> None:
        with resolved_temporary_directory() as directory:
            result = self._validate_fixture(directory)
            self.assertEqual(
                set(result["required_files"]), set(downloader.REQUIRED_MODEL_FILES)
            )
        for mutation in ("missing", "extra", "nested"):
            with self.subTest(mutation=mutation), resolved_temporary_directory() as directory:
                downloaded, selected, origin = self._download_fixture(directory)
                if mutation == "missing":
                    (downloaded / "model" / "tokenizer.json").unlink()
                elif mutation == "extra":
                    (downloaded / "model" / "unbound.bin").write_bytes(b"extra")
                else:
                    nested = downloaded / "model" / "nested"
                    nested.mkdir()
                    (nested / "payload").write_bytes(b"extra")
                with mock.patch.object(
                    downloader.launcher,
                    "validate_run_output",
                    return_value={"run_id": selected.run_id},
                ), self.assertRaisesRegex(
                    downloader.SelectedCheckpointError, "model output tree differs"
                ):
                    downloader.validate_downloaded_checkpoint(
                        downloaded,
                        selected=selected,
                        origin=origin,
                        entry={},
                    )

    def test_model_directory_and_file_symlinks_are_rejected(self) -> None:
        for target in ("model", "tokenizer"):
            with self.subTest(target=target), resolved_temporary_directory() as directory:
                downloaded, selected, origin = self._download_fixture(directory)
                if target == "model":
                    external = directory / "external-model"
                    (downloaded / "model").rename(external)
                    (downloaded / "model").symlink_to(
                        external, target_is_directory=True
                    )
                else:
                    tokenizer = downloaded / "model" / "tokenizer.json"
                    external = directory / "external-tokenizer.json"
                    tokenizer.rename(external)
                    tokenizer.symlink_to(external)
                with self.assertRaisesRegex(
                    downloader.SelectedCheckpointError, "contains a symlink"
                ):
                    downloader.validate_downloaded_checkpoint(
                        downloaded,
                        selected=selected,
                        origin=origin,
                        entry={},
                    )

    def test_safetensors_eof_offset_is_rejected(self) -> None:
        with resolved_temporary_directory() as directory:
            path = directory / "model.safetensors"
            header = json.dumps(
                {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}},
                separators=(",", ":"),
            ).encode("utf-8")
            header += b" " * ((8 - len(header) % 8) % 8)
            path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0\0\0")
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "safetensors validation"
            ):
                downloader.validate_safetensors_file(path)

    def test_iid_replay_refuses_any_fallback_field(self) -> None:
        import numpy as np
        import pandas as pd

        rows = downloader.IID_EXPECTED_ROWS
        frame = pd.DataFrame(
            {
                "pair_index": np.arange(rows, dtype=np.int64),
                "id1": np.arange(rows, dtype=np.int64),
                "id2": np.arange(rows, dtype=np.int64) + rows,
                "target": np.zeros(rows, dtype=np.int8),
                "product_text_1": ["alpha"] * rows,
                "product_text_2": ["beta"] * rows,
                "score": np.full(rows, 0.5, dtype=np.float32),
                "score_ab": np.full(rows, 0.5, dtype=np.float32),
                "score_ba": np.full(rows, 0.5, dtype=np.float32),
                "fallback_used": np.zeros(rows, dtype=bool),
            }
        )
        with mock.patch.object(
            downloader, "_read_parquet_secure", side_effect=[frame, frame.copy()]
        ), self.assertRaisesRegex(
            downloader.SelectedCheckpointError, "fallback fields"
        ):
            downloader.verify_full_iid_replay(
                model_dir=Path("/not/reached"),
                downloaded_iid_path=Path("/not/reached/downloaded.parquet"),
                bound_iid_path=Path("/not/reached/bound.parquet"),
                bound_iid_sha256=SHA_A,
            )

    def test_full_replay_tolerance_accepts_0029_and_rejects_0031(self) -> None:
        import math

        import numpy as np
        import pandas as pd
        import torch

        self.assertEqual(downloader.SCORE_ABSOLUTE_TOLERANCE, 0.003)
        rows = 2
        frame = pd.DataFrame(
            {
                "pair_index": np.arange(rows, dtype=np.int64),
                "id1": np.arange(rows, dtype=np.int64),
                "id2": np.arange(rows, dtype=np.int64) + rows,
                "target": np.zeros(rows, dtype=np.int8),
                "product_text_1": ["alpha"] * rows,
                "product_text_2": ["beta"] * rows,
                "score": np.full(rows, 0.5, dtype=np.float32),
                "score_ab": np.full(rows, 0.5, dtype=np.float32),
                "score_ba": np.full(rows, 0.5, dtype=np.float32),
            }
        )

        class FakeTokenizer:
            def __call__(self, first: list[str], second: list[str], **_: object) -> dict:
                self.last_pairs = list(zip(first, second))
                return {"input_ids": torch.zeros((len(first), 1), dtype=torch.long)}

        class FakeModel:
            def __init__(self, difference: float) -> None:
                self.logit = math.log((0.5 + difference) / (0.5 - difference))

            def __call__(self, *, input_ids: object) -> SimpleNamespace:
                count = int(input_ids.shape[0])
                return SimpleNamespace(
                    logits=torch.full((count, 1), self.logit, dtype=torch.float64)
                )

        def replay(difference: float) -> dict:
            with mock.patch.object(
                downloader, "IID_EXPECTED_ROWS", rows
            ), mock.patch.object(
                downloader,
                "_read_parquet_secure",
                side_effect=[frame.copy(), frame.copy()],
            ), mock.patch.object(
                downloader,
                "_load_offline_transformer",
                return_value=(torch, FakeTokenizer(), FakeModel(difference)),
            ):
                return downloader.verify_full_iid_replay(
                    model_dir=Path("/not/reached"),
                    downloaded_iid_path=Path("/not/reached/downloaded.parquet"),
                    bound_iid_path=Path("/not/reached/bound.parquet"),
                    bound_iid_sha256=SHA_A,
                    batch_size=2,
                )

        accepted = replay(0.0029)
        self.assertLessEqual(
            accepted["max_absolute_difference"],
            downloader.SCORE_ABSOLUTE_TOLERANCE,
        )
        with self.assertRaisesRegex(
            downloader.SelectedCheckpointError, "0.003"
        ):
            replay(0.0031)

    def test_full_iid_replay_rejects_altered_but_valid_model(self) -> None:
        import numpy as np
        import pandas as pd
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (
            PreTrainedTokenizerFast,
            XLMRobertaConfig,
            XLMRobertaForSequenceClassification,
        )

        with resolved_temporary_directory() as directory:
            model_dir = directory / "tiny-model"
            model_dir.mkdir()
            vocabulary = {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "<mask>": 4,
                "alpha": 5,
                "beta": 6,
            }
            backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=backend,
                bos_token="<s>",
                cls_token="<s>",
                eos_token="</s>",
                sep_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                mask_token="<mask>",
                model_max_length=512,
            )
            tokenizer.save_pretrained(model_dir)
            config = XLMRobertaConfig(
                vocab_size=len(vocabulary),
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                max_position_embeddings=514,
                type_vocab_size=1,
                num_labels=1,
                bos_token_id=0,
                pad_token_id=1,
                eos_token_id=2,
            )
            config.architectures = ["XLMRobertaForSequenceClassification"]
            model = XLMRobertaForSequenceClassification(config)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                # This remains a structurally valid, loadable checkpoint, but
                # is behaviorally different from the frozen 0.5 probabilities.
                model.classifier.out_proj.bias.fill_(2.0)
            model.save_pretrained(model_dir, safe_serialization=True)

            rows = downloader.IID_EXPECTED_ROWS
            frame = pd.DataFrame(
                {
                    "pair_index": np.arange(rows, dtype=np.int64),
                    "id1": np.arange(rows, dtype=np.int64),
                    "id2": np.arange(rows, dtype=np.int64) + rows,
                    "target": np.zeros(rows, dtype=np.int8),
                    "product_text_1": ["alpha"] * rows,
                    "product_text_2": ["beta"] * rows,
                    "score": np.full(rows, 0.5, dtype=np.float32),
                    "score_ab": np.full(rows, 0.5, dtype=np.float32),
                    "score_ba": np.full(rows, 0.5, dtype=np.float32),
                }
            )
            downloaded = directory / "downloaded.parquet"
            bound = directory / "bound.parquet"
            frame.to_parquet(downloaded, index=False)
            frame.to_parquet(bound, index=False)
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "IID replay differs"
            ):
                downloader.verify_full_iid_replay(
                    model_dir=model_dir,
                    downloaded_iid_path=downloaded,
                    bound_iid_path=bound,
                    bound_iid_sha256=downloader.secure_file_sha256(
                        bound, label="bound model-attack IID"
                    ),
                    batch_size=16,
                )

    def test_full_iid_replay_rejects_altered_but_valid_tokenizer(self) -> None:
        import numpy as np
        import pandas as pd
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (
            PreTrainedTokenizerFast,
            XLMRobertaConfig,
            XLMRobertaForSequenceClassification,
        )

        with resolved_temporary_directory() as directory:
            model_dir = directory / "tiny-model"
            model_dir.mkdir()
            vocabulary = {
                "<s>": 0,
                "<pad>": 1,
                "</s>": 2,
                "<unk>": 3,
                "<mask>": 4,
                "alpha": 5,
                "beta": 6,
                "gamma": 7,
            }

            def save_tokenizer(words: dict[str, int]) -> None:
                backend = Tokenizer(models.WordLevel(words, unk_token="<unk>"))
                backend.pre_tokenizer = pre_tokenizers.Whitespace()
                tokenizer = PreTrainedTokenizerFast(
                    tokenizer_object=backend,
                    bos_token="<s>",
                    cls_token="<s>",
                    eos_token="</s>",
                    sep_token="</s>",
                    unk_token="<unk>",
                    pad_token="<pad>",
                    mask_token="<mask>",
                    model_max_length=512,
                )
                tokenizer.save_pretrained(model_dir)

            torch.manual_seed(7)
            config = XLMRobertaConfig(
                vocab_size=len(vocabulary),
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                max_position_embeddings=514,
                type_vocab_size=1,
                num_labels=1,
                bos_token_id=0,
                pad_token_id=1,
                eos_token_id=2,
            )
            config.architectures = ["XLMRobertaForSequenceClassification"]
            model = XLMRobertaForSequenceClassification(config)
            with torch.no_grad():
                model.classifier.out_proj.weight.mul_(50.0)
            model.save_pretrained(model_dir, safe_serialization=True)
            save_tokenizer(vocabulary)

            torch_api, original_tokenizer, loaded_model = (
                downloader._load_offline_transformer(model_dir)
            )
            with torch_api.inference_mode():
                encoded = original_tokenizer(
                    ["alpha", "beta"],
                    ["beta", "alpha"],
                    add_special_tokens=True,
                    padding=True,
                    truncation="longest_first",
                    max_length=downloader.INFERENCE_MAX_LENGTH,
                    return_tensors="pt",
                )
                probabilities = (
                    loaded_model(**encoded).logits.reshape(-1).sigmoid().numpy()
                )
            del loaded_model
            score_ab = np.float32(probabilities[0])
            score_ba = np.float32(probabilities[1])
            score = np.float32((score_ab + score_ba) / 2.0)

            rows = downloader.IID_EXPECTED_ROWS
            frame = pd.DataFrame(
                {
                    "pair_index": np.arange(rows, dtype=np.int64),
                    "id1": np.arange(rows, dtype=np.int64),
                    "id2": np.arange(rows, dtype=np.int64) + rows,
                    "target": np.zeros(rows, dtype=np.int8),
                    "product_text_1": ["alpha"] * rows,
                    "product_text_2": ["beta"] * rows,
                    "score": np.full(rows, score, dtype=np.float32),
                    "score_ab": np.full(rows, score_ab, dtype=np.float32),
                    "score_ba": np.full(rows, score_ba, dtype=np.float32),
                }
            )
            downloaded = directory / "downloaded.parquet"
            bound = directory / "bound.parquet"
            frame.to_parquet(downloaded, index=False)
            frame.to_parquet(bound, index=False)

            altered = dict(vocabulary)
            altered["alpha"], altered["gamma"] = (
                altered["gamma"],
                altered["alpha"],
            )
            save_tokenizer(altered)
            with self.assertRaisesRegex(
                downloader.SelectedCheckpointError, "IID replay differs"
            ):
                downloader.verify_full_iid_replay(
                    model_dir=model_dir,
                    downloaded_iid_path=downloaded,
                    bound_iid_path=bound,
                    bound_iid_sha256=downloader.secure_file_sha256(
                        bound, label="bound tokenizer-attack IID"
                    ),
                    batch_size=16,
                )


if __name__ == "__main__":
    unittest.main()
