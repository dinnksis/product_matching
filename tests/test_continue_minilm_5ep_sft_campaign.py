from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import continue_minilm_5ep_sft_campaign as controller


class CampaignControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.temp = Path(self.temporary.name)
        self.report = self.temp / "reports" / "campaign"
        self.locks = self.report / "stage_locks"
        self.artifacts = self.temp / "artifacts"
        self.summary = self.report / "summary.json"
        self.state = self.report / "controller_state.json"
        self.baseline = self.report / "stages" / "lr_log_line" / "summary.json"
        self.locks.mkdir(parents=True)
        self.artifacts.mkdir(parents=True)
        self.plan = json.loads(controller.DEFAULT_PLAN.read_text(encoding="utf-8"))
        self.plan_path = self.temp / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.plan_sha = controller.canonical_sha256(self.plan)
        self.paths = controller.ControllerPaths(
            root=ROOT,
            plan=self.plan_path,
            report_dir=self.report,
            summary=self.summary,
            locks_dir=self.locks,
            artifacts_dir=self.artifacts,
            state_path=self.state,
            baseline_summary=self.baseline,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _variant(experiment: str, slug: str) -> dict:
        return {"experiment": experiment, "kernel_slug": slug}

    def _write_lock_payload(self, path: Path, payload: dict) -> dict:
        payload = dict(payload)
        payload["lock_payload_sha256"] = controller.canonical_sha256(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(controller.canonical_json_dumps(payload) + "\n", encoding="utf-8")
        return payload

    def _schema1_lock(
        self,
        *,
        effective_stage: str,
        source_stage: str,
        target_stage: str,
        variants: list[dict],
        coordinate: str | None = None,
        boundary: bool = False,
        path: Path | None = None,
    ) -> tuple[Path, dict]:
        if path is None:
            if boundary:
                path = self.locks / f"{effective_stage}_boundary.lock.json"
            elif coordinate:
                path = self.locks / f"regularization_coordinate_search_{coordinate}.lock.json"
            else:
                path = self.locks / f"{effective_stage}.lock.json"
        payload = {
            "schema_version": 1,
            "kind": "minilm_5ep_sft_stage_lock",
            "campaign": self.plan["campaign"],
            "source_stage": source_stage,
            "target_stage": target_stage,
            "coordinate": coordinate,
            "effective_stage": effective_stage,
            "source_plan_sha256": self.plan_sha,
            "resolved_stage": {"variants": variants},
        }
        if boundary:
            payload["transition_kind"] = "conditional_boundary_extension"
        return path, self._write_lock_payload(path, payload)

    def _adaptive_lock(
        self,
        *,
        mode: str,
        effective_stage: str,
        variants: list[dict],
        status: str,
        union: list[str],
        path: Path | None = None,
    ) -> tuple[Path, dict]:
        path = path or self.locks / f"{effective_stage}.lock.json"
        payload = {
            "schema_version": 2,
            "kind": (
                "minilm_5ep_sft_adaptive_branch_receipt"
                if status == "skipped"
                else "minilm_5ep_sft_adaptive_stage_lock"
            ),
            "campaign": self.plan["campaign"],
            "mode": mode,
            "effective_stage": effective_stage,
            "execution_status": status,
            "source_plan_sha256": self.plan_sha,
            "resolved_stage": {"variants": variants},
            "budget": {
                "hard_limit": 37,
                "all_unique_kernel_slugs_after": union,
                "resulting_unique_kernels": len(union),
            },
        }
        return path, self._write_lock_payload(path, payload)

    def _write_schema1_summary(
        self,
        stage: str,
        *,
        complete: bool = True,
        needs_boundary: bool = False,
        lock: tuple[Path, dict] | None = None,
        rows: list[dict] | None = None,
        decision_status: str = "ready",
    ) -> None:
        decision = {
            "complete": complete,
            "decision_status": decision_status,
            "control_gate": "passed",
            "needs_boundary_extension": needs_boundary,
            "recommended_experiment": f"selected-{stage}",
            "recommended_run_id": f"run-{stage}",
        }
        payload: dict = {
            "schema_version": 1,
            "campaign": self.plan["campaign"],
            "stages": {stage: decision},
            "runs": rows or [],
        }
        if lock is not None:
            payload["stage_lock"] = {
                "path": str(lock[0]),
                "lock_payload_sha256": lock[1]["lock_payload_sha256"],
                "effective_stage": stage,
            }
        self.summary.parent.mkdir(parents=True, exist_ok=True)
        self.summary.write_text(json.dumps(payload), encoding="utf-8")

    def _write_adaptive_summary(
        self,
        stage: str,
        *,
        closure: list[dict],
        union: list[str],
        complete: bool = True,
        runs_complete: bool = True,
        decision_status: str = "ready",
    ) -> None:
        runnable = sorted(
            item["lock_payload_sha256"]
            for item in closure
            if item["execution_status"] == "runnable"
        )
        receipts = sorted(
            item["lock_payload_sha256"]
            for item in closure
            if item["execution_status"] == "skipped"
        )
        payload = {
            "schema_version": 2,
            "campaign": self.plan["campaign"],
            "execution_status": "complete" if complete else "pending",
            "effective_stage": stage,
            "execution_lock_sha256s": runnable,
            "execution_receipt_sha256s": receipts,
            "execution_campaign_lock_sha256s": sorted(runnable + receipts),
            "budget": {
                "history_complete_through": stage,
                "unique_kernel_slugs": union,
                "unique_kernels": len(union),
                "hard_limit": 37,
            },
            "stages": {
                stage: {
                    "complete": complete,
                    "runs_complete": runs_complete,
                    "decision_status": decision_status,
                    "needs_boundary_extension": False,
                }
            },
            "runs": [],
        }
        self.summary.write_text(json.dumps(payload), encoding="utf-8")

    def _completion(self, variant: dict, *, sheets: str = "synced") -> None:
        directory = self.artifacts / variant["kernel_slug"]
        directory.mkdir(parents=True, exist_ok=True)
        run_id = f"run-{variant['experiment']}"
        (directory / "notebook_completed.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "experiment": variant["experiment"],
                    "experiment_group": "sft",
                    "run_id": run_id,
                }
            ),
            encoding="utf-8",
        )
        if sheets == "synced":
            (directory / "google_sheets_sync.json").write_text(
                json.dumps(
                    {
                        "status": "synced",
                        "run_id": run_id,
                        "experiment_group": "sft",
                        "comparison_sheet": "sft_exps",
                        "spreadsheet_id": controller.EXPERIMENT_SPREADSHEET_ID,
                    }
                ),
                encoding="utf-8",
            )
        elif sheets == "pending":
            (directory / "sheets_sync_pending.json").write_text(
                json.dumps({"status": "pending", "run_id": run_id}),
                encoding="utf-8",
            )
        elif sheets == "missing":
            pass
        elif sheets == "stale":
            (directory / "google_sheets_sync.json").write_text(
                json.dumps(
                    {
                        "status": "synced",
                        "run_id": f"stale-{run_id}",
                        "experiment_group": "sft",
                        "comparison_sheet": "sft_exps",
                        "spreadsheet_id": controller.EXPERIMENT_SPREADSHEET_ID,
                    }
                ),
                encoding="utf-8",
            )
        else:
            self.fail(f"Unknown sheets fixture state: {sheets}")

    def _epoch_fixture(self) -> tuple[tuple[Path, dict], list[dict]]:
        variants = [
            self._variant("epoch-two", "epoch-two-slug"),
            self._variant("epoch-three", "epoch-three-slug"),
        ]
        lock = self._schema1_lock(
            effective_stage="epoch_line",
            source_stage="lr_log_line",
            target_stage="epoch_line",
            variants=variants,
        )
        self._write_schema1_summary("lr_log_line")
        return lock, variants

    def _schema1_chain_through_max_grad(self) -> list[tuple[Path, dict]]:
        locks: list[tuple[Path, dict]] = []
        predecessor = "lr_log_line"
        epoch = self._schema1_lock(
            effective_stage="epoch_line",
            source_stage=predecessor,
            target_stage="epoch_line",
            variants=[self._variant("epoch-two", "epoch-two-slug")],
        )
        locks.append(epoch)
        predecessor = "epoch_line"
        for coordinate in (
            "effective_batch",
            "warmup_ratio",
            "weight_decay",
            "label_smoothing",
            "classifier_dropout",
            "max_grad_norm",
        ):
            effective = f"regularization_coordinate_search__{coordinate}"
            lock = self._schema1_lock(
                effective_stage=effective,
                source_stage=predecessor,
                target_stage="regularization_coordinate_search",
                coordinate=coordinate,
                variants=[
                    self._variant(
                        f"coordinate-{coordinate}",
                        f"coordinate-{coordinate}-slug",
                    )
                ],
            )
            locks.append(lock)
            predecessor = effective
        return locks

    def _base_slugs(self) -> list[str]:
        return [
            variant["kernel_slug"] for variant in self.plan["stages"][0]["variants"]
        ]

    def _success_runner_with_epoch_summary(self, epoch_lock: tuple[Path, dict]):
        def run(argv, **kwargs):
            if Path(argv[1]).name == controller.SUMMARIZER.name:
                self._write_schema1_summary("epoch_line", lock=epoch_lock)
            return subprocess.CompletedProcess(argv, 0)

        return run

    def test_default_is_plan_only_and_starts_no_subprocess(self) -> None:
        self._epoch_fixture()
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                stop_after="epoch_line",
            ).run()
        run.assert_not_called()
        self.assertEqual(result.status, "plan_only")
        self.assertEqual(result.state["planned_action"]["kind"], "execute_stage")
        self.assertTrue(self.state.is_file())

    def test_exact_order_is_generator_dry_submit_wait_then_summary(self) -> None:
        epoch_lock, _ = self._epoch_fixture()
        with mock.patch.object(
            controller.subprocess,
            "run",
            side_effect=self._success_runner_with_epoch_summary(epoch_lock),
        ) as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(
            [Path(command[1]).name for command in commands],
            [
                controller.GENERATOR.name,
                controller.LAUNCHER.name,
                controller.LAUNCHER.name,
                controller.SUMMARIZER.name,
            ],
        )
        self.assertIn("--dry-run", commands[1])
        self.assertEqual(commands[2][-2:], ["--submit", "--wait"])
        self.assertEqual(result.stop_reason, "requested_stop_stage_complete")

    def test_resume_after_midstage_leaves_completed_variant_to_strict_launcher(self) -> None:
        epoch_lock, variants = self._epoch_fixture()
        self._completion(variants[0])
        with mock.patch.object(
            controller.subprocess,
            "run",
            side_effect=self._success_runner_with_epoch_summary(epoch_lock),
        ) as run:
            controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in run.call_args_list]
        submit_commands = [command for command in commands if "--submit" in command]
        self.assertEqual(len(submit_commands), 1)
        self.assertIn("--stage-lock", submit_commands[0])
        self.assertNotIn("--only", submit_commands[0])

    def test_fully_synced_local_outputs_are_summarized_without_duplicate_submit(self) -> None:
        epoch_lock, variants = self._epoch_fixture()
        for variant in variants:
            self._completion(variant)
        with mock.patch.object(
            controller.subprocess,
            "run",
            side_effect=self._success_runner_with_epoch_summary(epoch_lock),
        ) as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual([Path(command[1]).name for command in commands], [controller.SUMMARIZER.name])
        self.assertTrue(all("--submit" not in command for command in commands))
        self.assertEqual(result.exit_code, 0)

    def test_pending_sheets_marker_uses_one_idempotent_launcher_resume(self) -> None:
        epoch_lock, variants = self._epoch_fixture()
        for variant in variants:
            self._completion(variant, sheets="pending")
        with mock.patch.object(
            controller.subprocess,
            "run",
            side_effect=self._success_runner_with_epoch_summary(epoch_lock),
        ) as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(sum("--submit" in command for command in commands), 1)
        self.assertEqual(
            [Path(command[1]).name for command in commands],
            [
                controller.GENERATOR.name,
                controller.LAUNCHER.name,
                controller.LAUNCHER.name,
                controller.SUMMARIZER.name,
            ],
        )
        self.assertFalse(
            controller.FORBIDDEN_LAUNCHER_FLAGS
            & {token for command in commands for token in command}
        )
        self.assertEqual(result.exit_code, 0)

    def test_missing_sheets_marker_uses_one_idempotent_launcher_resume(self) -> None:
        epoch_lock, variants = self._epoch_fixture()
        for variant in variants:
            self._completion(variant, sheets="missing")
        with mock.patch.object(
            controller.subprocess,
            "run",
            side_effect=self._success_runner_with_epoch_summary(epoch_lock),
        ) as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in run.call_args_list]
        self.assertEqual(sum("--submit" in command for command in commands), 1)
        self.assertEqual(sum("--dry-run" in command for command in commands), 1)
        self.assertEqual(result.exit_code, 0)

    def test_stale_sheets_marker_is_left_to_launcher_validator_and_fails_closed(self) -> None:
        _, variants = self._epoch_fixture()
        self._completion(variants[0], sheets="stale")
        self._completion(variants[1], sheets="synced")

        def run(argv, **kwargs):
            if Path(argv[1]).name == controller.LAUNCHER.name and "--submit" in argv:
                return subprocess.CompletedProcess(
                    argv,
                    1,
                    stdout="Refusing to trust a stale/mismatched synced Sheets marker",
                )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(controller.subprocess, "run", side_effect=run) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in mocked.call_args_list]
        self.assertEqual(sum("--submit" in command for command in commands), 1)
        self.assertFalse(any(Path(command[1]).name == controller.SUMMARIZER.name for command in commands))
        self.assertEqual(result.stop_reason, "kaggle_or_api_failure")
        self.assertEqual(result.exit_code, 2)

    def test_boundary_is_materialized_once_then_executed(self) -> None:
        normal = self._schema1_lock(
            effective_stage="epoch_line",
            source_stage="lr_log_line",
            target_stage="epoch_line",
            variants=[self._variant("epoch-two", "epoch-two-slug")],
        )
        self._write_schema1_summary(
            "epoch_line",
            needs_boundary=True,
            lock=normal,
        )
        boundary_path = self.locks / "epoch_line_boundary.lock.json"

        def run(argv, **kwargs):
            script = Path(argv[1]).name
            if script == controller.AXIS_MATERIALIZER.name:
                self._schema1_lock(
                    effective_stage="epoch_line",
                    source_stage="epoch_line",
                    target_stage="epoch_line",
                    variants=[self._variant("epoch-four", "epoch-four-slug")],
                    boundary=True,
                    path=boundary_path,
                )
            elif script == controller.SUMMARIZER.name:
                boundary = (
                    boundary_path,
                    controller.load_json_object(boundary_path, label="boundary lock"),
                )
                self._write_schema1_summary("epoch_line", lock=boundary)
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(controller.subprocess, "run", side_effect=run) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in mocked.call_args_list]
        self.assertEqual(Path(commands[0][1]).name, controller.AXIS_MATERIALIZER.name)
        self.assertIn("--boundary-extension", commands[0])
        self.assertEqual(sum("--boundary-extension" in command for command in commands), 1)
        self.assertEqual(result.stop_reason, "requested_stop_stage_complete")

    def test_dry_run_failure_stops_before_submit(self) -> None:
        self._epoch_fixture()

        def run(argv, **kwargs):
            if Path(argv[1]).name == controller.LAUNCHER.name and "--dry-run" in argv:
                return subprocess.CompletedProcess(argv, 1)
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(controller.subprocess, "run", side_effect=run) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in mocked.call_args_list]
        self.assertEqual(len(commands), 2)
        self.assertTrue(all("--submit" not in command for command in commands))
        self.assertEqual(result.stop_reason, "local_precheck_failure")

    def test_successful_resume_that_remains_pending_is_not_submitted_twice(self) -> None:
        self._epoch_fixture()
        with mock.patch.object(
            controller.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in mocked.call_args_list]
        self.assertEqual(sum("--submit" in command for command in commands), 1)
        self.assertEqual(result.stop_reason, "pending_stage_after_one_resume_attempt")

    def test_selection_ambiguity_stops_without_subprocess(self) -> None:
        self._write_schema1_summary(
            "lr_log_line",
            complete=True,
            decision_status="ambiguous",
        )
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
            ).run()
        run.assert_not_called()
        self.assertEqual(result.stop_reason, "selection_ambiguity")
        self.assertEqual(result.exit_code, 2)

    def test_stop_after_earlier_stage_never_resumes_later_pending_authority(self) -> None:
        epoch_lock, _ = self._epoch_fixture()
        self._write_schema1_summary(
            "epoch_line",
            complete=False,
            lock=epoch_lock,
            decision_status="pending",
        )
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="lr_log_line",
            ).run()
        run.assert_not_called()
        self.assertEqual(
            result.stop_reason,
            "requested_stop_stage_already_passed",
        )
        self.assertEqual(result.exit_code, 0)

    def test_pending_stage_with_future_coordinate_lock_fails_closed(self) -> None:
        epoch_lock, _ = self._epoch_fixture()
        future_lock = self._schema1_lock(
            effective_stage="regularization_coordinate_search__effective_batch",
            source_stage="epoch_line",
            target_stage="regularization_coordinate_search",
            coordinate="effective_batch",
            variants=[self._variant("batch-96", "batch-96-slug")],
        )
        self._write_schema1_summary(
            "epoch_line",
            complete=False,
            lock=epoch_lock,
            decision_status="pending",
        )
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
            ).run()
        run.assert_not_called()
        self.assertEqual(result.stop_reason, "local_authority_future_lock")
        self.assertEqual(
            result.state["planned_action"]["lock_sha256"],
            future_lock[1]["lock_payload_sha256"],
        )
        self.assertEqual(result.exit_code, 2)

    def test_pending_normal_stage_with_premature_same_stage_boundary_fails_closed(self) -> None:
        normal, _ = self._epoch_fixture()
        boundary = self._schema1_lock(
            effective_stage="epoch_line",
            source_stage="epoch_line",
            target_stage="epoch_line",
            variants=[self._variant("epoch-four", "epoch-four-slug")],
            boundary=True,
        )
        self._write_schema1_summary(
            "epoch_line",
            complete=False,
            lock=normal,
            decision_status="pending",
        )
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
            ).run()
        run.assert_not_called()
        self.assertEqual(
            result.stop_reason,
            "local_authority_premature_boundary_lock",
        )
        self.assertEqual(
            result.state["planned_action"]["lock_sha256"],
            boundary[1]["lock_payload_sha256"],
        )
        self.assertEqual(result.exit_code, 2)

    def test_pending_boundary_bound_summary_may_retain_prior_normal_lock(self) -> None:
        self._epoch_fixture()
        boundary = self._schema1_lock(
            effective_stage="epoch_line",
            source_stage="epoch_line",
            target_stage="epoch_line",
            variants=[self._variant("epoch-four", "epoch-four-slug")],
            boundary=True,
        )
        self._write_schema1_summary(
            "epoch_line",
            complete=False,
            lock=boundary,
            decision_status="pending",
        )
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                stop_after="epoch_line",
            ).run()
        run.assert_not_called()
        self.assertEqual(result.status, "plan_only")
        self.assertEqual(result.state["planned_action"]["kind"], "execute_stage")
        self.assertEqual(
            result.state["planned_action"]["lock_sha256"],
            boundary[1]["lock_payload_sha256"],
        )

    def test_quota_failure_stops_without_retry_or_followup_summary(self) -> None:
        self._epoch_fixture()

        def run(argv, **kwargs):
            if Path(argv[1]).name == controller.LAUNCHER.name and "--submit" in argv:
                return subprocess.CompletedProcess(argv, 1, stdout="GPU quota exceeded")
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(controller.subprocess, "run", side_effect=run) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        commands = [call.args[0] for call in mocked.call_args_list]
        self.assertEqual(len(commands), 3)
        self.assertEqual(sum("--submit" in command for command in commands), 1)
        self.assertEqual(result.stop_reason, "kaggle_quota_failure")

    def test_hard_cap_stops_before_materializing_next_coordinate(self) -> None:
        epoch_variants = [
            self._variant("epoch-two", "kernel-00"),
            self._variant("epoch-three", "kernel-01"),
        ]
        epoch_lock = self._schema1_lock(
            effective_stage="epoch_line",
            source_stage="lr_log_line",
            target_stage="epoch_line",
            variants=epoch_variants,
        )
        base_slugs = self._base_slugs()
        rows = [
            {"kernel_slug": slug, "completed": True}
            for slug in [*base_slugs, *(f"kernel-{index:02d}" for index in range(33))]
        ]
        self._write_schema1_summary("epoch_line", lock=epoch_lock, rows=rows)
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="regularization_coordinate_search__effective_batch",
            ).run()
        run.assert_not_called()
        self.assertEqual(result.stop_reason, "hard_kernel_cap_reached")
        self.assertEqual(result.exit_code, 2)

    def test_skipped_receipt_is_closed_by_summarizer_without_kaggle(self) -> None:
        primary = self._adaptive_lock(
            mode="loss_primary",
            effective_stage="special_loss_screen__primary",
            variants=[self._variant("balanced", "balanced-slug")],
            status="runnable",
            union=["balanced-slug"],
        )
        overlay = self._adaptive_lock(
            mode="loss_overlay",
            effective_stage="special_loss_screen__overlay",
            variants=[],
            status="skipped",
            union=["balanced-slug"],
        )
        self._write_adaptive_summary(
            "special_loss_screen__primary",
            closure=[primary[1]],
            union=["balanced-slug"],
        )

        def run(argv, **kwargs):
            self.assertEqual(Path(argv[1]).name, controller.SUMMARIZER.name)
            self._write_adaptive_summary(
                "special_loss_screen__overlay",
                closure=[primary[1], overlay[1]],
                union=["balanced-slug"],
            )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(controller.subprocess, "run", side_effect=run) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="special_loss_screen__overlay",
            ).run()
        self.assertEqual(mocked.call_count, 1)
        command = mocked.call_args.args[0]
        self.assertNotIn("--submit", command)
        self.assertEqual(result.exit_code, 0)

    def test_primary_materializer_uses_terminal_max_grad_lock_before_any_launcher(self) -> None:
        chain = self._schema1_chain_through_max_grad()
        max_grad = chain[-1]
        self._write_schema1_summary(
            "regularization_coordinate_search__max_grad_norm",
            lock=max_grad,
        )
        primary_path = self.locks / "special_loss_screen__primary.lock.json"
        prior_union = [
            *self._base_slugs(),
            *(variant["kernel_slug"] for _, lock in chain for variant in lock["resolved_stage"]["variants"]),
        ]
        primary_variants = [
            self._variant(f"loss-{index}", f"loss-{index}-slug")
            for index in range(4)
        ]

        def run(argv, **kwargs):
            script = Path(argv[1]).name
            if script == controller.ADAPTIVE_MATERIALIZER.name:
                self._adaptive_lock(
                    mode="loss_primary",
                    effective_stage="special_loss_screen__primary",
                    variants=primary_variants,
                    status="runnable",
                    union=[*prior_union, *(row["kernel_slug"] for row in primary_variants)],
                    path=primary_path,
                )
            elif script == controller.SUMMARIZER.name:
                primary = controller.load_json_object(primary_path, label="primary")
                self._write_adaptive_summary(
                    "special_loss_screen__primary",
                    closure=[primary],
                    union=[*prior_union, *(row["kernel_slug"] for row in primary_variants)],
                )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(controller.subprocess, "run", side_effect=run) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="special_loss_screen__primary",
            ).run()
        commands = [call.args[0] for call in mocked.call_args_list]
        self.assertEqual(Path(commands[0][1]).name, controller.ADAPTIVE_MATERIALIZER.name)
        self.assertEqual(commands[0][2], "loss_primary")
        prerequisite_index = commands[0].index("--prerequisite-lock")
        self.assertEqual(commands[0][prerequisite_index + 1], str(max_grad[0]))
        self.assertEqual(
            [Path(command[1]).name for command in commands[1:]],
            [
                controller.GENERATOR.name,
                controller.LAUNCHER.name,
                controller.LAUNCHER.name,
                controller.SUMMARIZER.name,
            ],
        )
        self.assertEqual(result.exit_code, 0)

    def test_lr_refine_and_confirmation_plans_preserve_prerequisite_order(self) -> None:
        union = [*self._base_slugs(), "primary-slug"]
        primary = self._adaptive_lock(
            mode="loss_primary",
            effective_stage="special_loss_screen__primary",
            variants=[self._variant("primary", "primary-slug")],
            status="runnable",
            union=union,
        )
        overlay = self._adaptive_lock(
            mode="loss_overlay",
            effective_stage="special_loss_screen__overlay",
            variants=[],
            status="skipped",
            union=union,
        )
        self._write_adaptive_summary(
            "special_loss_screen__overlay",
            closure=[primary[1], overlay[1]],
            union=union,
        )
        refine_plan = controller.CampaignController(
            paths=self.paths,
            stop_after="special_loss_screen__lr_refine",
        ).run()
        refine_command = refine_plan.state["planned_action"]["commands"][0]["argv"]
        self.assertEqual(refine_command[2], "loss_lr_refine")
        refine_prerequisites = [
            refine_command[index + 1]
            for index, token in enumerate(refine_command)
            if token == "--prerequisite-lock"
        ]
        self.assertEqual(refine_prerequisites, [str(primary[0]), str(overlay[0])])

        refine = self._adaptive_lock(
            mode="loss_lr_refine",
            effective_stage="special_loss_screen__lr_refine",
            variants=[],
            status="skipped",
            union=union,
        )
        self._write_adaptive_summary(
            "special_loss_screen__lr_refine",
            closure=[primary[1], overlay[1], refine[1]],
            union=union,
        )
        self.baseline.parent.mkdir(parents=True, exist_ok=True)
        self.baseline.write_text(
            json.dumps({"campaign": self.plan["campaign"], "runs": []}),
            encoding="utf-8",
        )
        confirmation_plan = controller.CampaignController(
            paths=self.paths,
            stop_after="confirmation__matched_seeds",
        ).run()
        confirmation_command = confirmation_plan.state["planned_action"]["commands"][0]["argv"]
        self.assertEqual(confirmation_command[2], "confirmation")
        confirmation_prerequisites = [
            confirmation_command[index + 1]
            for index, token in enumerate(confirmation_command)
            if token == "--prerequisite-lock"
        ]
        self.assertEqual(
            confirmation_prerequisites,
            [str(primary[0]), str(overlay[0]), str(refine[0])],
        )
        baseline_index = confirmation_command.index("--baseline-summary")
        self.assertEqual(confirmation_command[baseline_index + 1], str(self.baseline))

    def test_confirmation_runtime_gate_pauses_without_attestation(self) -> None:
        confirmation = self._adaptive_lock(
            mode="confirmation",
            effective_stage="confirmation__matched_seeds",
            variants=[self._variant("seed-confirm", "seed-confirm-slug")],
            status="runnable",
            union=["seed-confirm-slug"],
        )
        self._write_adaptive_summary(
            "confirmation__matched_seeds",
            closure=[confirmation[1]],
            union=["seed-confirm-slug"],
            complete=False,
            runs_complete=True,
            decision_status="runtime_gate_pending",
        )
        with mock.patch.object(controller.subprocess, "run") as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
            ).run()
        run.assert_not_called()
        self.assertEqual(result.stop_reason, "runtime_attestation_needed")
        self.assertEqual(result.exit_code, 0)

    def test_runtime_attestation_that_leaves_gate_pending_runs_summary_once(self) -> None:
        confirmation = self._adaptive_lock(
            mode="confirmation",
            effective_stage="confirmation__matched_seeds",
            variants=[self._variant("seed-confirm", "seed-confirm-slug")],
            status="runnable",
            union=["seed-confirm-slug"],
        )
        self._write_adaptive_summary(
            "confirmation__matched_seeds",
            closure=[confirmation[1]],
            union=["seed-confirm-slug"],
            complete=False,
            runs_complete=True,
            decision_status="runtime_gate_pending",
        )
        runtime_check = self.temp / "runtime-check.json"
        runtime_check.write_text("{}", encoding="utf-8")
        with mock.patch.object(
            controller.subprocess,
            "run",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                runtime_check=runtime_check,
            ).run()
        self.assertEqual(run.call_count, 1)
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, controller.SUMMARIZER.name)
        self.assertIn("--inference-runtime-check", command)
        self.assertNotIn("--submit", command)
        self.assertNotIn(controller.LAUNCHER.name, command)
        self.assertEqual(
            result.stop_reason,
            "runtime_gate_pending_after_one_attempt",
        )
        self.assertEqual(result.exit_code, 2)

    def test_confirmation_resume_then_runtime_attestation_have_independent_attempt_keys(self) -> None:
        confirmation = self._adaptive_lock(
            mode="confirmation",
            effective_stage="confirmation__matched_seeds",
            variants=[self._variant("seed-confirm", "seed-confirm-slug")],
            status="runnable",
            union=["seed-confirm-slug"],
        )
        self._write_adaptive_summary(
            "confirmation__matched_seeds",
            closure=[confirmation[1]],
            union=["seed-confirm-slug"],
            complete=False,
            runs_complete=False,
            decision_status="pending_runs",
        )
        runtime_check = self.temp / "runtime-check.json"
        runtime_check.write_text("{}", encoding="utf-8")

        def run(argv, **kwargs):
            if Path(argv[1]).name == controller.SUMMARIZER.name:
                # Both the ordinary post-training summary and the attested
                # summary intentionally leave the gate pending.  The latter
                # must still be attempted exactly once.
                self._write_adaptive_summary(
                    "confirmation__matched_seeds",
                    closure=[confirmation[1]],
                    union=["seed-confirm-slug"],
                    complete=False,
                    runs_complete=True,
                    decision_status="runtime_gate_pending",
                )
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch.object(controller.subprocess, "run", side_effect=run) as mocked:
            result = controller.CampaignController(
                paths=self.paths,
                submit=True,
                runtime_check=runtime_check,
            ).run()
        commands = [call.args[0] for call in mocked.call_args_list]
        summaries = [
            command
            for command in commands
            if Path(command[1]).name == controller.SUMMARIZER.name
        ]
        attested = [
            command for command in summaries if "--inference-runtime-check" in command
        ]
        ordinary = [
            command for command in summaries if "--inference-runtime-check" not in command
        ]
        self.assertEqual(len(ordinary), 1)
        self.assertEqual(len(attested), 1)
        self.assertEqual(sum("--submit" in command for command in commands), 1)
        self.assertEqual(
            result.stop_reason,
            "runtime_gate_pending_after_one_attempt",
        )
        self.assertEqual(result.exit_code, 2)

    def test_no_generated_command_uses_force_retry_or_background_flags(self) -> None:
        epoch_lock, _ = self._epoch_fixture()
        with mock.patch.object(
            controller.subprocess,
            "run",
            side_effect=self._success_runner_with_epoch_summary(epoch_lock),
        ) as run:
            controller.CampaignController(
                paths=self.paths,
                submit=True,
                stop_after="epoch_line",
            ).run()
        all_tokens = {
            token for call in run.call_args_list for token in call.args[0]
        }
        self.assertFalse(controller.FORBIDDEN_LAUNCHER_FLAGS & all_tokens)


if __name__ == "__main__":
    unittest.main()
