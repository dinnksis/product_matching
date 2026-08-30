from __future__ import annotations

import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr
from copy import deepcopy
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_minilm_5ep_sft_fast_loss_confirmation.py"
LAUNCHER_MODULE_PATH = (
    ROOT / "scripts" / "run_minilm_5ep_sft_fast_loss_confirmation_kaggle.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("fast_loss_confirmation", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_launcher_module():
    spec = importlib.util.spec_from_file_location(
        "fast_loss_confirmation_launcher", LAUNCHER_MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FastLossConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module()
        cls.launcher_module = load_launcher_module()
        cls.plan_path = ROOT / "configs" / "minilm_5ep_sft_hparam_search_v1.json"
        cls.protocol_path = ROOT / "configs" / "minilm_5ep_sft_fast_loss_confirm_v1.json"
        cls.plan = cls.module.builder.load_plan(cls.plan_path)
        cls.protocol = cls.module.load_protocol(cls.protocol_path, plan=cls.plan)
        cls.fast_policy = cls.module.fast_track.load_policy()
        cls.fast_freeze = cls.module.fast_track.load_freeze_manifest()
        cls.fast_receipt = cls.module.fast_track.validate_receipt(
            policy=cls.fast_policy,
            freeze_manifest_path=cls.module.fast_track.DEFAULT_FREEZE_MANIFEST,
        )

    @staticmethod
    def fixture():
        comparator_origin = {
            "origin_id": "origin-bce",
            "experiment": "bce-s42",
            "kernel_slug": "bce-s42-slug",
            "run_id": "bce-run-42",
            "resolved_config": {"seed": 42, "learning_rate": 8e-5},
            "loss_variant": "bce",
            "loss_hook_sha256": "a" * 64,
            "recipe_sha256": "b" * 64,
            "recipe_family_sha256": "c" * 64,
            "iid_predictions_sha256": "d" * 64,
            "completion_sha256": "e" * 64,
        }
        candidate_origin = {
            "origin_id": "origin-loss",
            "experiment": "loss-s42",
            "kernel_slug": "loss-s42-slug",
            "run_id": "loss-run-42",
            "resolved_config": {"seed": 42, "learning_rate": 4e-5},
            "loss_variant": "balanced_binary_bce",
            "loss_hook_sha256": "f" * 64,
            "recipe_sha256": "1" * 64,
            "recipe_family_sha256": "2" * 64,
            "iid_predictions_sha256": "3" * 64,
            "completion_sha256": "4" * 64,
        }
        groups = [
            {
                "recipe_group_id": "group-bce",
                "recipe_family_sha256": "c" * 64,
                "roles": ["selected_regularized_bce_recipe"],
                "origin_seed42_id": "origin-bce",
            },
            {
                "recipe_group_id": "group-loss",
                "recipe_family_sha256": "2" * 64,
                "roles": ["loss_finalist_1"],
                "origin_seed42_id": "origin-loss",
            },
        ]

        def entry(group, experiment, slug, config, loss, hook):
            return {
                "experiment": experiment,
                "kernel_slug": slug,
                "recipe_sha256": "9" * 64,
                "source_sha256": "8" * 64,
                "loss_variant": loss,
                "loss_hook_sha256": hook,
                "expected_config": config,
                "expected_notes": "{}",
                "variant": {"recipe_group_id": group},
            }

        variants = [
            entry(
                "group-bce",
                "bce-s17",
                "bce-s17-slug",
                {"seed": 17, "learning_rate": 8e-5},
                "bce",
                "a" * 64,
            ),
            entry(
                "group-loss",
                "loss-s17",
                "loss-s17-slug",
                {"seed": 17, "learning_rate": 4e-5},
                "balanced_binary_bce",
                "f" * 64,
            ),
            entry(
                "group-bce",
                "bce-s2026",
                "bce-s2026-slug",
                {"seed": 2026, "learning_rate": 8e-5},
                "bce",
                "a" * 64,
            ),
            entry(
                "group-loss",
                "loss-s2026",
                "loss-s2026-slug",
                {"seed": 2026, "learning_rate": 4e-5},
                "balanced_binary_bce",
                "f" * 64,
            ),
        ]
        lock = {
            "schema_version": 2,
            "mode": "confirmation",
            "lock_payload_sha256": "7" * 64,
            "decision": {"seed42_reuse": True, "seeds": [17, 42, 2026]},
            "origins": [comparator_origin, candidate_origin],
            "resolved_stage": {
                "recipe_groups": groups,
                "variants": [
                    {"experiment": variant["experiment"]} for variant in variants
                ],
            },
        }
        return lock, {"variants": variants}

    def test_protocol_pins_prespecified_second_seed_and_no_ods(self) -> None:
        self.assertEqual(self.protocol["screen_seed"], 42)
        self.assertEqual(self.protocol["replication_seed"], 17)
        self.assertFalse(self.protocol["ods_runtime_check"])
        self.assertEqual(self.protocol["execution"]["new_kernels"], 2)
        self.assertEqual(self.protocol["execution"]["google_sheets_tab"], "sft_exps")
        self.assertEqual(
            self.protocol["fast_track_authority"]["freeze_manifest_path"],
            "configs/minilm_5ep_sft_loss_fast_track_v1.freeze.json",
        )
        self.assertEqual(
            self.protocol["fast_track_authority"]["freeze_manifest_file_sha256"],
            self.module.file_sha256(self.module.fast_track.DEFAULT_FREEZE_MANIFEST),
        )
        self.assertEqual(
            self.protocol["fast_track_authority"][
                "freeze_manifest_payload_sha256"
            ],
            self.fast_freeze["manifest_payload_sha256"],
        )
        self.assertEqual(
            self.protocol["fast_track_authority"]["receipt_file_sha256"],
            self.module.file_sha256(self.module.fast_track.DEFAULT_RECEIPT),
        )
        self.assertEqual(
            self.protocol["fast_track_authority"]["receipt_payload_sha256"],
            self.fast_receipt["summary_payload_sha256"],
        )
        self.assertEqual(
            self.protocol["fast_track_authority"]["legacy_receipt_freeze"],
            self.fast_freeze["semantic_contract"]["legacy_receipt_freeze"],
        )
        self.assertEqual(
            self.protocol["recipes"]["candidate_rule"],
            "best_non_bce_after_optional_lr_refinement",
        )
        self.assertFalse(self.protocol["checkpoint_handling"]["download_supported"])

    def test_protocol_rejects_extra_keys_and_selection_semantic_drift(self) -> None:
        for mutation in (
            lambda payload: payload.update(extra_authority=True),
            lambda payload: payload["recipes"].update(candidate_rule="post_hoc_best"),
            lambda payload: payload["selection"].update(comparison="hard_macro_ap"),
            lambda payload: payload["selection"].update(rationale="changed"),
            lambda payload: payload["fast_track_authority"].update(
                freeze_manifest_payload_sha256="0" * 64
            ),
            lambda payload: payload["fast_track_authority"].update(
                receipt_payload_sha256="0" * 64
            ),
            lambda payload: payload["fast_track_authority"][
                "legacy_receipt_freeze"
            ].update(file_sha256="0" * 64),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                payload = deepcopy(self.protocol)
                mutation(payload)
                path = Path(directory) / "protocol.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaises(self.module.FastConfirmationError):
                    self.module.load_protocol(path, plan=self.plan)

    def test_confirmation_rejects_an_alternate_receipt_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "alternate-receipt.json"
            alternate.write_text("{}", encoding="utf-8")
            receipt = {
                "policy_authority": {
                    "path": str(self.module.fast_track.DEFAULT_POLICY.resolve()),
                    "canonical_sha256": self.module.fast_track.policy_sha256(
                        self.fast_policy
                    ),
                },
                "source_plan": {
                    "canonical_sha256": self.protocol["source_plan_canonical_sha256"]
                },
                "freeze_authority": {
                    "path": str(
                        self.module.fast_track.DEFAULT_FREEZE_MANIFEST.resolve()
                    ),
                    "file_sha256": self.module.file_sha256(
                        self.module.fast_track.DEFAULT_FREEZE_MANIFEST
                    ),
                    "manifest_payload_sha256": self.fast_freeze[
                        "manifest_payload_sha256"
                    ],
                    "reviewed_execution_file_sha256": self.fast_freeze[
                        "reviewed_execution_file_sha256"
                    ],
                    "reviewed_review_file_sha256": self.fast_freeze[
                        "reviewed_review_file_sha256"
                    ],
                },
                "external_actions": deepcopy(self.fast_policy["external_actions"]),
            }
            with self.assertRaisesRegex(
                self.module.FastConfirmationError, "exact reviewed"
            ):
                self.module._validate_fast_track_binding(
                    protocol=self.protocol,
                    policy_path=self.module.fast_track.DEFAULT_POLICY,
                    receipt_path=alternate,
                    freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                    policy=self.fast_policy,
                    receipt=receipt,
                    freeze=self.fast_freeze,
                )

    def test_confirmation_rejects_detached_freeze_tamper(self) -> None:
        receipt = deepcopy(self.fast_receipt)
        receipt["freeze_authority"]["manifest_payload_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            self.module.FastConfirmationError, "Legacy receipt detached freeze"
        ):
            self.module._validate_fast_track_binding(
                protocol=self.protocol,
                policy_path=self.module.fast_track.DEFAULT_POLICY,
                receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
                freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                policy=self.fast_policy,
                receipt=receipt,
                freeze=self.fast_freeze,
            )

    def test_fail_closed_decision_rule(self) -> None:
        accepted = self.module.decide_loss(
            screen_delta=0.003,
            replication_delta=0.0015,
            threshold=0.002,
        )
        self.assertTrue(accepted["accepted"])
        self.assertEqual(accepted["selected_role"], "loss_finalist_1")
        negative_second_seed = self.module.decide_loss(
            screen_delta=0.01,
            replication_delta=-0.00001,
            threshold=0.002,
        )
        self.assertFalse(negative_second_seed["accepted"])
        below_margin = self.module.decide_loss(
            screen_delta=0.001,
            replication_delta=0.001,
            threshold=0.002,
        )
        self.assertFalse(below_margin["accepted"])
        exact_margin = self.module.decide_loss(
            screen_delta=0.003,
            replication_delta=0.001,
            threshold=0.002,
        )
        self.assertTrue(exact_margin["accepted"])

    def test_absolute_pair_metrics_and_four_run_ids_are_independently_bound(self) -> None:
        comparison = {
            "baseline_macro_average_precision": 0.80,
            "candidate_macro_average_precision": 0.81,
            "delta_macro_average_precision": 0.01,
        }
        self.assertAlmostEqual(
            self.module._validate_paired_iid_absolute_metrics(
                comparison,
                seed=17,
                comparator_ap=0.80,
                candidate_ap=0.81,
            ),
            0.01,
        )
        drifted_absolute = {
            **comparison,
            "baseline_macro_average_precision": 0.79,
            "candidate_macro_average_precision": 0.80,
        }
        with self.assertRaisesRegex(self.module.FastConfirmationError, "baseline AP"):
            self.module._validate_paired_iid_absolute_metrics(
                drifted_absolute,
                seed=17,
                comparator_ap=0.80,
                candidate_ap=0.81,
            )
        identities = {
            42: {
                "comparator": {"run_id": "run-a"},
                "candidate": {"run_id": "run-b"},
            },
            17: {
                "comparator": {"run_id": "run-c"},
                "candidate": {"run_id": "run-d"},
            },
        }
        self.assertEqual(len(self.module._require_four_unique_run_ids(identities)), 4)
        identities[17]["candidate"]["run_id"] = "run-a"
        with self.assertRaisesRegex(self.module.FastConfirmationError, "four unique"):
            self.module._require_four_unique_run_ids(identities)
        self.assertEqual(
            self.module._require_validation_run_id_match(
                {"run_id": "run-c"}, {"run_id": "run-c"}, side="candidate"
            ),
            "run-c",
        )
        with self.assertRaisesRegex(self.module.FastConfirmationError, "another run_id"):
            self.module._require_validation_run_id_match(
                {"run_id": "run-x"}, {"run_id": "run-y"}, side="candidate"
            )
        self.assertEqual(
            self.module._require_bound_completion_run_id_match(
                bound_run_id="run-a",
                completion={"run_id": "run-a"},
                label="seed-42 comparator",
            ),
            "run-a",
        )
        with self.assertRaisesRegex(self.module.FastConfirmationError, "another run_id"):
            self.module._require_bound_completion_run_id_match(
                bound_run_id="run-a",
                completion={"run_id": "run-z"},
                label="seed-42 comparator",
            )

    def test_selects_only_matched_bce_and_first_loss_at_seed17(self) -> None:
        lock, contract = self.fixture()
        with mock.patch.object(
            self.module.builder,
            "normalized_campaign_execution_contract",
            return_value=contract,
        ):
            selected = self.module.select_fast_pair(
                plan=self.plan,
                lock=lock,
                protocol=self.protocol,
            )
        experiments = {
            selected[side]["entry"]["experiment"] for side in ("comparator", "candidate")
        }
        self.assertEqual(experiments, {"bce-s17", "loss-s17"})
        self.assertEqual(selected["comparator"]["entry"]["expected_config"]["seed"], 17)
        self.assertEqual(selected["candidate"]["origin"]["resolved_config"]["seed"], 42)
        self.assertEqual(
            selected["candidate"]["entry"]["baseline_metrics"],
            self.plan["baseline_metrics"],
        )

    def test_rejects_recipe_drift_beyond_seed(self) -> None:
        lock, contract = self.fixture()
        contract = deepcopy(contract)
        contract["variants"][0]["expected_config"]["learning_rate"] = 1e-5
        with mock.patch.object(
            self.module.builder,
            "normalized_campaign_execution_contract",
            return_value=contract,
        ):
            with self.assertRaisesRegex(self.module.FastConfirmationError, "beyond seed"):
                self.module.select_fast_pair(
                    plan=self.plan,
                    lock=lock,
                    protocol=self.protocol,
                )

    def test_submit_wrapper_is_sequential_and_never_forces_retry_or_fanout(self) -> None:
        manifest = {
            "launch_order": ["bce-s17", "loss-s17"],
            "fast_track_policy_path": self.module._relative(
                self.module.fast_track.DEFAULT_POLICY
            ),
            "fast_track_policy_sha256": self.module.fast_track.policy_sha256(
                self.fast_policy
            ),
            "fast_track_receipt_path": "receipt.json",
            "fast_track_receipt_file_sha256": "receipt-file-sha",
            "fast_track_receipt_payload_sha256": "5" * 64,
            "fast_track_freeze_manifest_path": self.module._relative(
                self.module.fast_track.DEFAULT_FREEZE_MANIFEST
            ),
            "fast_track_freeze_manifest_file_sha256": self.module.file_sha256(
                self.module.fast_track.DEFAULT_FREEZE_MANIFEST
            ),
            "fast_track_freeze_manifest_payload_sha256": self.fast_freeze[
                "manifest_payload_sha256"
            ],
            "ods_runtime_check": False,
            "execution_projection": {
                "kind": "exact_two_entry_seed17_projection",
                "runnable_entries": [
                    {"experiment": "bce-s17", "seed": 17},
                    {"experiment": "loss-s17", "seed": 17},
                ],
                "runnable_entry_count": 2,
                "forbidden_training_seeds": [2026],
                "allows_arbitrary_experiment_passthrough": False,
                "runtime_attestation_path_exposed": False,
                "ods_submission_path_exposed": False,
                "checkpoint_download_path_exposed": False,
            },
        }
        calls = []

        def record(command, **kwargs):
            calls.append(command)
            return mock.Mock(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            receipt_path = temporary / "receipt.json"
            receipt_path.write_text("{}", encoding="utf-8")
            lock_path = temporary / self.module.DEFAULT_SOURCE_LOCK_NAME
            lock_path.write_text("{}", encoding="utf-8")
            manifest["fast_track_receipt_path"] = self.module._relative(receipt_path)
            manifest["fast_track_receipt_file_sha256"] = self.module.file_sha256(
                receipt_path
            )
            manifest_path = temporary / self.module.DEFAULT_MANIFEST_NAME
            manifest_path.write_text(
                self.module.canonical_json_dumps(manifest) + "\n", encoding="utf-8"
            )
            with mock.patch.object(
                self.module.subprocess, "run", side_effect=record
            ), mock.patch.object(
                self.module.fast_track,
                "validate_receipt",
                return_value={"summary_payload_sha256": "5" * 64},
            ), mock.patch.object(
                self.module.fast_track,
                "load_freeze_manifest",
                return_value=self.fast_freeze,
            ), mock.patch.object(
                self.module.fast_track,
                "DEFAULT_RECEIPT",
                receipt_path,
            ), mock.patch.object(
                self.module,
                "_validate_fast_track_binding",
            ), mock.patch.object(
                self.module,
                "_validate_execution_projection",
                return_value=["bce-s17", "loss-s17"],
            ):
                self.module.run_selected(
                    manifest=manifest,
                    plan_path=self.plan_path,
                    lock_path=lock_path,
                    env_file=ROOT / ".env",
                    action="submit",
                    policy_path=self.module.fast_track.DEFAULT_POLICY,
                    receipt_path=receipt_path,
                    freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                )
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            Path(calls[0][1]).name,
            "run_minilm_5ep_sft_fast_loss_confirmation_kaggle.py",
        )
        self.assertIn("--fast-track-policy", calls[0])
        self.assertIn("--fast-track-receipt", calls[0])
        self.assertIn("--fast-track-freeze-manifest", calls[0])
        self.assertIn("--fast-confirmation-manifest", calls[0])
        self.assertIn("--submit", calls[0])
        self.assertIn("--wait", calls[0])
        self.assertNotIn("--only", calls[0])
        flattened = {token for call in calls for token in call}
        self.assertNotIn("--force-resubmit", flattened)
        self.assertNotIn("--retry-failed", flattened)
        self.assertNotIn("--allow-background-fanout", flattened)

    def test_narrow_launcher_derives_pair_and_forwards_exact_sequential_calls(self) -> None:
        entries = [
            {"experiment": "bce-s17", "kernel_slug": "bce-s17-slug"},
            {"experiment": "loss-s17", "kernel_slug": "loss-s17-slug"},
        ]
        with mock.patch.object(
            self.launcher_module,
            "load_bound_execution",
            return_value=({}, entries),
        ), mock.patch.object(
            self.launcher_module.fast_track, "run_forwarded_main"
        ) as forwarded, mock.patch.object(
            self.launcher_module.core_launcher,
            "validate_run_output",
            side_effect=[{"run_id": "run-bce"}, {"run_id": "run-loss"}],
        ) as validate:
            result = self.launcher_module.execute(
                manifest_path=ROOT / "reports" / "manifest.json",
                plan_path=self.plan_path,
                lock_path=ROOT / "reports" / "confirmation.lock.json",
                env_file=ROOT / ".env",
                policy_path=self.module.fast_track.DEFAULT_POLICY,
                receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
                freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                action="submit",
            )
        self.assertEqual(result, [{"run_id": "run-bce"}, {"run_id": "run-loss"}])
        self.assertEqual(forwarded.call_count, 4)
        self.assertEqual(validate.call_count, 2)
        forwarded_argv = [call.args[1] for call in forwarded.call_args_list]
        self.assertEqual(
            [argv[argv.index("--only") + 1] for argv in forwarded_argv],
            ["bce-s17", "bce-s17", "loss-s17", "loss-s17"],
        )
        self.assertEqual(
            [("--dry-run" in argv, "--submit" in argv) for argv in forwarded_argv],
            [(True, False), (False, True), (True, False), (False, True)],
        )
        forbidden = {
            "--force-resubmit",
            "--retry-failed",
            "--allow-background-fanout",
            "--no-wait",
            "--status",
            "--full-download",
        }
        self.assertFalse(forbidden & {token for argv in forwarded_argv for token in argv})
        for call in forwarded.call_args_list:
            self.assertEqual(
                call.kwargs["freeze_manifest_path"],
                self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
            )

    def test_narrow_launcher_cli_has_no_arbitrary_selector_or_status_path(self) -> None:
        base = [
            "--fast-confirmation-manifest",
            "/tmp/manifest.json",
            "--stage-lock",
            "/tmp/confirmation.lock.json",
        ]
        for extra in (
            ["--only", "bce-s17", "--dry-run"],
            ["--status"],
            ["--submit"],
            ["--dry-run", "--wait"],
            ["--submit", "--wait", "--force-resubmit"],
        ):
            with self.subTest(extra=extra), redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit):
                self.launcher_module.parse_args([*base, *extra])

    def test_narrow_launcher_requires_default_plan_authority(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alternate_plan = Path(directory) / "plan.json"
            alternate_plan.write_text(
                self.plan_path.read_text(encoding="utf-8"), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                self.launcher_module.confirmation.FastConfirmationError,
                "default frozen plan",
            ):
                self.launcher_module.load_bound_execution(
                    manifest_path=Path(directory) / "missing-manifest.json",
                    plan_path=alternate_plan,
                    lock_path=Path(directory) / "missing-lock.json",
                    policy_path=self.module.fast_track.DEFAULT_POLICY,
                    receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
                    freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                )

    def test_narrow_launcher_requires_exact_colocated_manifest_and_lock_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            lock_path = temporary / "copied-confirmation.lock.json"
            manifest_path = temporary / self.module.DEFAULT_MANIFEST_NAME
            lock_path.write_text("{}", encoding="utf-8")
            manifest_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(
                self.launcher_module.confirmation.FastConfirmationError,
                "exact source lock and colocated manifest",
            ):
                self.launcher_module.load_bound_execution(
                    manifest_path=manifest_path,
                    plan_path=self.plan_path,
                    lock_path=lock_path,
                    policy_path=self.module.fast_track.DEFAULT_POLICY,
                    receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
                    freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                )

    def test_prepare_fails_before_loss_locks_without_materializing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            receipt_path = temporary / "receipt.json"
            receipt_path.write_text("{}", encoding="utf-8")
            receipt = {"summary_payload_sha256": "5" * 64}
            with mock.patch.object(
                self.module.adaptive, "materialize"
            ) as materialize, mock.patch.object(
                self.module.fast_track,
                "patched_loss_predecessor",
                return_value=nullcontext(
                    (self.fast_policy, receipt, self.fast_freeze)
                ),
            ), mock.patch.object(
                self.module,
                "_validate_fast_track_binding",
            ):
                with self.assertRaisesRegex(self.module.FastConfirmationError, "waits for all loss"):
                    self.module.prepare(
                        protocol_path=self.protocol_path,
                        plan_path=self.plan_path,
                        summary_path=ROOT / "missing-summary.json",
                        baseline_summary_path=ROOT / "missing-baseline.json",
                        locks_dir=temporary / "locks",
                        artifacts_dir=temporary / "artifacts",
                        output_dir=temporary / "output",
                        policy_path=self.module.fast_track.DEFAULT_POLICY,
                        receipt_path=receipt_path,
                        freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                    )
            materialize.assert_not_called()

    def test_sheets_verification_requires_exact_sft_exps_identity(self) -> None:
        run_id = "run-17"
        valid = {
            "status": "synced",
            "run_id": run_id,
            "experiment_group": "sft",
            "comparison_sheet": "sft_exps",
            "spreadsheet_id": self.module.launcher.qwen_builder.EXPERIMENT_SPREADSHEET_ID,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "google_sheets_sync.json").write_text(
                json.dumps(valid), encoding="utf-8"
            )
            self.assertEqual(
                self.module._validate_sft_exps_sync(root, run_id=run_id)["status"],
                "synced",
            )
            invalid = {**valid, "comparison_sheet": "data_exps"}
            (root / "google_sheets_sync.json").write_text(
                json.dumps(invalid), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                self.module.FastConfirmationError, "not verifiably synchronized"
            ):
                self.module._validate_sft_exps_sync(root, run_id=run_id)

    def test_manifest_names_only_two_new_experiments(self) -> None:
        lock, contract = self.fixture()
        with mock.patch.object(
            self.module.builder,
            "normalized_campaign_execution_contract",
            return_value=contract,
        ):
            selected = self.module.select_fast_pair(
                plan=self.plan,
                lock=lock,
                protocol=self.protocol,
            )
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            lock_path = Path(directory) / "source.lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            manifest = self.module.build_manifest(
                protocol_path=self.protocol_path,
                protocol=self.protocol,
                plan_path=self.plan_path,
                lock_path=lock_path,
                lock=lock,
                selected=selected,
                policy_path=self.module.fast_track.DEFAULT_POLICY,
                policy=self.fast_policy,
                receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
                receipt=self.fast_receipt,
                freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                freeze=self.fast_freeze,
            )
            self.assertEqual(
                self.module._validate_execution_projection(
                    manifest,
                    plan_path=self.plan_path,
                    lock_path=lock_path,
                ),
                ["bce-s17", "loss-s17"],
            )
            tampered = deepcopy(manifest)
            tampered["execution_projection"]["runnable_entries"][0]["seed"] = 2026
            tampered.pop("manifest_payload_sha256")
            tampered["manifest_payload_sha256"] = self.module.canonical_sha256(tampered)
            with self.assertRaises(self.module.FastConfirmationError):
                self.module._validate_execution_projection(
                    tampered,
                    plan_path=self.plan_path,
                    lock_path=lock_path,
                )
        self.assertEqual(manifest["launch_order"], ["bce-s17", "loss-s17"])
        self.assertEqual(manifest["new_kernel_count"], 2)
        self.assertNotIn("2026", json.dumps(manifest["launch_order"]))
        unhashed = dict(manifest)
        stored = unhashed.pop("manifest_payload_sha256")
        self.assertEqual(stored, self.module.canonical_sha256(unhashed))

    def test_summary_uses_direct_two_seed_pair_and_writes_completion(self) -> None:
        lock, contract = self.fixture()
        with mock.patch.object(
            self.module.builder,
            "normalized_campaign_execution_contract",
            return_value=contract,
        ):
            selected = self.module.select_fast_pair(
                plan=self.plan,
                lock=lock,
                protocol=self.protocol,
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = root / "artifacts"
            output = root / "report"
            scores = {
                (42, "comparator"): (0.800, 0.40, 0.60),
                (42, "candidate"): (0.803, 0.41, 0.59),
                (17, "comparator"): (0.790, 0.39, 0.61),
                (17, "candidate"): (0.791, 0.42, 0.62),
            }

            def write_run(path, run_id, values):
                path.mkdir(parents=True)
                completion = {
                    "run_id": run_id,
                    "training_report": {
                        "validation_splits": {
                            split: {"macro_average_precision": value}
                            for split, value in zip(("iid", "hard", "ood"), values)
                        }
                    },
                }
                (path / "notebook_completed.json").write_text(
                    json.dumps(completion), encoding="utf-8"
                )
                (path / "iid_validation_predictions.parquet").write_bytes(b"stub")
                (path / "google_sheets_sync.json").write_text(
                    json.dumps(
                        {
                            "status": "synced",
                            "run_id": run_id,
                            "experiment_group": "sft",
                            "comparison_sheet": "sft_exps",
                            "spreadsheet_id": self.module.launcher.qwen_builder.EXPERIMENT_SPREADSHEET_ID,
                        }
                    ),
                    encoding="utf-8",
                )

            for side in ("comparator", "candidate"):
                origin = selected[side]["origin"]
                origin_dir = root / f"origin-{side}"
                write_run(origin_dir, str(origin["run_id"]), scores[(42, side)])
                origin["completion_artifact_path"] = str(
                    origin_dir / "notebook_completed.json"
                )
                entry = selected[side]["entry"]
                replication_dir = artifacts / str(entry["kernel_slug"])
                write_run(replication_dir, f"{side}-run-17", scores[(17, side)])

            comparisons = iter(
                [
                    {
                        "baseline_macro_average_precision": 0.800,
                        "candidate_macro_average_precision": 0.803,
                        "delta_macro_average_precision": 0.003,
                        "p_value": 0.01,
                        "ci95_low": 0.001,
                        "ci95_high": 0.005,
                        "permutations": 2000,
                        "bootstrap_resamples": 2000,
                        "seed": 42,
                    },
                    {
                        "baseline_macro_average_precision": 0.790,
                        "candidate_macro_average_precision": 0.791,
                        "delta_macro_average_precision": 0.001,
                        "p_value": 0.20,
                        "ci95_low": -0.001,
                        "ci95_high": 0.003,
                        "permutations": 2000,
                        "bootstrap_resamples": 2000,
                        "seed": 42,
                    },
                ]
            )
            with mock.patch.object(
                self.module.launcher,
                "validate_run_output",
                side_effect=lambda directory, entry: {
                    "run_id": json.loads(
                        (directory / "notebook_completed.json").read_text()
                    )["run_id"],
                    "directory": str(directory),
                },
            ) as validate, mock.patch.object(
                self.module.summarizer,
                "read_prediction_artifact",
                return_value=mock.Mock(),
            ), mock.patch.object(
                self.module.summarizer,
                "cached_anchor_comparison",
                side_effect=lambda **kwargs: next(comparisons),
            ):
                result = self.module.summarize(
                    protocol=self.protocol,
                    manifest={"manifest_payload_sha256": "6" * 64},
                    selected=selected,
                    artifacts_dir=artifacts,
                    output_dir=output,
                )
            self.assertEqual(validate.call_count, 2)
            self.assertTrue(result["selection"]["accepted"])
            self.assertEqual(
                result["selected_seed42_recipe_reference"]["loss_variant"],
                "balanced_binary_bce",
            )
            self.assertFalse(result["checkpoint_download_performed"])
            self.assertTrue((output / "summary.json").is_file())
            self.assertTrue((output / "runs.csv").is_file())
            marker = json.loads((output / "fast_confirmation_completed.json").read_text())
            self.assertEqual(marker["status"], "complete")

    def test_actual_receipt_and_active_primary_chain_validate_read_only(self) -> None:
        self.module._validate_fast_track_binding(
            protocol=self.protocol,
            policy_path=self.module.fast_track.DEFAULT_POLICY,
            receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
            freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
            policy=self.fast_policy,
            receipt=self.fast_receipt,
            freeze=self.fast_freeze,
        )
        archive_path, archived = self.module._load_legacy_receipt_freeze(
            protocol=self.protocol,
            freeze=self.fast_freeze,
        )
        self.assertEqual(
            self.fast_receipt["freeze_authority"]["file_sha256"],
            self.module.file_sha256(archive_path),
        )
        self.assertEqual(
            self.fast_receipt["freeze_authority"]["manifest_payload_sha256"],
            archived["manifest_payload_sha256"],
        )

        primary_path = (
            ROOT
            / "reports"
            / "minilm_5ep_sft_hparam_search_v1"
            / "stage_locks"
            / "special_loss_screen__primary.lock.json"
        )
        with self.assertRaises(self.module.builder.CampaignConfigError):
            self.module.builder.load_campaign_lock(primary_path, plan=self.plan)
        scoped = self.module.fast_track.load_scoped_loss_lock(
            plan_path=self.module.fast_track.DEFAULT_PLAN,
            stage_lock_path=primary_path,
            policy_path=self.module.fast_track.DEFAULT_POLICY,
            receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
            freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
        )
        variants = scoped["resolved_stage"]["variants"]
        self.assertEqual(scoped["mode"], "loss_primary")
        self.assertEqual(scoped["execution_status"], "runnable")
        self.assertEqual(len(variants), 4)
        self.assertEqual({variant["seed"] for variant in variants}, {42})
        self.assertTrue(
            all(
                variant["kernel_slug"] == variant["title"]
                and len(variant["kernel_slug"]) <= 50
                for variant in variants
            )
        )
        with self.module.fast_track.patched_loss_predecessor(
            policy_path=self.module.fast_track.DEFAULT_POLICY,
            receipt_path=self.module.fast_track.DEFAULT_RECEIPT,
            freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
        ):
            loaded = self.module.builder.load_campaign_lock(
                primary_path,
                plan=self.plan,
            )
            closure = self.module.summarizer.load_adaptive_prerequisite_closure(
                plan=self.plan,
                root_lock=loaded,
                root_lock_path=primary_path,
                base_config=self.module.builder.cross_builder.load_training_config(
                    self.module.builder.BASE_CONFIG_PATH
                ),
            )
            self.assertEqual(closure[-1][0]["mode"], "loss_primary")
        with self.assertRaises(self.module.builder.CampaignConfigError):
            self.module.builder.load_campaign_lock(primary_path, plan=self.plan)

    def test_real_fast_track_lock_chain_is_scoped_and_projects_exactly_two(self) -> None:
        fixture_spec = importlib.util.spec_from_file_location(
            "fast_confirmation_adaptive_fixtures",
            ROOT / "tests" / "test_materialize_minilm_5ep_sft_loss_confirmation.py",
        )
        assert fixture_spec is not None and fixture_spec.loader is not None
        fixture_module = importlib.util.module_from_spec(fixture_spec)
        fixture_spec.loader.exec_module(fixture_module)
        fixture = fixture_module.LossConfirmationMaterializerTest(methodName="runTest")
        fixture.setUp()
        try:
            dropout_stage = self.module.fast_track.SOURCE_STAGE
            report_dir = fixture.root / "fast-track-report"
            history_locks_dir = report_dir / "stage_locks"
            history_locks_dir.mkdir(parents=True)
            extra_history_slugs = [
                "fast-confirm-historical-boundary-a",
                "fast-confirm-historical-boundary-b",
            ]

            def mark_synced(row: Mapping[str, Any]) -> None:
                directory = fixture.artifacts / str(row["kernel_slug"])
                completion_path = directory / "notebook_completed.json"
                completion = json.loads(completion_path.read_text(encoding="utf-8"))
                completion["experiment_group"] = "sft"
                completion["train_data"] = {
                    "train_pairs": 306_669,
                    "items": 711_304,
                    "positive_rate": 0.26131105524197096,
                    "same_size_as_human_baseline": True,
                    "train_pairs_sha256": "001bb234bb631291a17fe3822989f9b47475a47185224a5f25b88a28fc6169a3",
                    "items_sha256": "5491ebbfd891a396af8c0b1a4b16b61b3ed89a19c2b84d407b059096794511e0",
                    "label_source_counts": {"unspecified": 306_669},
                }
                completion_path.write_text(json.dumps(completion), encoding="utf-8")
                (directory / "google_sheets_sync.json").write_text(
                    json.dumps(
                        {
                            "status": "synced",
                            "run_id": row["run_id"],
                            "experiment_group": "sft",
                            "comparison_sheet": "sft_exps",
                            "spreadsheet_id": self.module.launcher.qwen_builder.EXPERIMENT_SPREADSHEET_ID,
                        }
                    ),
                    encoding="utf-8",
                )

            def preserve_summary(
                stage: str,
                rows: list[dict[str, Any]],
                lock: Mapping[str, Any] | None = None,
                *,
                include_historical_budget: bool = False,
            ) -> tuple[dict[str, Any], Path]:
                selected_row = rows[0]
                document: dict[str, Any] = {
                    "schema_version": 1,
                    "campaign": fixture.plan["campaign"],
                    "stages": {
                        stage: {
                            "complete": True,
                            "decision_status": "ready",
                            "control_gate": "passed",
                            "needs_boundary_extension": False,
                            "recommended_experiment": selected_row["experiment"],
                            "recommended_run_id": selected_row["run_id"],
                        }
                    },
                    "runs": rows,
                }
                if lock is not None:
                    document["stage_lock"] = {
                        "lock_payload_sha256": lock["lock_payload_sha256"]
                    }
                if include_historical_budget:
                    document["budget"] = {
                        "unique_kernel_slugs": extra_history_slugs
                    }
                path = report_dir / "stages" / stage / "summary.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(document), encoding="utf-8")
                return document, path

            mark_synced(fixture.baseline_row)
            source_summary, source_summary_path = preserve_summary(
                "lr_log_line",
                [fixture.baseline_row],
                include_historical_budget=True,
            )
            source_stage = "lr_log_line"
            selected_anchor = deepcopy(fixture.baseline_row)
            dropout: dict[str, Any] | None = None
            dropout_path: Path | None = None
            tuned: dict[str, Any] | None = None
            for target_stage, coordinate in (
                ("epoch_line", None),
                ("regularization_coordinate_search", "effective_batch"),
                ("regularization_coordinate_search", "warmup_ratio"),
                ("regularization_coordinate_search", "weight_decay"),
                ("regularization_coordinate_search", "label_smoothing"),
                ("regularization_coordinate_search", "classifier_dropout"),
            ):
                effective_stage = (
                    f"{target_stage}__{coordinate}" if coordinate else target_stage
                )
                lock_path = history_locks_dir / f"{effective_stage}.lock.json"
                stage_lock = self.module.adaptive.axis_materializer.materialize_stage_lock(
                    plan_path=fixture.plan_path,
                    summary_path=source_summary_path,
                    artifacts_dir=fixture.artifacts,
                    source_stage=source_stage,
                    target_stage=target_stage,
                    coordinate=coordinate,
                    output_path=lock_path,
                )
                contract = self.module.builder.normalized_campaign_execution_contract(
                    fixture.plan,
                    stage_lock,
                    base_config=fixture.base_config,
                )
                stage_rows = []
                for entry in contract["variants"]:
                    row = fixture._make_run(
                        experiment=entry["experiment"],
                        slug=entry["kernel_slug"],
                        loss_variant=entry["loss_variant"],
                        config=entry["expected_config"],
                        iid_macro_ap=0.8,
                        notes=entry["expected_notes"],
                        stage=effective_stage,
                        role=entry["role"],
                    )
                    row.update(
                        effective_stage=effective_stage,
                        is_hypothesis=entry["is_hypothesis"],
                    )
                    if effective_stage == dropout_stage:
                        row.update(
                            hard_macro_ap=0.4,
                            ood_macro_ap=0.6,
                            sheets_sync_status="synced",
                        )
                    mark_synced(row)
                    stage_rows.append(row)
                if effective_stage == "regularization_coordinate_search__weight_decay":
                    selected_index = next(
                        index
                        for index, row in enumerate(stage_rows)
                        if row["resolved_config"]["weight_decay"] == 0.05
                    )
                    stage_anchor = deepcopy(stage_rows.pop(selected_index))
                else:
                    stage_anchor = deepcopy(selected_anchor)
                stage_anchor.update(
                    stage=effective_stage,
                    effective_stage=effective_stage,
                    role="stage_anchor",
                    is_hypothesis=False,
                )
                if effective_stage == dropout_stage:
                    stage_anchor.update(
                        hard_macro_ap=0.4,
                        ood_macro_ap=0.6,
                        sheets_sync_status="synced",
                    )
                stage_rows.insert(0, stage_anchor)
                source_summary, source_summary_path = preserve_summary(
                    effective_stage, stage_rows, stage_lock
                )
                selected_anchor = stage_anchor
                source_stage = effective_stage
                if effective_stage == dropout_stage:
                    dropout = stage_lock
                    dropout_path = lock_path
                    tuned = stage_anchor

            assert dropout is not None and dropout_path is not None and tuned is not None
            (report_dir / "summary.json").write_text(
                json.dumps(source_summary), encoding="utf-8"
            )
            policy = self.fast_policy
            freeze = self.fast_freeze
            receipt_path = report_dir / "fast_track" / "max_grad_norm_skip.receipt.json"
            receipt = self.module.fast_track.build_receipt(
                policy_path=self.module.fast_track.DEFAULT_POLICY,
                freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                plan_path=self.module.fast_track.DEFAULT_PLAN,
                summary_path=report_dir / "summary.json",
                artifacts_dir=fixture.artifacts,
                receipt_path=receipt_path,
            )
            self.module.fast_track.write_receipt_once(receipt_path, receipt)
            self.assertEqual(
                self.module.fast_track.validate_receipt(
                    receipt_path,
                    policy=policy,
                    freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                ),
                receipt,
            )
            original_receipt = receipt_path.read_text(encoding="utf-8")
            tampered_receipt = deepcopy(receipt)
            tampered_receipt["freeze_authority"]["manifest_payload_sha256"] = "0" * 64
            tampered_receipt.pop("summary_payload_sha256")
            tampered_receipt["summary_payload_sha256"] = (
                self.module.fast_track.canonical_sha256(tampered_receipt)
            )
            receipt_path.chmod(0o644)
            receipt_path.write_text(
                self.module.fast_track.canonical_json_dumps(tampered_receipt) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                self.module.fast_track.FastTrackError, "detached freeze"
            ):
                self.module.fast_track.validate_receipt(receipt_path, policy=policy)
            receipt_path.write_text(original_receipt, encoding="utf-8")
            receipt_path.chmod(0o444)

            fixture.source_lock = dropout
            fixture.source_lock_path = dropout_path
            fixture.source_stage = dropout_stage
            fixture.tuned_row = tuned
            fixture.source_summary = source_summary
            fixture.source_summary_path = source_summary_path
            primary_path = fixture.locks_dir / "special_loss_screen__primary.lock.json"
            overlay_path = fixture.locks_dir / "special_loss_screen__overlay.lock.json"
            refine_path = fixture.locks_dir / "special_loss_screen__lr_refine.lock.json"

            with self.module.fast_track.patched_loss_predecessor(
                policy_path=self.module.fast_track.DEFAULT_POLICY,
                receipt_path=receipt_path,
                freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
            ):
                primary = self.module.adaptive.materialize_loss_primary_lock(
                    plan=fixture.plan,
                    summary=source_summary,
                    summary_path=source_summary_path,
                    artifacts_dir=fixture.artifacts,
                    prerequisite_locks=[dropout],
                    prerequisite_lock_paths=[dropout_path],
                    history_documents=[receipt],
                    history_document_paths=[receipt_path],
                    output_path=primary_path,
                    source_stage=dropout_stage,
                )
                fixture._remember_trusted_context(primary, primary_path)
                primary_summary, _ = fixture._primary_summary(primary)
                for row in primary_summary["runs"]:
                    mark_synced(row)
                primary_summary_path = fixture._write_bound_summary(
                    primary_summary, "fast-confirm-primary-summary.json"
                )
                overlay = self.module.adaptive.materialize_loss_overlay_lock(
                    plan=fixture.plan,
                    summary=primary_summary,
                    summary_path=primary_summary_path,
                    artifacts_dir=fixture.artifacts,
                    prerequisite_locks=[primary],
                    prerequisite_lock_paths=[primary_path],
                    history_documents=[],
                    history_document_paths=[],
                    output_path=overlay_path,
                )
                fixture._remember_trusted_context(overlay, overlay_path)
                primary_final, _ = fixture._primary_final_summary(
                    primary, overlay, primary_summary
                )
                for row in primary_final["runs"]:
                    mark_synced(row)
                primary_final_path = fixture._write_bound_summary(
                    primary_final, "fast-confirm-overlay-summary.json"
                )
                refine = self.module.adaptive.materialize_loss_lr_refine_lock(
                    plan=fixture.plan,
                    summary=primary_final,
                    summary_path=primary_final_path,
                    artifacts_dir=fixture.artifacts,
                    prerequisite_locks=[primary, overlay],
                    prerequisite_lock_paths=[primary_path, overlay_path],
                    history_documents=[],
                    history_document_paths=[],
                    output_path=refine_path,
                )
                fixture._remember_trusted_context(refine, refine_path)
                loss_final, _ = fixture._loss_final_summary(
                    primary, overlay, refine, primary_final
                )
                for row in loss_final["runs"]:
                    mark_synced(row)
            loss_summary_path = fixture._write_bound_summary(
                loss_final, "fast-confirm-loss-final.json"
            )

            output_dir = fixture.root / "fast-confirm-output"
            confirmation_path = output_dir / self.module.DEFAULT_SOURCE_LOCK_NAME
            with self.module.fast_track.patched_loss_predecessor(
                policy_path=self.module.fast_track.DEFAULT_POLICY,
                receipt_path=receipt_path,
                freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
            ):
                self.module.adaptive.materialize(
                    mode="confirmation",
                    plan_path=fixture.plan_path,
                    summary_path=loss_summary_path,
                    baseline_summary_path=fixture.baseline_summary_path,
                    artifacts_dir=fixture.artifacts,
                    prerequisite_lock_paths=[primary_path, overlay_path, refine_path],
                    output_path=confirmation_path,
                )
                loaded = self.module.builder.load_campaign_lock(
                    confirmation_path, plan=fixture.plan
                )
                selected = self.module.select_fast_pair(
                    plan=fixture.plan,
                    lock=loaded,
                    protocol=self.protocol,
                )
                manifest = self.module.build_manifest(
                    protocol_path=self.protocol_path,
                    protocol=self.protocol,
                    plan_path=fixture.plan_path,
                    lock_path=confirmation_path,
                    lock=loaded,
                    selected=selected,
                    policy_path=self.module.fast_track.DEFAULT_POLICY,
                    policy=policy,
                    receipt_path=receipt_path,
                    receipt=receipt,
                    freeze_manifest_path=self.module.fast_track.DEFAULT_FREEZE_MANIFEST,
                    freeze=freeze,
                )
                self.module._write_once(
                    output_dir / self.module.DEFAULT_MANIFEST_NAME, manifest
                )
                closure = self.module.summarizer.load_adaptive_prerequisite_closure(
                    plan=fixture.plan,
                    root_lock=loaded,
                    root_lock_path=confirmation_path,
                    base_config=fixture.base_config,
                )
                self.assertEqual(closure[-1][0]["mode"], "confirmation")
            projected = manifest["execution_projection"]["runnable_entries"]
            self.assertEqual(len(projected), 2)
            self.assertEqual([entry["seed"] for entry in projected], [17, 17])
            self.assertEqual(
                [entry["experiment"] for entry in projected],
                manifest["launch_order"],
            )
            self.assertFalse(
                manifest["execution_projection"]["runtime_attestation_path_exposed"]
            )
            self.assertFalse(
                manifest["execution_projection"]["ods_submission_path_exposed"]
            )

            self.assertEqual(
                {selected[side]["entry"]["expected_config"]["seed"] for side in selected},
                {17},
            )
            with self.assertRaises(self.module.builder.CampaignConfigError):
                self.module.builder.load_campaign_lock(
                    confirmation_path, plan=fixture.plan
                )
        finally:
            fixture.tearDown()


if __name__ == "__main__":
    unittest.main()
