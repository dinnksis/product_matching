from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import continue_minilm_5ep_sft_campaign as core
import continue_minilm_5ep_sft_loss_fast_track as controller_module
import create_minilm_5ep_sft_hparam_notebooks as generator
import create_minilm_5ep_sft_loss_fast_track_notebooks as create_wrapper
import materialize_minilm_5ep_sft_hparam_stage as axis_materializer
import materialize_minilm_5ep_sft_loss_fast_track as materialize_wrapper
import minilm_5ep_sft_loss_fast_track_support as support
import recover_minilm_5ep_sft_loss_fast_track_short_slugs as recovery
import run_kaggle_notebook as kaggle_runner
import run_minilm_5ep_sft_loss_fast_track_kaggle as launcher_wrapper
import summarize_minilm_5ep_sft_loss_fast_track as summarize_wrapper


def _lock_info(
    path: Path,
    stage: str,
    *,
    schema: int = 1,
    boundary: bool = False,
) -> core.LockInfo:
    payload = {
        "schema_version": schema,
        "effective_stage": stage,
        "transition_kind": "conditional_boundary_extension" if boundary else "stage_transition",
        "lock_payload_sha256": "a" * 64,
    }
    path.write_text("{}\n", encoding="utf-8")
    return core.LockInfo(
        path=path,
        payload=payload,
        schema_version=schema,
        effective_stage=stage,
        payload_sha256="a" * 64,
        execution_status="runnable",
        is_boundary=boundary,
        kernel_slugs=(),
    )


class FastTrackPolicyAndFreezeTest(unittest.TestCase):
    def test_policy_keeps_exact_loss_budget_and_no_ods(self) -> None:
        policy = support.load_policy()
        self.assertEqual(support.policy_sha256(policy), support.POLICY_CANONICAL_SHA256)
        self.assertEqual(
            policy["loss_execution"]["primary_new_loss_variants"],
            list(support.PRIMARY_LOSSES),
        )
        self.assertEqual(policy["budget"]["maximum_new_loss_kernels"], 7)
        self.assertFalse(policy["external_actions"]["ods_submission_allowed"])
        self.assertTrue(policy["external_actions"]["require_exact_sheets_sync"])

    def test_detached_freeze_manifest_pins_sources_and_semantic_probes(self) -> None:
        manifest = support.load_freeze_manifest()
        self.assertEqual(manifest["semantic_contract"], support.FREEZE_SEMANTIC_CONTRACT)
        self.assertEqual(
            set(manifest["reviewed_execution_file_sha256"]),
            set(support.FAST_TRACK_EXECUTION_PATHS),
        )
        original = support.file_sha256

        def drift(path: Path) -> str:
            if path.name == "run_minilm_5ep_sft_loss_fast_track_kaggle.py":
                return "0" * 64
            return original(path)

        with mock.patch.object(support, "file_sha256", side_effect=drift):
            with self.assertRaisesRegex(support.FastTrackError, "drifted"):
                support.load_freeze_manifest()

    def test_policy_and_freeze_paths_are_not_payload_selected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            alternate = Path(directory) / "policy.json"
            alternate.write_text(support.DEFAULT_POLICY.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(support.FastTrackError):
                support.load_policy(alternate)
            alternate_manifest = Path(directory) / "freeze.json"
            alternate_manifest.write_text(support.DEFAULT_FREEZE_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
            with self.assertRaises(support.FastTrackError):
                support.load_freeze_manifest(alternate_manifest)


class FastTrackHistoryTest(unittest.TestCase):
    def test_exact_allowed_lock_graph_rejects_schema2_maxgrad_and_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, lock in (
                ("schema2", _lock_info(root / "s2", support.LOSS_PRIMARY_STAGE, schema=2)),
                ("maxgrad", _lock_info(root / "mg", support.SKIPPED_STAGE)),
                ("dropout_boundary", _lock_info(root / "db", support.SOURCE_STAGE, boundary=True)),
            ):
                with self.subTest(name=name), self.assertRaises(support.FastTrackError):
                    support._validate_allowed_lock_identities((lock,), require_completed_dropout=False)

    def test_completed_receipt_history_requires_every_normal_stage_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            locks = tuple(
                _lock_info(root / f"{index}.json", stage)
                for index, stage in enumerate(support.REQUIRED_NORMAL_LOCK_STAGES)
            )
            support._validate_allowed_lock_identities(locks, require_completed_dropout=True)
            with self.assertRaisesRegex(support.FastTrackError, "exact normal"):
                support._validate_allowed_lock_identities(locks[:-1], require_completed_dropout=True)

    def test_pre_receipt_validator_rejects_future_and_premature_boundary(self) -> None:
        plan = generator.load_plan(support.DEFAULT_PLAN)
        current = "regularization_coordinate_search__label_smoothing"
        decision = {
            "complete": True,
            "decision_status": "ready",
            "control_gate": "passed",
            "needs_boundary_extension": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            future = _lock_info(root / "future.json", support.SOURCE_STAGE)
            later = _lock_info(root / "later.json", support.SKIPPED_STAGE)
            premature = _lock_info(root / "boundary.json", current, boundary=True)
            base = core.Authority(
                plan=plan,
                plan_sha256=support.canonical_sha256(plan),
                stages=core.validate_plan(plan),
                summary={"schema_version": 1},
                summary_sha256=None,
                current_stage=current,
                current_decision=decision,
                locks=(),
                unique_kernel_slugs=(),
            )
            with self.assertRaisesRegex(support.FastTrackError, "max_grad_norm"):
                support.validate_pre_receipt_authority(
                    core.Authority(**{**base.__dict__, "locks": (later,)}),
                    report_dir=root,
                    artifacts_dir=root,
                    require_completed_dropout=False,
                )
            strict_payload = {
                "schema_version": 1,
                "effective_stage": current,
                "transition_kind": "conditional_boundary_extension",
                "lock_payload_sha256": "a" * 64,
            }
            with mock.patch.object(support, "_strict_schema1_lock", return_value=strict_payload), mock.patch.object(
                support, "_validate_history_summaries", return_value=([], [])
            ):
                with self.assertRaisesRegex(support.FastTrackError, "Premature"):
                    support.validate_pre_receipt_authority(
                        core.Authority(**{**base.__dict__, "locks": (premature,)}),
                        report_dir=root,
                        artifacts_dir=root,
                        require_completed_dropout=False,
                    )
            # A completed current stage may have exactly its immediate next normal lock.
            strict_future = {
                "schema_version": 1,
                "effective_stage": support.SOURCE_STAGE,
                "transition_kind": "stage_transition",
                "lock_payload_sha256": "a" * 64,
            }
            with mock.patch.object(support, "_strict_schema1_lock", return_value=strict_future), mock.patch.object(
                support, "_validate_history_summaries", return_value=([], [])
            ):
                support.validate_pre_receipt_authority(
                    core.Authority(**{**base.__dict__, "locks": (future,)}),
                    report_dir=root,
                    artifacts_dir=root,
                    require_completed_dropout=False,
                )

    def test_controller_calls_history_validator_before_receipt_intercept(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            instance = controller_module.LossFastTrackController.__new__(
                controller_module.LossFastTrackController
            )
            root = Path(directory)
            instance.receipt_path = root / "missing.receipt.json"
            instance.paths = core.ControllerPaths(report_dir=root, artifacts_dir=root)
            authority = mock.Mock(current_stage=support.SOURCE_STAGE)
            authority.current_decision = {"complete": True}
            with mock.patch.object(
                support,
                "validate_pre_receipt_authority",
                side_effect=support.FastTrackError("future lock"),
            ):
                with self.assertRaisesRegex(support.FastTrackError, "future lock"):
                    instance.decide(authority)


class FastTrackContextAndScopeTest(unittest.TestCase):
    def _real_completion(self) -> dict:
        return json.loads(
            (
                support.DEFAULT_ARTIFACTS_DIR
                / "pm-minilm5-sft-e3-lr8e5-v1"
                / "notebook_completed.json"
            ).read_text(encoding="utf-8")
        )

    def test_real_training_contract_adapter_is_exact_and_restored(self) -> None:
        original_validator = support.adaptive._validate_frozen_training_contract
        original_template = support.adaptive._frozen_training_contract_template
        completion = self._real_completion()
        with support._patched_real_training_contract_validator():
            self.assertIsNot(
                support.adaptive._validate_frozen_training_contract,
                original_validator,
            )
            self.assertEqual(
                support.adaptive._validate_frozen_training_contract(
                    completion, experiment=completion["experiment"]
                ),
                support.FROZEN_CORE_TRAINING_CONTRACT,
            )
            with self.assertRaisesRegex(support.FastTrackError, "not re-entrant"):
                with support._patched_real_training_contract_validator():
                    pass
        self.assertIs(
            support.adaptive._validate_frozen_training_contract,
            original_validator,
        )
        self.assertIs(
            support.adaptive._frozen_training_contract_template,
            original_template,
        )

        with self.assertRaisesRegex(RuntimeError, "probe"):
            with support._patched_real_training_contract_validator():
                raise RuntimeError("probe")
        self.assertIs(
            support.adaptive._validate_frozen_training_contract,
            original_validator,
        )

    def test_real_training_contract_adapter_rejects_rehashed_semantic_drift(self) -> None:
        completion = self._real_completion()

        def change_train_hash(value: dict) -> None:
            value["train_data"]["train_pairs_sha256"] = "0" * 64

        def omit_prevalence(value: dict) -> None:
            value["train_data"].pop("positive_rate")

        def add_train_field(value: dict) -> None:
            value["train_data"]["extra"] = "forbidden"

        def change_label_counts(value: dict) -> None:
            value["train_data"]["label_source_counts"] = {"human": 306_669}

        def change_sampling(value: dict) -> None:
            value["training_report"]["training_sampling"] = "weighted"

        def change_weight(value: dict) -> None:
            value["training_report"]["training_loss_weight_max"] = 1.0001

        for mutate in (
            change_train_hash,
            omit_prevalence,
            add_train_field,
            change_label_counts,
            change_sampling,
            change_weight,
        ):
            with self.subTest(mutate=mutate):
                changed = deepcopy(completion)
                mutate(changed)
                with support._patched_real_training_contract_validator(), self.assertRaises(
                    support.adaptive.AdaptiveMaterializationError
                ):
                    support.adaptive._validate_frozen_training_contract(
                        changed, experiment=changed["experiment"]
                    )

    def test_short_remote_identity_is_canonical_bounded_and_loss_only(self) -> None:
        config = generator.cross_builder.load_training_config(generator.BASE_CONFIG_PATH)
        original = support.adaptive._variant_identity
        observed: set[str] = set()
        cases = (
            ("loss_primary", "balanced_binary_bce"),
            ("loss_primary", "balanced_category_class_sqrt_bce"),
            ("loss_primary", "balanced_category_class_bce"),
            ("loss_primary", "focal_bce_gamma2_scale4"),
            ("loss_overlay", "balanced_binary_focal_gamma2_scale4"),
            ("loss_overlay", "balanced_category_class_focal_gamma2_scale4"),
            ("loss_lr_refine", "balanced_binary_bce"),
        )
        original_results = {
            case: original(mode=case[0], loss_variant=case[1], config=config)
            for case in cases
        }
        confirmation = original(
            mode="confirmation", loss_variant="balanced_binary_bce", config=config
        )
        with support._patched_short_remote_identity():
            self.assertIsNot(support.adaptive._variant_identity, original)
            for mode, loss_variant in cases:
                with self.subTest(mode=mode, loss_variant=loss_variant):
                    result = support.adaptive._variant_identity(
                        mode=mode, loss_variant=loss_variant, config=config
                    )
                    baseline = original_results[(mode, loss_variant)]
                    self.assertEqual(result[0], baseline[0])
                    self.assertEqual(result[2:], baseline[2:])
                    self.assertLessEqual(
                        len(result[1]), support.KAGGLE_REMOTE_IDENTITY_MAX_LENGTH
                    )
                    self.assertRegex(result[1], r"^[a-z0-9][a-z0-9-]*$")
                    self.assertNotIn(result[1], observed)
                    observed.add(result[1])
            self.assertEqual(
                support.adaptive._variant_identity(
                    mode="confirmation",
                    loss_variant="balanced_binary_bce",
                    config=config,
                ),
                confirmation,
            )
            with self.assertRaisesRegex(support.FastTrackError, "not re-entrant"):
                with support._patched_short_remote_identity():
                    pass
        self.assertIs(support.adaptive._variant_identity, original)

        with self.assertRaisesRegex(RuntimeError, "identity probe"):
            with support._patched_short_remote_identity():
                raise RuntimeError("identity probe")
        self.assertIs(support.adaptive._variant_identity, original)

    def test_exact_long_kaggle_identity_is_rejected_before_remote_use(self) -> None:
        config = support.load_json(
            support.DEFAULT_RECEIPT, label="actual fast-track receipt"
        )["selected_parent"]["resolved_config"]
        _, long_slug, _, family_sha = support._ORIGINAL_VARIANT_IDENTITY(
            mode="loss_primary",
            loss_variant="balanced_binary_bce",
            config=config,
        )
        self.assertEqual(
            long_slug,
            "pm-minilm5-sft-loss-balanced-binary-bce-ff031cb1-s42-v1",
        )
        self.assertEqual(len(long_slug), 55)
        with self.assertRaisesRegex(support.FastTrackError, "<=50-character"):
            support._validate_short_remote_variant(
                {"kernel_slug": long_slug, "title": long_slug},
                label="reviewed long identity",
            )
        short_slug = support._short_loss_remote_slug(
            mode="loss_primary", family_sha=family_sha, seed=42
        )
        self.assertEqual(
            short_slug,
            "pm-m5-lp-ff031cb18cc2c235d9aa3393-s42-v1",
        )
        self.assertEqual(len(short_slug), 40)

    def test_short_remote_lock_rejects_title_drift_and_collision(self) -> None:
        family_a = "1" * 64
        family_b = "2" * 64
        slug_a = support._short_loss_remote_slug(
            mode="loss_primary", family_sha=family_a, seed=42
        )
        slug_b = support._short_loss_remote_slug(
            mode="loss_primary", family_sha=family_b, seed=42
        )
        lock = {
            "mode": "loss_primary",
            "resolved_stage": {
                "variants": [
                    {
                        "kernel_slug": slug_a,
                        "title": slug_a,
                        "seed": 42,
                        "expected_recipe_family_sha256": family_a,
                    },
                    {
                        "kernel_slug": slug_b,
                        "title": slug_b,
                        "seed": 42,
                        "expected_recipe_family_sha256": family_b,
                    },
                ]
            },
        }
        support._validate_short_remote_lock(lock)
        changed = deepcopy(lock)
        changed["resolved_stage"]["variants"][0]["title"] = "another-title"
        with self.assertRaises(support.FastTrackError):
            support._validate_short_remote_lock(changed)
        changed = deepcopy(lock)
        changed["resolved_stage"]["variants"][1].update(
            kernel_slug=slug_a,
            title=slug_a,
            expected_recipe_family_sha256=family_a,
        )
        with self.assertRaisesRegex(support.FastTrackError, "collide"):
            support._validate_short_remote_lock(changed)

    def test_scoped_predecessor_requires_freeze_and_restores(self) -> None:
        original = axis_materializer.expected_source_stage
        original_validator = support.adaptive._validate_frozen_training_contract
        original_identity = support.adaptive._variant_identity
        plan = generator.load_plan(support.DEFAULT_PLAN)
        with mock.patch.object(support, "load_freeze_manifest", return_value={}), mock.patch.object(
            support, "validate_receipt", return_value={}
        ):
            with support.patched_loss_predecessor():
                self.assertIsNot(support.adaptive._variant_identity, original_identity)
                self.assertEqual(
                    axis_materializer.expected_source_stage(
                        plan, target_stage="special_loss_screen", coordinate=None
                    ),
                    support.SOURCE_STAGE,
                )
                self.assertEqual(
                    axis_materializer.expected_source_stage(
                        plan, target_stage="epoch_line", coordinate=None
                    ),
                    "lr_log_line",
                )
                with self.assertRaises(support.FastTrackError):
                    with support.patched_loss_predecessor():
                        pass
        self.assertIs(axis_materializer.expected_source_stage, original)
        self.assertIs(
            support.adaptive._validate_frozen_training_contract,
            original_validator,
        )
        self.assertIs(support.adaptive._variant_identity, original_identity)

    def test_scoped_lock_loader_allows_only_three_schema2_loss_modes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock_path = Path(directory) / "lock.json"
            lock_path.write_text("{}\n", encoding="utf-8")
            for mode, stage in support.LOSS_STAGE_BY_MODE.items():
                payload = {
                    "schema_version": 2,
                    "mode": mode,
                    "effective_stage": stage,
                    "execution_status": "skipped",
                    "resolved_stage": {"variants": []},
                }
                with self.subTest(mode=mode), mock.patch.object(
                    support, "patched_loss_predecessor", return_value=nullcontext()
                ), mock.patch.object(generator, "load_campaign_lock", return_value=payload):
                    self.assertEqual(
                        support.load_scoped_loss_lock(
                            plan_path=support.DEFAULT_PLAN, stage_lock_path=lock_path
                        ),
                        payload,
                    )
            for payload in (
                {"schema_version": 1, "mode": None, "effective_stage": "epoch_line", "execution_status": "runnable"},
                {"schema_version": 2, "mode": "confirmation", "effective_stage": "confirmation__matched_seeds", "execution_status": "runnable"},
                {"schema_version": 2, "mode": "loss_primary", "effective_stage": "special_loss_screen__overlay", "execution_status": "runnable"},
            ):
                with mock.patch.object(support, "patched_loss_predecessor", return_value=nullcontext()), mock.patch.object(
                    generator, "load_campaign_lock", return_value=payload
                ), self.assertRaises(support.FastTrackError):
                    support.load_scoped_loss_lock(
                        plan_path=support.DEFAULT_PLAN, stage_lock_path=lock_path
                    )

    def test_core_predecessor_outside_context_remains_maxgrad(self) -> None:
        plan = generator.load_plan(support.DEFAULT_PLAN)
        self.assertEqual(
            axis_materializer.expected_source_stage(
                plan, target_stage="special_loss_screen", coordinate=None
            ),
            support.SKIPPED_STAGE,
        )


class ActualWorkspaceReceiptSmokeTest(unittest.TestCase):
    def test_materializes_and_revalidates_real_dropout_receipt_in_temp_output(self) -> None:
        root_summary = support.load_json(
            support.DEFAULT_SUMMARY, label="actual workspace root summary"
        )
        stages = root_summary.get("stages")
        if (
            root_summary.get("schema_version") != 1
            or not isinstance(stages, dict)
            or set(stages) != {support.SOURCE_STAGE}
        ):
            validated = support.validate_receipt()
            self.assertEqual(validated["budget"]["unique_kernels"], 18)
            return
        schema2_locks = list(
            (support.DEFAULT_REPORT_DIR / "stage_locks").glob("special_loss*.lock.json")
        )
        if schema2_locks:
            validated = support.validate_receipt()
            self.assertEqual(validated["budget"]["unique_kernels"], 18)
            self.assertEqual(validated["skipped_coordinate"]["new_kernels"], 0)
            return

        with tempfile.TemporaryDirectory() as directory:
            receipt_path = Path(directory) / "skip.receipt.json"
            payload = support.build_receipt(receipt_path=receipt_path)
            written = support.write_receipt_once(receipt_path, payload)
            validated = support.validate_receipt(receipt_path)
        self.assertEqual(written, validated)
        self.assertEqual(
            validated["selected_parent"]["experiment"],
            "minilm5_sft_e3_lr8e5_v1",
        )
        self.assertEqual(validated["budget"]["unique_kernels"], 18)
        self.assertEqual(validated["skipped_coordinate"]["new_kernels"], 0)


class ShortRemoteIdentityIntegrationTest(unittest.TestCase):
    def test_existing_receipt_keeps_exact_legacy_freeze_and_prior18(self) -> None:
        receipt = support.validate_receipt()
        self.assertEqual(
            receipt["freeze_authority"]["file_sha256"],
            support.LEGACY_RECEIPT_FREEZE_FILE_SHA256,
        )
        self.assertEqual(
            receipt["freeze_authority"]["manifest_payload_sha256"],
            support.LEGACY_RECEIPT_FREEZE_PAYLOAD_SHA256,
        )
        self.assertNotEqual(
            support.file_sha256(support.DEFAULT_FREEZE_MANIFEST),
            support.LEGACY_RECEIPT_FREEZE_FILE_SHA256,
        )
        self.assertEqual(receipt["budget"]["unique_kernels"], 18)

    def test_actual_recovery_evidence_has_local_read_only_preflight(self) -> None:
        if recovery.RECOVERY_COMPLETION.exists():
            completed = recovery.validate_completed_recovery()
            self.assertEqual(completed["kaggle_mutations"], 0)
            return
        payload = recovery.build_preflight()
        self.assertEqual(payload["prior_unique_kernels"], 18)
        self.assertEqual(payload["kaggle_mutations"], 0)
        self.assertEqual(payload["source_lock_payload_sha256"], recovery.BAD_LOCK_PAYLOAD_SHA256)
        roles = [row["role"] for row in payload["moves"]]
        self.assertEqual(roles.count("source_lock"), 1)
        self.assertEqual(
            roles.count("trusted_provenance_manifest_forensic_original"), 1
        )
        self.assertEqual(roles.count("trusted_provenance_archive"), 1)
        self.assertEqual(roles.count("failed_controller_state"), 1)
        self.assertEqual(roles.count("generated_notebook"), 4)
        self.assertEqual(roles.count("kaggle_staging"), 4)
        self.assertFalse(recovery.RECOVERY_ARCHIVE.exists())

    def test_rehashed_remote_audit_drift_and_long_lock_are_rejected(self) -> None:
        receipt = support.load_json(
            recovery.AUDIT_RECEIPT, label="reviewed remote absence audit"
        )
        changed = deepcopy(receipt)
        changed["assertions"][
            "all_remote_kernel_identities_absent_from_authenticated_owner_scope"
        ] = False
        changed.pop("receipt_payload_sha256")
        changed["receipt_payload_sha256"] = support.canonical_sha256(changed)
        original_loader = recovery._load_single_link_json

        def load_tampered_audit(path: Path, *, label: str) -> dict[str, object]:
            if path == recovery.AUDIT_RECEIPT:
                return changed
            return original_loader(path, label=label)

        with mock.patch.object(
            recovery, "_load_single_link_json", side_effect=load_tampered_audit
        ), self.assertRaisesRegex(recovery.RecoveryError, "identity/hash"):
            recovery.validate_audit_receipt()

        if (
            recovery.BAD_LOCK.is_file()
            and support.file_sha256(recovery.BAD_LOCK)
            == recovery.BAD_LOCK_FILE_SHA256
        ):
            with self.assertRaisesRegex(support.FastTrackError, "identity"):
                support.load_scoped_loss_lock(
                    plan_path=support.DEFAULT_PLAN,
                    stage_lock_path=recovery.BAD_LOCK,
                )

    def test_recovery_move_is_exact_and_recoverable_in_temp_tree(self) -> None:
        reports_dir = support.ROOT / "reports"
        with tempfile.TemporaryDirectory(dir=reports_dir) as directory:
            root = Path(directory)
            source_file = root / "source.lock.json"
            source_file.write_text("frozen\n", encoding="utf-8")
            source_dir = root / "sidecar"
            source_dir.mkdir()
            (source_dir / "snapshot.json").write_text("{}\n", encoding="utf-8")
            target_file = root / "archive" / "source.lock.json"
            target_dir = root / "archive" / "sidecar"
            items = []
            for index, (role, source, target, kind) in enumerate(
                (
                    ("source_lock", source_file, target_file, "file"),
                    (
                        "trusted_provenance_archive",
                        source_dir,
                        target_dir,
                        "directory",
                    ),
                )
            ):
                item = {
                    "index": index,
                    "role": role,
                    "source": str(source.relative_to(support.ROOT)),
                    "target": str(target.relative_to(support.ROOT)),
                    "kind": kind,
                    "file_sha256": support.file_sha256(source)
                    if kind == "file"
                    else None,
                    "tree_file_sha256s": recovery._tree_hashes(source)
                    if kind == "directory"
                    else None,
                }
                item["move_payload_sha256"] = support.canonical_sha256(item)
                items.append(item)
            with mock.patch.object(
                recovery, "RECOVERY_JOURNAL", root / "archive" / "journal"
            ):
                for item in items:
                    recovery._reconcile_one_move(
                        item, preflight_sha="a" * 64, fault_hook=None
                    )
                # Exact replay is a no-op and revalidates the terminal state.
                for item in items:
                    recovery._reconcile_one_move(
                        item, preflight_sha="a" * 64, fault_hook=None
                    )
            self.assertFalse(source_file.exists())
            self.assertFalse(source_dir.exists())
            self.assertEqual(target_file.read_text(encoding="utf-8"), "frozen\n")
            self.assertEqual(
                recovery._tree_hashes(target_dir),
                items[1]["tree_file_sha256s"],
            )

    def test_real_primary_rematerializes_short_metadata_and_reloads_for_resume(self) -> None:
        old_lock_path = recovery.BAD_LOCK
        if not old_lock_path.is_file() or support.file_sha256(old_lock_path) != recovery.BAD_LOCK_FILE_SHA256:
            old_lock_path = recovery.RECOVERY_ARCHIVE / "authority" / recovery.BAD_LOCK.name
        old_lock = support.load_json(old_lock_path, label="rejected long primary lock")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "special_loss_screen__primary.lock.json"
            argv = [
                str(materialize_wrapper.__file__),
                "loss_primary",
                "--prerequisite-lock",
                str(
                    support.DEFAULT_REPORT_DIR
                    / "stage_locks"
                    / "regularization_coordinate_search_classifier_dropout.lock.json"
                ),
                "--source-stage",
                support.SOURCE_STAGE,
                "--history-summary",
                str(support.DEFAULT_RECEIPT),
                "--output",
                str(lock_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                materialize_wrapper.main()
            lock = support.load_scoped_loss_lock(
                plan_path=support.DEFAULT_PLAN,
                stage_lock_path=lock_path,
            )
            support._validate_short_remote_lock(lock)
            old_variants = old_lock["resolved_stage"]["variants"]
            new_variants = lock["resolved_stage"]["variants"]
            self.assertEqual(
                [row["experiment"] for row in new_variants],
                [row["experiment"] for row in old_variants],
            )
            self.assertEqual(
                [row["expected_recipe_sha256"] for row in new_variants],
                [row["expected_recipe_sha256"] for row in old_variants],
            )
            self.assertEqual(
                [row["expected_recipe_family_sha256"] for row in new_variants],
                [row["expected_recipe_family_sha256"] for row in old_variants],
            )

            notebook_dir = root / "notebooks"
            with support.patched_loss_predecessor():
                built = generator.build_campaign(
                    plan_path=support.DEFAULT_PLAN,
                    output_dir=notebook_dir,
                    stage_lock_path=lock_path,
                )
                resumable = launcher_wrapper.core.campaign_variants(
                    generator.load_plan(support.DEFAULT_PLAN),
                    stage=None,
                    only=None,
                    stage_lock=lock,
                )
            self.assertEqual(len(built), 4)
            self.assertEqual(
                [row["kernel_slug"] for row in resumable],
                [row["kernel_slug"] for row in built],
            )

            staging_root = root / "staging"
            for entry in built:
                with self.subTest(experiment=entry["experiment"]), mock.patch.object(
                    kaggle_runner, "STAGE_ROOT", staging_root
                ), mock.patch.dict(
                    "os.environ", {"KAGGLE_USERNAME": "shortslugtest"}, clear=False
                ), mock.patch.object(
                    sys,
                    "argv",
                    [
                        str(kaggle_runner.__file__),
                        entry["notebook"],
                        "--slug",
                        entry["kernel_slug"],
                        "--title",
                        entry["title"],
                        "--cpu",
                        "--dry-run",
                        "--no-env-sources",
                        "--no-google-sheets-credentials",
                    ],
                ):
                    self.assertEqual(kaggle_runner.main(), 0)
                metadata = support.load_json(
                    staging_root / entry["kernel_slug"] / "kernel-metadata.json",
                    label="short Kaggle dry-run metadata",
                )
                self.assertEqual(metadata["title"], entry["kernel_slug"])
                self.assertEqual(
                    metadata["id"], f"shortslugtest/{entry['kernel_slug']}"
                )
                self.assertLessEqual(len(metadata["title"]), 50)
                self.assertLessEqual(len(entry["kernel_slug"]), 50)


class RecoveryTransactionCrashTest(unittest.TestCase):
    class InjectedCrash(RuntimeError):
        pass

    def _base_preflight(self, freeze: dict) -> dict:
        with mock.patch.object(
            support, "load_freeze_manifest", return_value=freeze
        ):
            if recovery.RECOVERY_PREFLIGHT.exists():
                return support.load_json(
                    recovery.RECOVERY_PREFLIGHT,
                    label="existing real recovery preflight",
                )
            return recovery.build_preflight()

    def _exact_source_for_move(self, item: dict) -> Path:
        for key in ("source", "target"):
            path = support.ROOT / item[key]
            try:
                if recovery._artifact_matches(path, item):
                    return path
            except recovery.RecoveryError:
                continue
        self.fail(f"No exact real source/target remains for move {item['index']}")

    def _fixture(
        self,
        root: Path,
        *,
        base_preflight: dict,
        freeze: dict,
    ) -> tuple[ExitStack, dict, dict[str, Path]]:
        archive = root / "archive"
        relocated_lock = archive / "authority" / recovery.BAD_LOCK.name
        relocated_provenance = Path(
            str(relocated_lock) + ".trusted-provenance.json"
        )
        relocated_archive = Path(str(relocated_lock) + ".trusted-provenance")
        forensic_manifest = (
            archive
            / "forensic"
            / f"{recovery.BAD_LOCK.name}.trusted-provenance.original.json"
        )
        sources: list[Path] = []
        targets: list[Path] = []
        roster = []
        for item in base_preflight["moves"]:
            original = self._exact_source_for_move(item)
            source = root / "sources" / f"{item['index']:02d}" / original.name
            source.parent.mkdir(parents=True, exist_ok=True)
            if original.is_dir():
                shutil.copytree(original, source)
            else:
                shutil.copy2(original, source)
            role = item["role"]
            if role == "source_lock":
                target = relocated_lock
            elif role == "trusted_provenance_manifest_forensic_original":
                target = forensic_manifest
            elif role == "trusted_provenance_archive":
                target = relocated_archive
            elif role == "failed_controller_state":
                target = archive / "evidence" / source.name
            elif role == "generated_notebook":
                target = archive / "notebooks" / source.name
            elif role == "kaggle_staging":
                target = archive / "staging" / source.name
            else:  # pragma: no cover - guarded by the frozen real preflight.
                self.fail(f"Unexpected move role {role}")
            sources.append(source)
            targets.append(target)
            roster.append((role, source, target, item["kind"]))
        paths = {
            "archive": archive,
            "preflight": archive / "preflight.json",
            "completion": archive / "completion.json",
            "journal": archive / "journal",
            "audit_copy": archive / "evidence" / recovery.AUDIT_RECEIPT.name,
            "forensic_manifest": forensic_manifest,
            "relocated_lock": relocated_lock,
            "relocated_provenance": relocated_provenance,
            "relocated_archive": relocated_archive,
            "bad_lock": sources[0],
            "controller": sources[3],
            "mutex": root / ".recovery.lock",
        }
        preflight = deepcopy(base_preflight)
        preflight["recovery_archive"] = recovery._relative(archive)
        preflight["journal_directory"] = recovery._relative(paths["journal"])
        rewritten_moves = []
        for index, (role, source, target, kind) in enumerate(roster):
            row = {
                "index": index,
                "role": role,
                "source": recovery._relative(source),
                "target": recovery._relative(target),
                "kind": kind,
                "file_sha256": support.file_sha256(source)
                if kind == "file"
                else None,
                "tree_file_sha256s": recovery._tree_hashes(source)
                if kind == "directory"
                else None,
            }
            row["move_payload_sha256"] = support.canonical_sha256(row)
            rewritten_moves.append(row)
        preflight["moves"] = rewritten_moves
        preflight.pop("preflight_payload_sha256", None)
        preflight["preflight_payload_sha256"] = support.canonical_sha256(
            preflight
        )
        patches = ExitStack()
        for name, value in (
            ("RECOVERY_ARCHIVE", archive),
            ("RECOVERY_PREFLIGHT", paths["preflight"]),
            ("RECOVERY_COMPLETION", paths["completion"]),
            ("RECOVERY_JOURNAL", paths["journal"]),
            ("RECOVERY_MUTEX", paths["mutex"]),
            ("RECOVERY_AUDIT_COPY", paths["audit_copy"]),
            ("RECOVERY_FORENSIC_MANIFEST", forensic_manifest),
            ("RECOVERY_RELOCATED_LOCK", relocated_lock),
            ("RECOVERY_RELOCATED_PROVENANCE", relocated_provenance),
            ("RECOVERY_RELOCATED_ARCHIVE", relocated_archive),
            ("BAD_LOCK", sources[0]),
            ("FAILED_CONTROLLER_STATE", sources[3]),
        ):
            patches.enter_context(mock.patch.object(recovery, name, value))
        patches.enter_context(
            mock.patch.object(
                recovery, "_expected_move_paths", return_value=roster
            )
        )
        patches.enter_context(
            mock.patch.object(
                support, "load_freeze_manifest", return_value=freeze
            )
        )
        return patches, preflight, paths

    def test_fault_injection_every_move_and_major_crash_window_is_resumable(
        self,
    ) -> None:
        freeze = support.load_json(
            support.DEFAULT_FREEZE_MANIFEST, label="current fast freeze"
        )
        base = self._base_preflight(freeze)
        fault_points = [
            "after_archive_mkdir",
            "recovery_preflight:partial_written",
            "after_preflight",
            "audit_copy:partial_written",
            *[
                f"move_{index:02d}:after_rename"
                for index in range(len(base["moves"]))
            ],
            "after_archive_complete",
            "relocated_old_provenance_manifest:partial_written",
            "before_rematerialize",
            "materialization_provenance_manifest:partial_written",
            "materialization_primary_lock:partial_written",
            "after_rematerialize",
            "after_materialization_journal",
            "before_completion",
            "recovery_completion:partial_written",
            "after_completion",
        ]
        reports = support.ROOT / "reports"
        for fault_point in fault_points:
            with self.subTest(fault_point=fault_point), tempfile.TemporaryDirectory(
                dir=reports
            ) as directory:
                root = Path(directory)
                patches, preflight, paths = self._fixture(
                    root, base_preflight=base, freeze=freeze
                )
                with patches:
                    fired = False

                    def inject(point: str) -> None:
                        nonlocal fired
                        if point == fault_point and not fired:
                            fired = True
                            raise self.InjectedCrash(point)

                    with self.assertRaisesRegex(
                        self.InjectedCrash, fault_point
                    ):
                        recovery.apply_recovery(preflight, fault_hook=inject)
                    self.assertTrue(fired)
                    completed = recovery.apply_recovery(preflight)
                    replayed = recovery.apply_recovery(preflight)
                    self.assertEqual(replayed, completed)
                    self.assertEqual(completed["kaggle_mutations"], 0)
                    self.assertTrue(paths["completion"].is_file())
                    for item in preflight["moves"]:
                        self.assertTrue(
                            recovery._artifact_matches(
                                support.ROOT / item["target"], item
                            )
                        )

    def test_every_pending_complete_fault_is_resumable_and_replayable(
        self,
    ) -> None:
        freeze = support.load_json(
            support.DEFAULT_FREEZE_MANIFEST, label="current fast freeze"
        )
        base = self._base_preflight(freeze)
        reports = support.ROOT / "reports"
        observed: list[str] = []
        with tempfile.TemporaryDirectory(dir=reports) as directory:
            patches, preflight, _ = self._fixture(
                Path(directory), base_preflight=base, freeze=freeze
            )
            with patches:
                recovery.apply_recovery(preflight, fault_hook=observed.append)
        fault_points = [
            point for point in observed if point.endswith(":pending_complete")
        ]
        self.assertEqual(len(fault_points), 25)
        self.assertEqual(len(set(fault_points)), 25)
        self.assertEqual(
            {point.rsplit(":", 1)[0] for point in fault_points},
            {
                "recovery_preflight",
                "audit_copy",
                "initialization.complete_journal",
                *{f"move_{index:02d}_journal" for index in range(12)},
                "archive.complete_journal",
                "relocated_old_provenance_manifest",
                "relocated_old_authority.complete_journal",
                "materialization.intent_journal",
                *{
                    f"materialization_snapshot_{document_sha}"
                    for document_sha in (
                        "31b314e877b096130a31b25e38c0637427db9397d36648ffa4ad2e43008aaa3a",
                        "4458e389a111c4aa09dd7fa211f289f7c7dab38910b43cac7fd98c84b0f61986",
                    )
                },
                "materialization_provenance_manifest",
                "materialization_primary_lock",
                "materialization.complete_journal",
                "recovery_completion",
            },
        )
        for fault_point in fault_points:
            with self.subTest(fault_point=fault_point), tempfile.TemporaryDirectory(
                dir=reports
            ) as directory:
                patches, preflight, paths = self._fixture(
                    Path(directory), base_preflight=base, freeze=freeze
                )
                with patches:
                    fired = False

                    def inject(point: str) -> None:
                        nonlocal fired
                        if point == fault_point and not fired:
                            fired = True
                            raise self.InjectedCrash(point)

                    with self.assertRaisesRegex(
                        self.InjectedCrash, fault_point
                    ):
                        recovery.apply_recovery(preflight, fault_hook=inject)
                    self.assertTrue(fired)
                    completed = recovery.apply_recovery(preflight)
                    replayed = recovery.apply_recovery(preflight)
                    self.assertEqual(replayed, completed)
                    self.assertEqual(completed["kaggle_mutations"], 0)
                    self.assertTrue(paths["completion"].is_file())

    def test_read_only_complete_pending_installs_without_rdwr_reopen(self) -> None:
        reports = support.ROOT / "reports"
        with tempfile.TemporaryDirectory(dir=reports) as directory:
            target = Path(directory) / "immutable.json"
            pending = recovery._pending_path(target)
            payload = b'{"complete":true}\n'

            def crash_after_chmod(point: str) -> None:
                if point == "read_only_resume:pending_complete":
                    raise self.InjectedCrash(point)

            with self.assertRaisesRegex(
                self.InjectedCrash, "pending_complete"
            ):
                recovery._atomic_write_bytes_once(
                    target,
                    payload,
                    mode=0o444,
                    label="read_only_resume",
                    fault_hook=crash_after_chmod,
                )
            self.assertFalse(target.exists())
            self.assertEqual(pending.read_bytes(), payload)
            self.assertEqual(stat.S_IMODE(pending.stat().st_mode), 0o444)

            original_open = recovery._open_verified_single_link_file
            pending_open_flags: list[int] = []

            def record_open(
                path: Path, *, flags: int, label: str
            ) -> tuple[int, int, tuple[int, int], tuple[int, int, int]]:
                if path == pending:
                    pending_open_flags.append(flags)
                return original_open(path, flags=flags, label=label)

            with mock.patch.object(
                recovery,
                "_open_verified_single_link_file",
                side_effect=record_open,
            ):
                recovery._atomic_write_bytes_once(
                    target,
                    payload,
                    mode=0o444,
                    label="read_only_resume",
                )
            self.assertTrue(pending_open_flags)
            self.assertTrue(
                all(flags == os.O_RDONLY for flags in pending_open_flags)
            )
            self.assertFalse(pending.exists())
            self.assertEqual(target.read_bytes(), payload)
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o444)
            recovery._atomic_write_bytes_once(
                target,
                payload,
                mode=0o444,
                label="read_only_resume",
            )

    def test_old_forensic_manifest_is_not_used_as_active_authority(self) -> None:
        freeze = support.load_json(
            support.DEFAULT_FREEZE_MANIFEST, label="current fast freeze"
        )
        base = self._base_preflight(freeze)
        reports = support.ROOT / "reports"
        with tempfile.TemporaryDirectory(dir=reports) as directory:
            patches, preflight, paths = self._fixture(
                Path(directory), base_preflight=base, freeze=freeze
            )
            with patches:
                completion = recovery.apply_recovery(preflight)
                forensic = support.load_json(
                    paths["forensic_manifest"], label="forensic original"
                )
                relocated = support.load_json(
                    paths["relocated_provenance"], label="relocated authority"
                )
                self.assertEqual(
                    Path(forensic["archive_dir"]).resolve(strict=False),
                    recovery.REJECTED_ORIGINAL_PROVENANCE_ARCHIVE.resolve(
                        strict=False
                    ),
                )
                self.assertEqual(
                    Path(relocated["archive_dir"]).resolve(strict=True),
                    paths["relocated_archive"].resolve(strict=True),
                )
                self.assertFalse(
                    completion["archived_old_authority"][
                        "active_sidecar_dependency"
                    ]
                )
                active_manifest = Path(
                    str(paths["bad_lock"]) + ".trusted-provenance.json"
                )
                held = active_manifest.with_name(active_manifest.name + ".held")
                active_manifest.rename(held)
                try:
                    evidence, old_lock = recovery._relocated_old_authority()
                finally:
                    held.rename(active_manifest)
                self.assertFalse(evidence["active_sidecar_dependency"])
                self.assertEqual(
                    old_lock["lock_payload_sha256"],
                    recovery.BAD_LOCK_PAYLOAD_SHA256,
                )

    def test_target_race_never_overwrites_either_file(self) -> None:
        reports = support.ROOT / "reports"
        with tempfile.TemporaryDirectory(dir=reports) as directory:
            root = Path(directory)
            source = root / "source.json"
            target = root / "archive" / "target.json"
            source.write_text("source\n", encoding="utf-8")
            item = {
                "index": 0,
                "role": "source_lock",
                "source": recovery._relative(source),
                "target": recovery._relative(target),
                "kind": "file",
                "file_sha256": support.file_sha256(source),
                "tree_file_sha256s": None,
            }
            item["move_payload_sha256"] = support.canonical_sha256(item)

            def race(point: str) -> None:
                if point == "move_00:before_rename":
                    target.write_text("racer\n", encoding="utf-8")

            with mock.patch.object(
                recovery, "RECOVERY_JOURNAL", root / "journal"
            ), self.assertRaisesRegex(recovery.RecoveryError, "target"):
                recovery._reconcile_one_move(
                    item, preflight_sha="a" * 64, fault_hook=race
                )
            self.assertEqual(source.read_text(encoding="utf-8"), "source\n")
            self.assertEqual(target.read_text(encoding="utf-8"), "racer\n")

    def test_atomic_materializer_patch_restores_on_exception(self) -> None:
        adaptive = support.adaptive
        originals = (
            adaptive._write_source_snapshot_once,
            adaptive._write_trusted_provenance_once,
            adaptive._write_once,
        )
        with self.assertRaisesRegex(RuntimeError, "atomic patch probe"):
            with recovery._patched_atomic_materialization_writes():
                self.assertIsNot(adaptive._write_source_snapshot_once, originals[0])
                self.assertIsNot(
                    adaptive._write_trusted_provenance_once, originals[1]
                )
                self.assertIsNot(adaptive._write_once, originals[2])
                raise RuntimeError("atomic patch probe")
        self.assertEqual(
            (
                adaptive._write_source_snapshot_once,
                adaptive._write_trusted_provenance_once,
                adaptive._write_once,
            ),
            originals,
        )

    def test_pending_and_final_hardlinks_preserve_victim_and_fail_closed(self) -> None:
        reports = support.ROOT / "reports"
        payload = b'{"safe":true}\n'
        for kind, initial in (
            ("empty_pending", b""),
            ("prefix_pending", payload[:7]),
            ("exact_final", payload),
        ):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory(
                dir=reports
            ) as directory:
                root = Path(directory)
                target = root / "authority.json"
                victim = root / "victim.bin"
                victim.write_bytes(initial)
                victim.chmod(0o640)
                attacked = (
                    target
                    if kind == "exact_final"
                    else recovery._pending_path(target)
                )
                os.link(victim, attacked)
                before = victim.read_bytes()
                before_mode = victim.stat().st_mode
                before_inode = (victim.stat().st_dev, victim.stat().st_ino)
                with self.assertRaisesRegex(
                    recovery.RecoveryError, "single-link"
                ):
                    recovery._atomic_write_bytes_once(
                        target,
                        payload,
                        mode=0o444,
                        label="hardlink victim probe",
                    )
                self.assertEqual(victim.read_bytes(), before)
                self.assertEqual(victim.stat().st_mode, before_mode)
                self.assertEqual(
                    (attacked.stat().st_dev, attacked.stat().st_ino), before_inode
                )
                self.assertEqual(victim.stat().st_nlink, 2)

    def test_completed_journal_manifest_and_lock_hardlinks_fail_closed(self) -> None:
        freeze = support.load_json(
            support.DEFAULT_FREEZE_MANIFEST, label="current fast freeze"
        )
        base = self._base_preflight(freeze)
        reports = support.ROOT / "reports"
        for artifact in ("journal", "manifest", "lock"):
            with self.subTest(artifact=artifact), tempfile.TemporaryDirectory(
                dir=reports
            ) as directory:
                patches, preflight, paths = self._fixture(
                    Path(directory), base_preflight=base, freeze=freeze
                )
                with patches:
                    recovery.apply_recovery(preflight)
                    attacked = {
                        "journal": paths["journal"] / "move_00.complete.json",
                        "manifest": paths["relocated_provenance"],
                        "lock": paths["bad_lock"],
                    }[artifact]
                    victim_alias = attacked.with_name(
                        f"{attacked.name}.victim-alias"
                    )
                    before = attacked.read_bytes()
                    before_mode = attacked.stat().st_mode
                    os.link(attacked, victim_alias)
                    with self.assertRaisesRegex(
                        recovery.RecoveryError, "single-link"
                    ):
                        recovery.validate_completed_recovery()
                    self.assertEqual(attacked.read_bytes(), before)
                    self.assertEqual(victim_alias.read_bytes(), before)
                    self.assertEqual(attacked.stat().st_mode, before_mode)
                    self.assertEqual(attacked.stat().st_nlink, 2)

    def test_actual_completed_recovery_has_only_single_link_authority_files(
        self,
    ) -> None:
        if not recovery.RECOVERY_COMPLETION.exists():
            self.skipTest("Actual one-time recovery has not been applied yet")
        completed = recovery.validate_completed_recovery()
        self.assertEqual(
            completed["completion_payload_sha256"],
            "84a3b2d5ec8e302436d6e63ef43366c5dbd9437856d0b18c1a71aa81c35d71ab",
        )
        files = [
            path
            for path in recovery.RECOVERY_ARCHIVE.rglob("*")
            if path.is_file()
        ]
        files.extend(
            [
                recovery.BAD_LOCK,
                Path(str(recovery.BAD_LOCK) + ".trusted-provenance.json"),
            ]
        )
        for path in files:
            with self.subTest(path=path):
                self.assertFalse(path.is_symlink())
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_nlink, 1)


class FastTrackWrapperArgvTest(unittest.TestCase):
    def _call(self, module: object, argv: list[str]) -> mock.Mock:
        with mock.patch.object(module.support, "load_scoped_loss_lock") as loader, mock.patch.object(
            module.support, "run_forwarded_main"
        ) as forwarded, mock.patch.object(sys, "argv", [str(module.__file__), *argv]):
            module.main()
        loader.assert_called_once()
        forwarded.assert_called_once()
        return forwarded

    def test_generator_and_summarizer_are_lock_only(self) -> None:
        lock = "/tmp/loss.lock.json"
        generated = self._call(create_wrapper, ["--stage-lock", lock])
        self.assertEqual(generated.call_args.args[1], ["--plan", str(support.DEFAULT_PLAN), "--stage-lock", lock])
        summarized = self._call(summarize_wrapper, ["--stage-lock", lock])
        forwarded = summarized.call_args.args[1]
        self.assertIn("--stage-lock", forwarded)
        self.assertNotIn("--stage", forwarded)
        for module in (create_wrapper, summarize_wrapper):
            with self.subTest(module=module.__name__), mock.patch.object(
                sys, "argv", [str(module.__file__), "--stage", support.LOSS_PRIMARY_STAGE]
            ), self.assertRaises(SystemExit):
                module.main()

    def test_launcher_accepts_only_dry_run_or_submit_wait(self) -> None:
        lock_args = ["--stage-lock", "/tmp/loss.lock.json"]
        dry = self._call(launcher_wrapper, [*lock_args, "--dry-run"])
        self.assertEqual(dry.call_args.args[1][-1], "--dry-run")
        submit = self._call(launcher_wrapper, [*lock_args, "--submit", "--wait"])
        self.assertEqual(submit.call_args.args[1][-2:], ["--submit", "--wait"])
        invalid = (
            [*lock_args, "--submit"],
            [*lock_args, "--dry-run", "--wait"],
            [*lock_args, "--force-resubmit", "--dry-run"],
            [*lock_args, "--retry-failed", "--dry-run"],
            [*lock_args, "--allow-background-fanout", "--dry-run"],
            [*lock_args, "--no-wait", "--submit"],
            [*lock_args, "--status"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), mock.patch.object(sys, "argv", [launcher_wrapper.__file__, *argv]), self.assertRaises(SystemExit):
                launcher_wrapper.main()

    def test_materializer_rejects_confirmation_and_requires_exact_mode_shape(self) -> None:
        with mock.patch.object(sys, "argv", [materialize_wrapper.__file__, "confirmation"]), self.assertRaises(SystemExit):
            materialize_wrapper.main()
        primary = [
            "loss_primary", "--prerequisite-lock", "/tmp/dropout.lock.json",
            "--source-stage", support.SOURCE_STAGE,
            "--history-summary", str(support.DEFAULT_RECEIPT),
            "--output", "/tmp/primary.lock.json",
        ]
        with mock.patch.object(
            materialize_wrapper.support, "run_forwarded_main"
        ) as forwarded, mock.patch.object(
            materialize_wrapper.support, "load_scoped_loss_lock"
        ) as reloaded, mock.patch.object(
            sys, "argv", [materialize_wrapper.__file__, *primary]
        ):
            materialize_wrapper.main()
        forwarded.assert_called_once()
        reloaded.assert_called_once()
        with mock.patch.object(
            sys,
            "argv",
            [materialize_wrapper.__file__, "loss_primary", "--output", "/tmp/x"],
        ), self.assertRaises(SystemExit):
            materialize_wrapper.main()

    def test_controller_schema2_commands_use_strict_wrappers_and_freeze(self) -> None:
        instance = controller_module.LossFastTrackController.__new__(controller_module.LossFastTrackController)
        instance.python_executable = sys.executable
        instance.policy_path = support.DEFAULT_POLICY
        instance.receipt_path = support.DEFAULT_RECEIPT
        instance.freeze_manifest_path = support.DEFAULT_FREEZE_MANIFEST
        instance.paths = core.ControllerPaths()
        lock = core.LockInfo(
            path=Path("/tmp/loss.lock.json"), payload={}, schema_version=2,
            effective_stage=support.LOSS_PRIMARY_STAGE, payload_sha256="a" * 64,
            execution_status="runnable", is_boundary=False, kernel_slugs=(),
        )
        for submit in (False, True):
            command = instance._launcher_command(lock=lock, stage=support.LOSS_PRIMARY_STAGE, submit=submit)
            self.assertEqual(Path(command.argv[1]).name, "run_minilm_5ep_sft_loss_fast_track_kaggle.py")
            self.assertIn("--fast-track-freeze-manifest", command.argv)
            self.assertIn("--stage-lock", command.argv)
            self.assertNotIn("--stage", command.argv)
            self.assertFalse(core.FORBIDDEN_LAUNCHER_FLAGS & set(command.argv))


class ReceiptSchemaProbeTest(unittest.TestCase):
    """Exercise exact nested receipt schemas with a fully path-bound local ledger."""

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.report = self.root / "report"
        self.locks_dir = self.report / "stage_locks"
        self.artifacts = self.root / "artifacts"
        self.receipt_path = self.report / "fast_track" / "skip.receipt.json"
        self.locks_dir.mkdir(parents=True)
        self.artifacts.mkdir()
        self.plan = generator.load_plan(support.DEFAULT_PLAN)
        self.prior = [f"prior-{index:02d}" for index in range(18)]
        self.config = generator.cross_builder.load_training_config(generator.BASE_CONFIG_PATH)
        self.selected = self._selected()
        self.lock_refs, self.strict_locks = self._locks()
        self.summary_refs, self.summaries = self._summaries()
        source = self.summaries[-1]
        self.root_summary = self.report / "summary.json"
        self.root_summary.write_text(json.dumps(source), encoding="utf-8")
        doc_sha = support.adaptive._summary_document_sha(source)
        snapshot = self.receipt_path.parent / "authority" / f"dropout_root_summary.{doc_sha}.json"
        snapshot.parent.mkdir(parents=True)
        snapshot.write_text(support.canonical_json_dumps(source) + "\n", encoding="utf-8")
        freeze = support.load_freeze_manifest()
        self.payload = {
            "schema_version": 1,
            "kind": support.RECEIPT_KIND,
            "campaign": self.plan["campaign"],
            "policy_authority": {"path": str(support.DEFAULT_POLICY.resolve()), "canonical_sha256": support.POLICY_CANONICAL_SHA256},
            "freeze_authority": {
                "path": str(support.DEFAULT_FREEZE_MANIFEST.resolve()),
                "file_sha256": support.file_sha256(support.DEFAULT_FREEZE_MANIFEST),
                "manifest_payload_sha256": freeze["manifest_payload_sha256"],
                "reviewed_execution_file_sha256": freeze["reviewed_execution_file_sha256"],
                "reviewed_review_file_sha256": freeze["reviewed_review_file_sha256"],
            },
            "source_plan": {
                "path": str(support.DEFAULT_PLAN.resolve()),
                "canonical_sha256": support.PLAN_CANONICAL_SHA256,
                "file_sha256": support.file_sha256(support.DEFAULT_PLAN),
            },
            "source_stage": support.SOURCE_STAGE,
            "source_stage_summary": deepcopy(self.summary_refs[-1]),
            "source_root_summary": {
                "original_path": str(self.root_summary.resolve()),
                "original_file_sha256_at_materialization": support.file_sha256(self.root_summary),
                "snapshot_path": str(snapshot.resolve()),
                "document_sha256": doc_sha,
                "snapshot_file_sha256": support.file_sha256(snapshot),
            },
            "source_lock": deepcopy(self.lock_refs[-1]),
            "selected_parent": deepcopy(self.selected),
            "skipped_coordinate": {
                "name": "max_grad_norm", "stage": support.SKIPPED_STAGE,
                "evaluated": False, "new_kernels": 0, "metric_claim": None,
                "inherited_value": 1.0,
                "reason": support.load_policy()["skip_semantics"]["reason"],
            },
            "loss_execution": deepcopy(support.load_policy()["loss_execution"]),
            "external_actions": deepcopy(support.load_policy()["external_actions"]),
            "artifacts_dir": str(self.artifacts.resolve()),
            "prior_stage_locks": deepcopy(self.lock_refs),
            "prior_stage_summaries": deepcopy(self.summary_refs),
            "budget": support._budget_payload(plan=self.plan, prior_slugs=self.prior, selected_parent=self.selected),
        }
        self._rehash()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _selected(self) -> dict:
        paths = {}
        hashes = {}
        for key in ("completion", "training_config", "iid_predictions", "sheets_sync"):
            path = self.artifacts / key
            path.write_text(key, encoding="utf-8")
            paths[key] = str(path.resolve())
            hashes[key] = support.file_sha256(path)
        recipe = support.canonical_sha256(self.config)
        return {
            "experiment": "selected", "run_id": "1" * 32, "kernel_slug": "pm-selected",
            "source_stage": support.SOURCE_STAGE, "source_role": "stage_anchor",
            "source_is_hypothesis": False, "loss_variant": "bce",
            "loss_hook_sha256": generator.LOSS_VARIANT_SHA256["bce"], "seed": 42,
            "resolved_config": deepcopy(self.config), "recipe_sha256": recipe,
            "recipe_family_sha256": "2" * 64, "code_bundle_sha256": "3" * 64,
            "expected_source_sha256": "3" * 64, "iid_predictions_sha256": "4" * 64,
            "completion_sha256": "5" * 64, "training_config_artifact_sha256": "6" * 64,
            "completion_notes_sha256": "7" * 64, "iid_macro_ap": 0.81,
            "hard_macro_ap": 0.42, "ood_macro_ap": 0.65,
            "sheets_sync_status": "synced", "summary_row_sha256": "8" * 64,
            "artifact_paths": paths, "artifact_file_sha256s": hashes,
        }

    def _locks(self) -> tuple[list[dict], list[dict]]:
        refs = []
        payloads = []
        for index, stage in enumerate(support.REQUIRED_NORMAL_LOCK_STAGES):
            path = self.locks_dir / f"{index}.lock.json"
            sha = f"{index + 1:064x}"
            payload = {"schema_version": 1, "effective_stage": stage, "transition_kind": "stage_transition", "lock_payload_sha256": sha}
            path.write_text(json.dumps(payload), encoding="utf-8")
            refs.append({
                "schema_version": 1, "effective_stage": stage, "is_boundary": False,
                "path": str(path.resolve()), "lock_payload_sha256": sha,
                "file_sha256": support.file_sha256(path),
            })
            payloads.append(payload)
        return refs, payloads

    def _summaries(self) -> tuple[list[dict], list[dict]]:
        refs = []
        documents = []
        by_stage = {ref["effective_stage"]: ref for ref in self.lock_refs}
        for stage in support.HISTORY_STAGES:
            bound = by_stage.get(stage)
            document = {
                "schema_version": 1,
                "campaign": self.plan["campaign"],
                "stages": {stage: {"complete": True, "decision_status": "ready"}},
                "runs": [],
            }
            if bound:
                document["stage_lock"] = {"lock_payload_sha256": bound["lock_payload_sha256"]}
            path = self.report / "stages" / stage / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(document), encoding="utf-8")
            refs.append({
                "effective_stage": stage, "path": str(path.resolve()),
                "document_sha256": support.adaptive._summary_document_sha(document),
                "file_sha256": support.file_sha256(path),
                "bound_lock_payload_sha256": bound["lock_payload_sha256"] if bound else None,
                "bound_lock_is_boundary": False if bound else None,
            })
            documents.append(document)
        return refs, documents

    def _rehash(self) -> None:
        self.payload.pop("summary_payload_sha256", None)
        self.payload["summary_payload_sha256"] = support.canonical_sha256(self.payload)
        self.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        self.receipt_path.write_text(support.canonical_json_dumps(self.payload) + "\n", encoding="utf-8")

    def _validate(self) -> dict:
        def strict(path: Path, **_: object) -> dict:
            index = int(path.name.split(".")[0])
            return deepcopy(self.strict_locks[index])

        with mock.patch.object(support, "_strict_schema1_lock", side_effect=strict), mock.patch.object(
            support, "_selected_parent_payload", return_value=deepcopy(self.selected)
        ), mock.patch.object(
            support, "_reconstruct_history_kernel_slugs", return_value=list(self.prior)
        ):
            return support.validate_receipt(self.receipt_path)

    def test_exact_nested_receipt_validates(self) -> None:
        self.assertEqual(self._validate(), self.payload)

    def test_rehashed_nested_extra_or_omission_is_rejected(self) -> None:
        for mutate in (
            lambda value: value["selected_parent"].update(extra="forbidden"),
            lambda value: value["budget"].pop("maximum_new_loss_kernels"),
            lambda value: value["source_lock"].update(effective_stage=support.SKIPPED_STAGE),
            lambda value: value["selected_parent"].update(iid_macro_ap=0.1),
            lambda value: value["selected_parent"]["artifact_file_sha256s"].update(
                completion="0" * 64
            ),
            lambda value: value["budget"]["primary_new_kernel_slugs"].pop(),
            lambda value: value["freeze_authority"].update(extra="forbidden"),
        ):
            original = deepcopy(self.payload)
            with self.subTest(mutate=mutate):
                self.payload = deepcopy(original)
                mutate(self.payload)
                self._rehash()
                with self.assertRaises(support.FastTrackError):
                    self._validate()
            self.payload = original

    def test_budget_union_arithmetic_and_snapshot_tamper_are_rejected(self) -> None:
        self.payload["budget"]["maximum_resulting_kernels"] -= 1
        self._rehash()
        with self.assertRaises(support.FastTrackError):
            self._validate()
        self.payload["budget"] = support._budget_payload(
            plan=self.plan,
            prior_slugs=self.prior,
            selected_parent=self.selected,
        )
        snapshot = Path(self.payload["source_root_summary"]["snapshot_path"])
        snapshot.write_text("{}\n", encoding="utf-8")
        self._rehash()
        with self.assertRaises(support.FastTrackError):
            self._validate()

    def test_source_stage_and_materialized_root_hash_are_exact(self) -> None:
        self.payload["source_stage"] = support.SKIPPED_STAGE
        self._rehash()
        with self.assertRaises(support.FastTrackError):
            self._validate()
        self.payload["source_stage"] = support.SOURCE_STAGE
        self.payload["source_root_summary"][
            "original_file_sha256_at_materialization"
        ] = "0" * 64
        self._rehash()
        with self.assertRaises(support.FastTrackError):
            self._validate()

    def test_post_schema2_root_cannot_rehash_materialized_dropout_file_sha(self) -> None:
        advanced = {
            "schema_version": 2,
            "campaign": self.plan["campaign"],
            "stages": {support.LOSS_PRIMARY_STAGE: {"complete": True}},
        }
        self.root_summary.write_text(json.dumps(advanced), encoding="utf-8")
        self.payload["source_root_summary"][
            "original_file_sha256_at_materialization"
        ] = "0" * 64
        self._rehash()
        with self.assertRaisesRegex(support.FastTrackError, "not bound"):
            self._validate()


class StageProjectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.original = core._stage_specs
        if hasattr(core, "_ORIGINAL_FAST_TRACK_STAGE_SPECS"):
            delattr(core, "_ORIGINAL_FAST_TRACK_STAGE_SPECS")

    def tearDown(self) -> None:
        core._stage_specs = self.original
        if hasattr(core, "_ORIGINAL_FAST_TRACK_STAGE_SPECS"):
            delattr(core, "_ORIGINAL_FAST_TRACK_STAGE_SPECS")

    def test_projection_skips_maxgrad_and_confirmation_only(self) -> None:
        controller_module.install_stage_projection()
        stages = core.validate_plan(generator.load_plan(support.DEFAULT_PLAN))
        names = [stage.effective_stage for stage in stages]
        self.assertNotIn(support.SKIPPED_STAGE, names)
        self.assertNotIn("confirmation__matched_seeds", names)
        self.assertEqual(names[-3:], list(support.LOSS_STAGES))
        primary = next(stage for stage in stages if stage.effective_stage == support.LOSS_PRIMARY_STAGE)
        self.assertEqual(primary.predecessor, support.SOURCE_STAGE)


if __name__ == "__main__":
    unittest.main()
