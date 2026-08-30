from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import create_minilm_5ep_sft_hparam_notebooks as generator
import materialize_minilm_5ep_sft_hparam_stage as axis_materializer
import materialize_minilm_5ep_sft_loss_confirmation as materializer


class LossConfirmationMaterializerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.artifacts = self.root / "artifacts"
        self.locks_dir = self.root / "locks"
        self.plan = json.loads(materializer.DEFAULT_PLAN.read_text(encoding="utf-8"))
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.base_config = json.loads(
            generator.BASE_CONFIG_PATH.read_text(encoding="utf-8")
        )
        self.current_source_sha256 = materializer._current_source_sha256()
        self.source_stage = axis_materializer.expected_source_stage(
            self.plan,
            target_stage="special_loss_screen",
            coordinate=None,
        )
        self.previous_source_stage = (
            "regularization_coordinate_search__classifier_dropout"
        )
        self.counter = 1
        self.trusted_contexts: dict[str, dict] = {}

        self.baseline_notes = (
            '{"batch_size_per_gpu":96,"campaign":"minilm_5ep_sft_hparam_search_v1",'
            '"effective_batch":192,"epochs":1,"gradient_accumulation":1,'
            '"label_smoothing":0.0,"learning_rate":2e-05,"max_grad_norm":1.0,'
            '"model_load_kwargs":{},"role":"current_protocol_control","seed":42,'
            '"stage":"lr_epochs_grid","warmup_ratio":0.05,"weight_decay":0.01}'
        )
        self.baseline_row = self._make_run(
            experiment="minilm5_sft_e1_lr2e5_control_v1",
            slug="minilm5-sft-grid-exact-control",
            loss_variant="bce",
            config=self.base_config,
            iid_macro_ap=0.790,
            notes=self.baseline_notes,
            stage="lr_log_line",
            role="current_protocol_control",
        )

        self.tuned_config = deepcopy(self.base_config)
        self.tuned_config["weight_decay"] = 0.05
        self.tuned_notes = materializer.canonical_json_dumps(
            {
                "campaign": self.plan["campaign"],
                "stage": self.previous_source_stage,
                "role": "candidate",
                "loss_variant": "bce",
                "seed": 42,
            }
        )
        tuned_parent_row = self._make_run(
            experiment="selected_regularized_bce",
            slug="pm-selected-regularized-bce",
            loss_variant="bce",
            config=self.tuned_config,
            iid_macro_ap=0.800,
            notes=self.tuned_notes,
            stage=self.previous_source_stage,
        )
        transition_summary_path = self.root / "max-grad-source-summary.json"
        transition_summary_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "campaign": self.plan["campaign"],
                    "stages": {
                        self.previous_source_stage: {
                            "complete": True,
                            "decision_status": "ready",
                            "control_gate": "passed",
                            "needs_boundary_extension": False,
                            "recommended_experiment": tuned_parent_row["experiment"],
                            "recommended_run_id": tuned_parent_row["run_id"],
                        }
                    },
                    "runs": [tuned_parent_row],
                }
            ),
            encoding="utf-8",
        )
        source_lock_path = self.locks_dir / "source.lock.json"
        self.source_lock_path = source_lock_path
        self.source_lock = axis_materializer.materialize_stage_lock(
            plan_path=self.plan_path,
            summary_path=transition_summary_path,
            artifacts_dir=self.artifacts,
            source_stage=self.previous_source_stage,
            target_stage="regularization_coordinate_search",
            coordinate="max_grad_norm",
            output_path=source_lock_path,
        )
        # Prove the prerequisite fixture satisfies the same strict schema-v1
        # validation used by the notebook generator and by the adaptive loader.
        self.assertEqual(
            generator.load_stage_lock(
                source_lock_path,
                plan=self.plan,
                base_config=self.base_config,
            ),
            self.source_lock,
        )
        self.tuned_row = deepcopy(tuned_parent_row)
        self.tuned_row.update(
            {
                "stage": self.source_stage,
                "effective_stage": self.source_stage,
                "role": "stage_anchor",
                "is_hypothesis": False,
            }
        )
        source_variant_slugs = [
            variant["kernel_slug"]
            for variant in self.source_lock["resolved_stage"]["variants"]
        ]
        prior_slugs = [f"prior-kernel-{index:02d}" for index in range(18)]
        prior_slugs.extend(
            [
                self.baseline_row["kernel_slug"],
                self.tuned_row["kernel_slug"],
                *source_variant_slugs,
            ]
        )
        self.source_summary = {
            "schema_version": 1,
            "campaign": self.plan["campaign"],
            "budget": {
                "history_complete_through": self.source_stage,
                "unique_kernel_slugs": prior_slugs,
            },
            "stages": {
                self.source_stage: {
                    "complete": True,
                    "decision_status": "ready",
                    "control_gate": "passed",
                    "needs_boundary_extension": False,
                    "recommended_experiment": self.tuned_row["experiment"],
                    "recommended_run_id": self.tuned_row["run_id"],
                }
            },
            "runs": [self.tuned_row],
        }
        self.source_summary_path = self._write_bound_summary(
            self.source_summary, "source-summary.json"
        )
        self.baseline_summary = {
            "campaign": self.plan["campaign"],
            "runs": [self.baseline_row],
        }
        self.baseline_summary_path = self._write_bound_summary(
            self.baseline_summary, "baseline-summary.json"
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _next_run_id(self) -> str:
        value = f"{self.counter:032x}"
        self.counter += 1
        return value

    def _write_bound_summary(self, summary: dict, name: str) -> Path:
        path = self.root / "summaries" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary), encoding="utf-8")
        return path

    def _make_run(
        self,
        *,
        experiment: str,
        slug: str,
        loss_variant: str,
        config: dict,
        iid_macro_ap: float,
        notes: str,
        stage: str,
        role: str = "candidate",
    ) -> dict:
        run_id = self._next_run_id()
        model_dir = self.artifacts / slug / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        artifact_config = deepcopy(config)
        artifact_config["model"] = "/kaggle/input/frozen/model"
        (model_dir / "training_config.json").write_text(
            json.dumps(artifact_config), encoding="utf-8"
        )
        iid_path = model_dir / "iid_validation_predictions.parquet"
        iid_path.write_bytes(f"iid:{experiment}:{run_id}".encode("utf-8"))
        completion = {
            "status": "complete",
            "experiment": experiment,
            "run_id": run_id,
            "frozen_recipe_sha256": materializer.canonical_sha256(config),
            "loss_hook_sha256": generator.LOSS_VARIANT_SHA256[loss_variant],
            "code_bundle_sha256": self.current_source_sha256,
            "notes": notes,
            "train_data": {
                "train_pairs": 306_669,
                "items": 711_304,
                "same_size_as_human_baseline": True,
            },
            "training_report": {
                "training_sampling": "none",
                "training_loss_weighting": "none",
                "training_subset": "all",
                "original_training_examples": 306_669,
                "training_unique_coverage_per_epoch": 1.0,
                "training_loss_weight_min": 1.0,
                "training_loss_weight_median": 1.0,
                "training_loss_weight_max": 1.0,
            },
        }
        (self.artifacts / slug / "notebook_completed.json").write_text(
            json.dumps(completion), encoding="utf-8"
        )
        return {
            "stage": stage,
            "effective_stage": stage,
            "experiment": experiment,
            "kernel_slug": slug,
            "role": role,
            "loss_variant": loss_variant,
            "loss_hook_sha256": generator.LOSS_VARIANT_SHA256[loss_variant],
            "completed": True,
            "status": "complete",
            "run_id": run_id,
            "seed": config["seed"],
            "iid_macro_ap": iid_macro_ap,
            "recipe_sha256": materializer.canonical_sha256(config),
            "iid_predictions_sha256": materializer.file_sha256(iid_path),
            "resolved_config": deepcopy(config),
            "expected_notes_sha256": materializer.text_sha256(notes),
        }

    def _primary(self) -> dict:
        output_path = self.locks_dir / "primary.lock.json"
        lock = materializer.materialize_loss_primary_lock(
            plan=self.plan,
            summary=self.source_summary,
            summary_path=self.source_summary_path,
            artifacts_dir=self.artifacts,
            prerequisite_locks=[self.source_lock],
            prerequisite_lock_paths=[self.source_lock_path],
            history_documents=[],
            history_document_paths=[],
            output_path=output_path,
        )
        return self._remember_trusted_context(lock, output_path)

    def _remember_trusted_context(self, lock: dict, lock_path: Path) -> dict:
        trusted = materializer.load_trusted_provenance(
            materializer.trusted_provenance_manifest_path(lock_path),
            plan=self.plan,
        )
        self.trusted_contexts[lock["lock_payload_sha256"]] = trusted
        return lock

    def _run_from_variant(self, lock: dict, variant: dict, ap: float) -> dict:
        return self._make_run(
            experiment=variant["experiment"],
            slug=variant["kernel_slug"],
            loss_variant=variant["loss_variant"],
            config=variant["resolved_config"],
            iid_macro_ap=ap,
            notes=materializer.expected_variant_notes(lock, variant),
            stage=lock["effective_stage"],
            role=variant["role"],
        )

    def _primary_summary(
        self,
        primary: dict,
        *,
        metrics: dict[str, float] | None = None,
    ) -> tuple[dict, dict[str, dict]]:
        metrics = metrics or {
            "balanced_binary_bce": 0.805,
            "balanced_category_class_sqrt_bce": 0.804,
            "balanced_category_class_bce": 0.799,
            "focal_bce_gamma2_scale4": 0.801,
        }
        rows = {}
        for variant in primary["resolved_stage"]["variants"]:
            rows[variant["loss_variant"]] = self._run_from_variant(
                primary, variant, metrics[variant["loss_variant"]]
            )
        summary = {
            "schema_version": 2,
            "campaign": self.plan["campaign"],
            "execution_status": "complete",
            "execution_lock_sha256s": [primary["lock_payload_sha256"]],
            "runs": [self.tuned_row, *rows.values()],
        }
        return summary, rows

    def _overlay(
        self,
        primary: dict,
        summary: dict,
        *,
        name: str = "overlay.lock.json",
    ) -> dict:
        summary_path = self._write_bound_summary(summary, f"{name}.summary.json")
        output_path = self.locks_dir / name
        lock = materializer.materialize_loss_overlay_lock(
            plan=self.plan,
            summary=summary,
            summary_path=summary_path,
            artifacts_dir=self.artifacts,
            prerequisite_locks=[primary],
            prerequisite_lock_paths=[self.locks_dir / "primary.lock.json"],
            history_documents=[],
            history_document_paths=[],
            output_path=output_path,
        )
        return self._remember_trusted_context(lock, output_path)

    def _primary_final_summary(
        self,
        primary: dict,
        overlay: dict,
        primary_summary: dict,
        *,
        overlay_ap: float = 0.803,
    ) -> tuple[dict, dict | None]:
        rows = list(primary_summary["runs"])
        overlay_row = None
        if overlay["execution_status"] == "runnable":
            overlay_variant = overlay["resolved_stage"]["variants"][0]
            overlay_row = self._run_from_variant(overlay, overlay_variant, overlay_ap)
            rows.append(overlay_row)
        return (
            {
                "schema_version": 2,
                "campaign": self.plan["campaign"],
                "execution_status": "complete",
                "execution_lock_sha256s": [
                    primary["lock_payload_sha256"],
                    *(
                        [overlay["lock_payload_sha256"]]
                        if overlay["execution_status"] == "runnable"
                        else []
                    ),
                ],
                "runs": rows,
            },
            overlay_row,
        )

    def _refine(self, primary: dict, overlay: dict, summary: dict, *, name: str = "lr.lock.json") -> dict:
        summary_path = self._write_bound_summary(summary, f"{name}.summary.json")
        output_path = self.locks_dir / name
        lock = materializer.materialize_loss_lr_refine_lock(
            plan=self.plan,
            summary=summary,
            summary_path=summary_path,
            artifacts_dir=self.artifacts,
            prerequisite_locks=[primary, overlay],
            prerequisite_lock_paths=[
                self.locks_dir / "primary.lock.json",
                self.locks_dir / "overlay.lock.json",
            ],
            history_documents=[],
            history_document_paths=[],
            output_path=output_path,
        )
        return self._remember_trusted_context(lock, output_path)

    def _loss_final_summary(
        self,
        primary: dict,
        overlay: dict,
        refine: dict,
        primary_final_summary: dict,
        *,
        lr_metrics: tuple[float, float] = (0.806, 0.804),
    ) -> tuple[dict, list[dict]]:
        rows = list(primary_final_summary["runs"])
        lr_rows = []
        if refine["execution_status"] == "runnable":
            for variant, ap in zip(refine["resolved_stage"]["variants"], lr_metrics):
                row = self._run_from_variant(refine, variant, ap)
                lr_rows.append(row)
                rows.append(row)
        return (
            {
                "schema_version": 2,
                "campaign": self.plan["campaign"],
                "execution_status": "complete",
                "execution_lock_sha256s": [
                    primary["lock_payload_sha256"],
                    *(
                        [overlay["lock_payload_sha256"]]
                        if overlay["execution_status"] == "runnable"
                        else []
                    ),
                    *(
                        [refine["lock_payload_sha256"]]
                        if refine["execution_status"] == "runnable"
                        else []
                    ),
                ],
                "runs": rows,
            },
            lr_rows,
        )

    def _full_chain(self, *, metrics: dict[str, float] | None = None):
        primary = self._primary()
        primary_summary, primary_rows = self._primary_summary(primary, metrics=metrics)
        overlay = self._overlay(primary, primary_summary)
        primary_final, overlay_row = self._primary_final_summary(
            primary, overlay, primary_summary
        )
        refine = self._refine(primary, overlay, primary_final)
        loss_final, lr_rows = self._loss_final_summary(
            primary, overlay, refine, primary_final
        )
        return {
            "primary": primary,
            "primary_rows": primary_rows,
            "overlay": overlay,
            "overlay_row": overlay_row,
            "refine": refine,
            "lr_rows": lr_rows,
            "loss_final": loss_final,
        }

    def _confirmation(self, chain: dict, *, name: str = "confirmation.lock.json") -> dict:
        summary_path = self._write_bound_summary(
            chain["loss_final"], f"{name}.summary.json"
        )
        output_path = self.locks_dir / name
        lock = materializer.materialize_confirmation_lock(
            plan=self.plan,
            summary=chain["loss_final"],
            summary_path=summary_path,
            baseline_summary=self.baseline_summary,
            baseline_summary_path=self.baseline_summary_path,
            artifacts_dir=self.artifacts,
            prerequisite_locks=[
                chain["primary"],
                chain["overlay"],
                chain["refine"],
            ],
            prerequisite_lock_paths=[
                self.locks_dir / "primary.lock.json",
                self.locks_dir / "overlay.lock.json",
                self.locks_dir / "lr.lock.json",
            ],
            history_documents=[],
            history_document_paths=[],
            output_path=output_path,
        )
        return self._remember_trusted_context(lock, output_path)

    def _rehashed_path(self, payload: dict, name: str) -> Path:
        rewritten = deepcopy(payload)
        rewritten.pop("lock_payload_sha256", None)
        rewritten["lock_payload_sha256"] = materializer.canonical_sha256(rewritten)
        path = self.locks_dir / name
        path.write_text(
            materializer.canonical_json_dumps(rewritten) + "\n",
            encoding="utf-8",
        )
        return path

    def _assert_rehashed_rejected(
        self,
        payload: dict,
        *,
        name: str,
        mutate,
    ) -> None:
        trusted = self.trusted_contexts[payload["lock_payload_sha256"]]
        changed = deepcopy(payload)
        mutate(changed)
        path = self._rehashed_path(changed, name)
        with self.assertRaises(materializer.AdaptiveMaterializationError):
            materializer.read_lock(
                path,
                plan=self.plan,
                trusted_provenance=trusted,
            )

    @staticmethod
    def _refresh_decision_evidence(payload: dict) -> None:
        payload["decision_evidence_sha256"] = materializer.canonical_sha256(
            {"origins": payload["origins"]}
        )

    @staticmethod
    def _refresh_variant_budget(payload: dict) -> None:
        budget = payload["budget"]
        new_slugs = sorted(
            variant["kernel_slug"]
            for variant in payload["resolved_stage"]["variants"]
        )
        budget["new_unique_kernel_slugs"] = new_slugs
        budget["all_unique_kernel_slugs_after"] = sorted(
            set(budget["prior_unique_kernel_slugs"]) | set(new_slugs)
        )
        budget["new_unique_kernels"] = len(new_slugs)
        budget["resulting_unique_kernels"] = len(
            budget["all_unique_kernel_slugs_after"]
        )

    @staticmethod
    def _delete_slug_from_embedded_budget(budget: dict, slug: str) -> None:
        for key in (
            "prior_unique_kernel_slugs",
            "all_unique_kernel_slugs_after",
        ):
            if slug in budget[key]:
                budget[key].remove(slug)
        history = budget["history_snapshot"]
        if slug in history["prior_unique_kernel_slugs"]:
            history["prior_unique_kernel_slugs"].remove(slug)
        for document in history["source_documents"]:
            if slug in document["kernel_slugs"]:
                document["kernel_slugs"].remove(slug)
        for prerequisite in history["prerequisite_budgets"]:
            if slug in prerequisite["kernel_slugs"]:
                prerequisite["kernel_slugs"].remove(slug)
        if slug in history["frozen_origin_kernel_slugs"]:
            history["frozen_origin_kernel_slugs"].remove(slug)
        budget["prior_unique_kernels"] = len(budget["prior_unique_kernel_slugs"])
        budget["new_unique_kernels"] = len(budget["new_unique_kernel_slugs"])
        budget["resulting_unique_kernels"] = len(
            budget["all_unique_kernel_slugs_after"]
        )
        budget["history_snapshot_sha256"] = materializer.canonical_sha256(history)

    @classmethod
    def _coherently_delete_historical_slug(cls, lock: dict, slug: str) -> None:
        cls._delete_slug_from_embedded_budget(lock["budget"], slug)
        snapshots = {
            row["lock_payload_sha256"]: row
            for row in lock["budget"]["history_snapshot"][
                "prerequisite_budgets"
            ]
        }
        for reference in lock["prerequisites"]:
            if reference["schema_version"] != 2:
                continue
            cls._delete_slug_from_embedded_budget(reference["frozen_budget"], slug)
            if slug in reference["budget_all_unique_kernel_slugs_after"]:
                reference["budget_all_unique_kernel_slugs_after"].remove(slug)
            reference["budget_resulting_unique_kernels"] = len(
                reference["budget_all_unique_kernel_slugs_after"]
            )
            reference["budget_payload_sha256"] = materializer.canonical_sha256(
                reference["frozen_budget"]
            )
            snapshot = snapshots[reference["lock_payload_sha256"]]
            snapshot["budget_payload_sha256"] = reference[
                "budget_payload_sha256"
            ]
        lock["budget"]["history_snapshot_sha256"] = (
            materializer.canonical_sha256(lock["budget"]["history_snapshot"])
        )

    def test_primary_lock_is_canonical_m5_and_replay_does_not_reselect(self) -> None:
        lock = self._primary()
        self.assertEqual(lock["schema_version"], 2)
        self.assertEqual(lock["mode"], "loss_primary")
        self.assertEqual(lock["family"]["maximum_hypotheses"], 5)
        self.assertEqual(lock["family"]["reserved_conditional_hypotheses"], 1)
        variants = lock["resolved_stage"]["variants"]
        self.assertEqual(len(variants), 4)
        self.assertEqual({row["seed"] for row in variants}, {42})
        self.assertEqual({row["resolved_config"]["weight_decay"] for row in variants}, {0.05})
        self.assertEqual(lock["budget"]["prior_unique_kernels"], 22)
        self.assertEqual(lock["budget"]["resulting_unique_kernels"], 26)
        path = self.locks_dir / "primary.lock.json"
        self.assertEqual(
            path.read_text(encoding="utf-8"),
            materializer.canonical_json_dumps(lock) + "\n",
        )

        changed = deepcopy(self.source_summary)
        changed["runs"][0]["iid_macro_ap"] = 0.1
        self.source_summary_path.write_text(json.dumps(changed), encoding="utf-8")
        replay = materializer.materialize_loss_primary_lock(
            plan=self.plan,
            summary=changed,
            summary_path=self.source_summary_path,
            artifacts_dir=self.artifacts,
            prerequisite_locks=[self.source_lock],
            prerequisite_lock_paths=[self.source_lock_path],
            history_documents=[],
            history_document_paths=[],
            output_path=path,
        )
        self.assertEqual(replay, lock)
        cli_replay = materializer.materialize(
            mode="loss_primary",
            plan_path=self.plan_path,
            summary_path=self.source_summary_path,
            artifacts_dir=self.artifacts,
            prerequisite_lock_paths=[self.source_lock_path],
            output_path=path,
        )
        self.assertEqual(cli_replay, lock)
        moved_summary_path = self.source_summary_path.with_name(
            "source-summary-moved-after-lock.json"
        )
        self.source_summary_path.rename(moved_summary_path)
        self.assertEqual(
            materializer.materialize(
                mode="loss_primary",
                plan_path=self.plan_path,
                summary_path=self.source_summary_path,
                artifacts_dir=self.artifacts,
                prerequisite_lock_paths=[self.source_lock_path],
                output_path=path,
            ),
            lock,
        )
        trusted = self.trusted_contexts[lock["lock_payload_sha256"]]
        archived = json.loads(
            Path(trusted["source_documents"][0]["snapshot_path"]).read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(archived, self.source_summary)

    def test_primary_rejects_non_bce_parent_and_exact_notes_drift(self) -> None:
        bad = deepcopy(self.tuned_row)
        bad["loss_variant"] = "focal_bce_gamma2_scale4"
        summary = deepcopy(self.source_summary)
        summary["runs"] = [bad]
        summary_path = self._write_bound_summary(summary, "bad-loss-summary.json")
        with self.assertRaises(materializer.AdaptiveMaterializationError):
            materializer.materialize_loss_primary_lock(
                plan=self.plan,
                summary=summary,
                summary_path=summary_path,
                artifacts_dir=self.artifacts,
                prerequisite_locks=[self.source_lock],
                prerequisite_lock_paths=[self.source_lock_path],
                history_documents=[],
                history_document_paths=[],
                output_path=self.locks_dir / "bad-loss.lock.json",
            )

        completion_path = (
            self.artifacts / self.tuned_row["kernel_slug"] / "notebook_completed.json"
        )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["notes"] = materializer.canonical_json_dumps({"stage": "drift"})
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "notes differ"
        ):
            materializer.materialize_loss_primary_lock(
                plan=self.plan,
                summary=self.source_summary,
                summary_path=self.source_summary_path,
                artifacts_dir=self.artifacts,
                prerequisite_locks=[self.source_lock],
                prerequisite_lock_paths=[self.source_lock_path],
                history_documents=[],
                history_document_paths=[],
                output_path=self.locks_dir / "bad-notes.lock.json",
            )

    def test_primary_requires_complete_plausible_budget_history(self) -> None:
        missing = deepcopy(self.source_summary)
        missing.pop("budget")
        missing_path = self._write_bound_summary(missing, "missing-ledger-summary.json")
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "attested kernel ledger"
        ):
            materializer.materialize_loss_primary_lock(
                plan=self.plan,
                summary=missing,
                summary_path=missing_path,
                artifacts_dir=self.artifacts,
                prerequisite_locks=[self.source_lock],
                prerequisite_lock_paths=[self.source_lock_path],
                history_documents=[],
                history_document_paths=[],
                output_path=self.locks_dir / "missing-ledger.lock.json",
            )

        too_short = deepcopy(self.source_summary)
        too_short["budget"]["unique_kernel_slugs"] = [
            self.tuned_row["kernel_slug"]
        ]
        too_short_path = self._write_bound_summary(
            too_short, "short-ledger-summary.json"
        )
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "18 to 22"
        ):
            materializer.materialize_loss_primary_lock(
                plan=self.plan,
                summary=too_short,
                summary_path=too_short_path,
                artifacts_dir=self.artifacts,
                prerequisite_locks=[self.source_lock],
                prerequisite_lock_paths=[self.source_lock_path],
                history_documents=[],
                history_document_paths=[],
                output_path=self.locks_dir / "short-ledger.lock.json",
            )

        regularization = next(
            stage
            for stage in self.plan["stages"]
            if stage["name"] == "regularization_coordinate_search"
        )
        expected_stages = ["lr_log_line", "epoch_line"] + [
            f"regularization_coordinate_search__{coordinate}"
            for coordinate in regularization["execution_order"]
        ]
        stage_history = {
            "stages": {
                stage: {
                    "complete": True,
                    "decision_status": "ready",
                    "needs_boundary_extension": False,
                }
                for stage in expected_stages
            },
            "runs": [
                {
                    "completed": True,
                    "kernel_slug": slug,
                }
                for slug in self.source_summary["budget"]["unique_kernel_slugs"]
            ],
        }
        without_attestation = deepcopy(self.source_summary)
        without_attestation.pop("budget")
        without_attestation_path = self._write_bound_summary(
            without_attestation, "stage-history-source-summary.json"
        )
        stage_history_path = self._write_bound_summary(
            stage_history, "complete-stage-history.json"
        )
        duplicate_stage_history_path = self._write_bound_summary(
            deepcopy(stage_history), "complete-stage-history-copy.json"
        )
        lock = materializer.materialize_loss_primary_lock(
            plan=self.plan,
            summary=without_attestation,
            summary_path=without_attestation_path,
            artifacts_dir=self.artifacts,
            prerequisite_locks=[self.source_lock],
            prerequisite_lock_paths=[self.source_lock_path],
            # Root/latest and its immutable stage copy may be byte-identical;
            # the hashed history snapshot must deduplicate that one document.
            history_documents=[stage_history, deepcopy(stage_history)],
            history_document_paths=[
                stage_history_path,
                duplicate_stage_history_path,
            ],
            output_path=self.locks_dir / "stage-history-ledger.lock.json",
        )
        self.assertEqual(lock["budget"]["prior_unique_kernels"], 22)

    def test_plan_trigger_drift_is_rejected_before_materialization(self) -> None:
        changed = deepcopy(self.plan)
        loss = next(
            stage for stage in changed["stages"] if stage["name"] == "special_loss_screen"
        )
        loss["winner_lr_refinement_trigger"]["minimum_iid_delta_vs_tuned_bce"] = 0.001
        path = self.root / "changed-plan.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "LR-refinement trigger changed"
        ):
            materializer.load_plan(path)

    def test_exact_adaptive_plan_budgets_tie_runtime_and_counting_are_frozen(self) -> None:
        def stage(plan: dict, name: str) -> dict:
            return next(row for row in plan["stages"] if row["name"] == name)

        cases = {
            "loss_typical_4": lambda plan: stage(
                plan, "special_loss_screen"
            ).__setitem__("typical_new_runs", 5),
            "loss_maximum_7": lambda plan: stage(
                plan, "special_loss_screen"
            ).__setitem__("maximum_new_runs", 6),
            "confirmation_typical_6": lambda plan: stage(
                plan, "confirmation"
            ).__setitem__("typical_new_runs", 8),
            "confirmation_maximum_8": lambda plan: stage(
                plan, "confirmation"
            ).__setitem__("maximum_new_runs", 6),
            "tie_order": lambda plan: stage(plan, "confirmation").__setitem__(
                "final_tie_break_order",
                list(reversed(stage(plan, "confirmation")["final_tie_break_order"])),
            ),
            "runtime": lambda plan: stage(plan, "confirmation").__setitem__(
                "require_inference_runtime_check", False
            ),
            "counting": lambda plan: plan["budget"].__setitem__(
                "counting_rule", "count every row"
            ),
        }
        for name, mutate in cases.items():
            with self.subTest(name=name):
                changed = deepcopy(self.plan)
                mutate(changed)
                path = self.root / f"plan-{name}.json"
                path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(materializer.AdaptiveMaterializationError):
                    materializer.load_plan(path)

    def test_schema1_prerequisite_uses_strict_generator_loader(self) -> None:
        source_path = self.locks_dir / "source.lock.json"
        self.assertEqual(
            materializer.load_provenance_lock(source_path, plan=self.plan),
            self.source_lock,
        )

        skeletal = axis_materializer._with_payload_hash(
            {
                "schema_version": 1,
                "kind": "minilm_5ep_sft_stage_lock",
                "campaign": self.plan["campaign"],
                "source_stage": self.source_stage,
                "target_stage": "regularization_coordinate_search",
                "coordinate": "max_grad_norm",
                "effective_stage": self.source_stage,
                "source_plan_sha256": materializer.canonical_sha256(self.plan),
                "source_stage_snapshot_sha256": "b" * 64,
                "selection_metric": "iid_macro_ap",
                "parent": {
                    "experiment": self.tuned_row["experiment"],
                    "run_id": self.tuned_row["run_id"],
                    "kernel_slug": self.tuned_row["kernel_slug"],
                    "notes": self.tuned_notes,
                },
                "resolved_stage": {"variants": []},
            }
        )
        path = self.locks_dir / "skeletal-v1.lock.json"
        path.write_text(
            materializer.canonical_json_dumps(skeletal) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError,
            "Strict schema-v1 provenance validation failed",
        ):
            materializer.load_provenance_lock(path, plan=self.plan)

    def test_rehashed_primary_semantic_tampers_are_rejected(self) -> None:
        primary = self._primary()
        attacks = {
            "selection": lambda lock: lock.__setitem__(
                "selection_metric", "hard_macro_ap"
            ),
            "source": lambda lock: lock.__setitem__(
                "source_stage", "regularization_coordinate_search__weight_decay"
            ),
            "role": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "role", "stage_anchor"
            ),
            "hypothesis": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "is_hypothesis", False
            ),
            "lineage": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "origin_ids", []
            ),
            "slot": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "loss_declaration_rank", 99
            ),
        }
        for name, mutate in attacks.items():
            with self.subTest(name=name):
                self._assert_rehashed_rejected(
                    primary,
                    name=f"tampered-primary-{name}.json",
                    mutate=mutate,
                )

        def safe_but_wrong_config(lock: dict) -> None:
            anchor = lock["origins"][0]
            original = lock["resolved_stage"]["variants"][0]
            config = deepcopy(anchor["resolved_config"])
            config["weight_decay"] = 0.01
            lock["resolved_stage"]["variants"][0] = materializer._variant(
                plan=self.plan,
                mode="loss_primary",
                config=config,
                loss_variant=original["loss_variant"],
                expected_source_sha256=lock["expected_source_sha256"],
                origin_ids=[anchor["origin_id"]],
                family_size=5,
                extra={"loss_declaration_rank": original["loss_declaration_rank"]},
            )
            self._refresh_variant_budget(lock)

        self._assert_rehashed_rejected(
            primary,
            name="tampered-primary-safe-config.json",
            mutate=safe_but_wrong_config,
        )

        def anchor_role(lock: dict) -> None:
            lock["origins"][0]["source_role"] = "candidate"
            lock["origins"][0]["source_is_hypothesis"] = True
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            primary,
            name="tampered-primary-anchor-role.json",
            mutate=anchor_role,
        )

        def v1_parent_provenance(lock: dict) -> None:
            reference = lock["prerequisites"][0]
            reference["frozen_parent"]["iid_predictions_sha256"] = "0" * 64
            reference["frozen_parent_sha256"] = materializer.canonical_sha256(
                reference["frozen_parent"]
            )

        self._assert_rehashed_rejected(
            primary,
            name="tampered-primary-v1-parent-provenance.json",
            mutate=v1_parent_provenance,
        )

        def coherent_v1_anchor_source_tamper(lock: dict) -> None:
            anchor = lock["origins"][0]
            anchor["code_bundle_sha256"] = "f" * 64
            anchor["expected_source_sha256"] = "f" * 64
            reference = lock["prerequisites"][0]
            reference["frozen_parent"]["code_bundle_sha256"] = "f" * 64
            reference["frozen_parent_sha256"] = materializer.canonical_sha256(
                reference["frozen_parent"]
            )
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            primary,
            name="tampered-primary-coherent-v1-anchor-source.json",
            mutate=coherent_v1_anchor_source_tamper,
        )

    def test_overlay_uses_strict_iid_trigger_declaration_tie_break_and_reserved_slot(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(
            primary,
            metrics={
                "balanced_binary_bce": 0.805,
                "balanced_category_class_sqrt_bce": 0.805,
                "balanced_category_class_bce": 0.799,
                "focal_bce_gamma2_scale4": 0.801,
            },
        )
        summary["runs"][1]["hard_macro_ap"] = 0.0
        summary["runs"][2]["ood_macro_ap"] = 1.0
        lock = self._overlay(primary, summary)
        self.assertEqual(lock["execution_status"], "runnable")
        self.assertEqual(
            lock["decision"]["best_balance_loss_variant"],
            "balanced_binary_bce",
        )
        self.assertEqual(
            lock["resolved_stage"]["variants"][0]["loss_variant"],
            "balanced_binary_focal_gamma2_scale4",
        )
        self.assertEqual(lock["family"]["maximum_hypotheses"], 5)
        self.assertEqual(lock["family"]["reserved_slot_state"], "materialized")
        self.assertEqual(lock["budget"]["new_unique_kernels"], 1)

    def test_overlay_zero_delta_is_immutable_skip_receipt(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(
            primary,
            metrics={
                "balanced_binary_bce": 0.805,
                "balanced_category_class_sqrt_bce": 0.804,
                "balanced_category_class_bce": 0.799,
                "focal_bce_gamma2_scale4": 0.800,
            },
        )
        lock = self._overlay(primary, summary)
        self.assertEqual(lock["kind"], materializer.RECEIPT_KIND)
        self.assertEqual(lock["execution_status"], "skipped")
        self.assertFalse(lock["decision"]["triggered"])
        self.assertEqual(lock["family"]["reserved_slot_state"], "unused_p_equals_1")
        self.assertEqual(lock["resolved_stage"]["variants"], [])
        self.assertEqual(lock["budget"]["new_unique_kernels"], 0)

    def test_rehashed_overlay_semantics_and_frozen_evidence_tampers_are_rejected(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(primary)
        overlay = self._overlay(primary, summary)
        attacks = {
            "threshold": lambda lock: lock["decision"].__setitem__(
                "threshold", -0.001
            ),
            "reserved_slot": lambda lock: lock["family"].__setitem__(
                "reserved_slot", 4
            ),
            "resolved_source": lambda lock: lock.__setitem__(
                "source_stage", "special_loss_screen__arbitrary"
            ),
        }
        for name, mutate in attacks.items():
            with self.subTest(name=name):
                self._assert_rehashed_rejected(
                    overlay,
                    name=f"tampered-overlay-{name}.json",
                    mutate=mutate,
                )

        def evidence_metric(lock: dict) -> None:
            focal = next(
                origin
                for origin in lock["origins"]
                if origin["loss_variant"] == "focal_bce_gamma2_scale4"
            )
            focal["iid_macro_ap"] = 0.799
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            overlay,
            name="tampered-overlay-evidence-metric.json",
            mutate=evidence_metric,
        )

        def evidence_notes_lineage(lock: dict) -> None:
            origin = next(
                row
                for row in lock["origins"]
                if row["loss_variant"] == "balanced_binary_bce"
            )
            notes = json.loads(origin["completion_notes"])
            notes["origin_lineage"] = []
            origin["completion_notes"] = materializer.canonical_json_dumps(notes)
            origin["completion_notes_sha256"] = materializer.text_sha256(
                origin["completion_notes"]
            )
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            overlay,
            name="tampered-overlay-origin-lineage.json",
            mutate=evidence_notes_lineage,
        )

        def mixed_origin_source(lock: dict) -> None:
            origin = next(
                row
                for row in lock["origins"]
                if row["loss_variant"] == "balanced_binary_bce"
            )
            origin["code_bundle_sha256"] = "f" * 64
            origin["expected_source_sha256"] = "f" * 64
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            overlay,
            name="tampered-overlay-mixed-source.json",
            mutate=mixed_origin_source,
        )

    def test_lr_refine_changes_only_lr_and_keeps_loss_hook_m2(self) -> None:
        chain = self._full_chain()
        lock = chain["refine"]
        self.assertEqual(lock["execution_status"], "runnable")
        self.assertEqual(lock["family"]["maximum_hypotheses"], 2)
        self.assertEqual(len(lock["resolved_stage"]["variants"]), 2)
        center = next(
            origin
            for origin in lock["origins"]
            if origin["origin_id"] == lock["family"]["anchor_origin_id"]
        )
        for variant, multiplier in zip(
            lock["resolved_stage"]["variants"], (0.5, 2.0)
        ):
            expected = deepcopy(center["resolved_config"])
            expected["learning_rate"] *= multiplier
            self.assertEqual(variant["resolved_config"], expected)
            self.assertEqual(variant["loss_variant"], center["loss_variant"])
            self.assertEqual(
                variant["expected_loss_hook_sha256"], center["loss_hook_sha256"]
            )
        self.assertEqual(lock["budget"]["resulting_unique_kernels"], 29)

    def test_lr_refine_delta_exactly_margin_is_skipped(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(
            primary,
            metrics={
                "balanced_binary_bce": 0.802,
                "balanced_category_class_sqrt_bce": 0.801,
                "balanced_category_class_bce": 0.799,
                "focal_bce_gamma2_scale4": 0.800,
            },
        )
        overlay = self._overlay(primary, summary)
        final, _ = self._primary_final_summary(primary, overlay, summary)
        lock = self._refine(primary, overlay, final)
        self.assertEqual(lock["execution_status"], "skipped")
        self.assertFalse(lock["decision"]["triggered"])
        self.assertEqual(lock["resolved_stage"]["variants"], [])

    def test_rehashed_lr_threshold_role_and_line_tampers_are_rejected(self) -> None:
        chain = self._full_chain()
        refine = chain["refine"]
        attacks = {
            "threshold": lambda lock: lock["decision"].__setitem__(
                "threshold", 0.001
            ),
            "family": lambda lock: lock["family"].__setitem__(
                "maximum_hypotheses", 1
            ),
            "role": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "role", "lr_anchor"
            ),
            "line_multiplier": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "learning_rate_multiplier", 0.25
            ),
        }
        for name, mutate in attacks.items():
            with self.subTest(name=name):
                self._assert_rehashed_rejected(
                    refine,
                    name=f"tampered-lr-{name}.json",
                    mutate=mutate,
                )

    def test_confirmation_reuses_seed42_adds_eight_and_freezes_l2(self) -> None:
        chain = self._full_chain()
        lock = self._confirmation(chain)
        self.assertTrue(lock["decision"]["second_loss_finalist_triggered"])
        self.assertEqual(len(lock["decision"]["recipe_groups"]), 4)
        variants = lock["resolved_stage"]["variants"]
        self.assertEqual(len(variants), 8)
        self.assertEqual({variant["seed"] for variant in variants}, {17, 2026})
        self.assertNotIn(42, {variant["seed"] for variant in variants})
        self.assertEqual(lock["budget"]["resulting_unique_kernels"], 37)
        self.assertEqual(lock["decision"]["selected_checkpoint_seed"], 42)
        for group in lock["decision"]["recipe_groups"]:
            group_variants = [
                variant
                for variant in variants
                if variant["recipe_group_id"] == group["recipe_group_id"]
            ]
            self.assertEqual(len(group_variants), 2)
            self.assertEqual(
                {variant["expected_recipe_family_sha256"] for variant in group_variants},
                {group["recipe_family_sha256"]},
            )

    def test_confirmation_without_positive_close_l2_adds_six(self) -> None:
        chain = self._full_chain(
            metrics={
                "balanced_binary_bce": 0.805,
                "balanced_category_class_sqrt_bce": 0.801,
                "balanced_category_class_bce": 0.799,
                "focal_bce_gamma2_scale4": 0.799,
            }
        )
        lock = self._confirmation(chain, name="confirmation-six.lock.json")
        self.assertFalse(lock["decision"]["second_loss_finalist_triggered"])
        self.assertEqual(len(lock["decision"]["recipe_groups"]), 3)
        self.assertEqual(len(lock["resolved_stage"]["variants"]), 6)

    def test_rehashed_confirmation_baseline_roles_tie_and_runtime_are_rejected(self) -> None:
        chain = self._full_chain()
        confirmation = self._confirmation(chain)
        attacks = {
            "baseline_group": lambda lock: lock["family"].__setitem__(
                "baseline_recipe_group_id", "recipe_not_the_baseline"
            ),
            "matched_baseline_role": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "matched_baseline_role", "selected_regularized_bce_recipe"
            ),
            "candidate_role": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "role", "candidate"
            ),
            "candidate_hypothesis": lambda lock: lock["resolved_stage"]["variants"][0].__setitem__(
                "is_hypothesis", True
            ),
            "tie_order": lambda lock: lock["decision"].__setitem__(
                "final_tie_break_order",
                list(reversed(lock["decision"]["final_tie_break_order"])),
            ),
            "runtime": lambda lock: lock["decision"].__setitem__(
                "require_inference_runtime_check", False
            ),
            "group_roles": lambda lock: lock["decision"]["recipe_groups"][0].__setitem__(
                "roles", ["loss_finalist_1"]
            ),
        }
        for name, mutate in attacks.items():
            with self.subTest(name=name):
                self._assert_rehashed_rejected(
                    confirmation,
                    name=f"tampered-confirmation-{name}.json",
                    mutate=mutate,
                )

        def baseline_role(lock: dict) -> None:
            baseline_id = lock["decision"]["evidence_contract"][
                "baseline_origin_id"
            ]
            baseline = next(
                origin for origin in lock["origins"] if origin["origin_id"] == baseline_id
            )
            baseline["source_role"] = "stage_anchor"
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            confirmation,
            name="tampered-confirmation-baseline-origin-role.json",
            mutate=baseline_role,
        )

        def evidence_order(lock: dict) -> None:
            lock["decision"]["evidence_contract"][
                "primary_family_origin_ids"
            ].reverse()

        self._assert_rehashed_rejected(
            confirmation,
            name="tampered-confirmation-evidence-order.json",
            mutate=evidence_order,
        )

        def baseline_source(lock: dict) -> None:
            baseline_id = lock["decision"]["evidence_contract"][
                "baseline_origin_id"
            ]
            baseline = next(
                origin
                for origin in lock["origins"]
                if origin["origin_id"] == baseline_id
            )
            baseline["code_bundle_sha256"] = "f" * 64
            baseline["expected_source_sha256"] = "f" * 64
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            confirmation,
            name="tampered-confirmation-baseline-source.json",
            mutate=baseline_source,
        )

    def test_confirmation_budget_union_rejects_38(self) -> None:
        chain = self._full_chain()
        extra = {
            "runs": [
                {
                    "completed": True,
                    "kernel_slug": "one-unplanned-extra-kernel",
                }
            ]
        }
        loss_final_path = self._write_bound_summary(
            chain["loss_final"], "confirmation-overflow-summary.json"
        )
        extra_path = self._write_bound_summary(extra, "one-extra-history.json")
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "above hard cap 37"
        ):
            materializer.materialize_confirmation_lock(
                plan=self.plan,
                summary=chain["loss_final"],
                summary_path=loss_final_path,
                baseline_summary=self.baseline_summary,
                baseline_summary_path=self.baseline_summary_path,
                artifacts_dir=self.artifacts,
                prerequisite_locks=[
                    chain["primary"],
                    chain["overlay"],
                    chain["refine"],
                ],
                prerequisite_lock_paths=[
                    self.locks_dir / "primary.lock.json",
                    self.locks_dir / "overlay.lock.json",
                    self.locks_dir / "lr.lock.json",
                ],
                history_documents=[extra],
                history_document_paths=[extra_path],
                output_path=self.locks_dir / "confirmation-overflow.lock.json",
            )

    def test_rehashed_budget_history_counting_chain_origin_and_range_tampers_are_rejected(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(primary)
        overlay = self._overlay(primary, summary)

        self._assert_rehashed_rejected(
            primary,
            name="tampered-budget-counting.json",
            mutate=lambda lock: lock["budget"].__setitem__(
                "counting_identity", "run_id_union"
            ),
        )
        self._assert_rehashed_rejected(
            primary,
            name="tampered-budget-plan-rule.json",
            mutate=lambda lock: lock["budget"].__setitem__(
                "plan_counting_rule", "count rows"
            ),
        )

        def delete_prior_but_rehash_snapshot(lock: dict) -> None:
            budget = lock["budget"]
            slug = next(
                value
                for value in budget["prior_unique_kernel_slugs"]
                if value.startswith("prior-kernel-")
            )
            budget["prior_unique_kernel_slugs"].remove(slug)
            budget["all_unique_kernel_slugs_after"].remove(slug)
            budget["prior_unique_kernels"] -= 1
            budget["resulting_unique_kernels"] -= 1
            budget["history_snapshot"]["prior_unique_kernel_slugs"].remove(slug)
            budget["history_snapshot_sha256"] = materializer.canonical_sha256(
                budget["history_snapshot"]
            )

        self._assert_rehashed_rejected(
            primary,
            name="tampered-budget-deleted-prior.json",
            mutate=delete_prior_but_rehash_snapshot,
        )

        def omit_frozen_origin(lock: dict) -> None:
            history = lock["budget"]["history_snapshot"]
            history["frozen_origin_kernel_slugs"] = []
            lock["budget"]["history_snapshot_sha256"] = (
                materializer.canonical_sha256(history)
            )

        self._assert_rehashed_rejected(
            primary,
            name="tampered-budget-origin-omission.json",
            mutate=omit_frozen_origin,
        )

        def coherent_below_range(lock: dict) -> None:
            budget = lock["budget"]
            history = budget["history_snapshot"]
            source = history["source_documents"][0]
            removed = [
                slug
                for slug in source["kernel_slugs"]
                if slug.startswith("prior-kernel-")
            ][:5]
            for slug in removed:
                source["kernel_slugs"].remove(slug)
                history["prior_unique_kernel_slugs"].remove(slug)
                budget["prior_unique_kernel_slugs"].remove(slug)
                budget["all_unique_kernel_slugs_after"].remove(slug)
            budget["prior_unique_kernels"] = len(
                budget["prior_unique_kernel_slugs"]
            )
            budget["resulting_unique_kernels"] = len(
                budget["all_unique_kernel_slugs_after"]
            )
            budget["history_snapshot_sha256"] = materializer.canonical_sha256(
                history
            )

        self._assert_rehashed_rejected(
            primary,
            name="tampered-budget-below-range.json",
            mutate=coherent_below_range,
        )

        def delete_predecessor_reference(lock: dict) -> None:
            lock["prerequisites"] = []
            history = lock["budget"]["history_snapshot"]
            history["prerequisite_budgets"] = []
            lock["budget"]["history_snapshot_sha256"] = (
                materializer.canonical_sha256(history)
            )

        self._assert_rehashed_rejected(
            overlay,
            name="tampered-budget-deleted-predecessor.json",
            mutate=delete_predecessor_reference,
        )

    def test_external_budget_authorities_reject_coherent_single_slug_deletion_across_chain(self) -> None:
        chain = self._full_chain()
        confirmation = self._confirmation(chain)
        historical_slug = next(
            slug
            for slug in chain["primary"]["budget"]["prior_unique_kernel_slugs"]
            if slug.startswith("prior-kernel-")
        )
        expected_priors = {
            "primary": 22,
            "overlay": 26,
            "refine": 27,
            "confirmation": 29,
        }
        locks = {
            "primary": chain["primary"],
            "overlay": chain["overlay"],
            "refine": chain["refine"],
            "confirmation": confirmation,
        }
        for name, lock in locks.items():
            with self.subTest(stage=name):
                self.assertEqual(
                    lock["budget"]["prior_unique_kernels"], expected_priors[name]
                )

                def delete(payload: dict) -> None:
                    self._coherently_delete_historical_slug(
                        payload, historical_slug
                    )
                    self.assertEqual(
                        payload["budget"]["prior_unique_kernels"],
                        expected_priors[name] - 1,
                    )

                self._assert_rehashed_rejected(
                    lock,
                    name=f"coherent-one-slug-delete-{name}.json",
                    mutate=delete,
                )

    def test_forged_alternate_summary_cannot_become_primary_budget_authority(self) -> None:
        primary = self._primary()
        historical_slug = next(
            slug
            for slug in primary["budget"]["prior_unique_kernel_slugs"]
            if slug.startswith("prior-kernel-")
        )
        forged_summary = deepcopy(self.source_summary)
        forged_summary["budget"]["unique_kernel_slugs"].remove(historical_slug)
        forged_path = self._write_bound_summary(
            forged_summary, "forged-alternate-source-summary.json"
        ).resolve()
        forged_sha = materializer._summary_document_sha(forged_summary)

        def repoint(payload: dict) -> None:
            self._coherently_delete_historical_slug(payload, historical_slug)
            document = payload["budget"]["history_snapshot"][
                "source_documents"
            ][0]
            document["document_path"] = str(forged_path)
            document["document_sha256"] = forged_sha
            payload["decision_inputs_summary_sha256"] = forged_sha
            payload["budget"]["history_snapshot_sha256"] = (
                materializer.canonical_sha256(
                    payload["budget"]["history_snapshot"]
                )
            )

        self._assert_rehashed_rejected(
            primary,
            name="forged-alternate-summary-primary.json",
            mutate=repoint,
        )

        byte_identical_path = self._write_bound_summary(
            deepcopy(self.source_summary), "byte-identical-alternate-source.json"
        ).resolve()

        def repoint_identical(payload: dict) -> None:
            payload["budget"]["history_snapshot"]["source_documents"][0][
                "document_path"
            ] = str(byte_identical_path)
            payload["budget"]["history_snapshot_sha256"] = (
                materializer.canonical_sha256(
                    payload["budget"]["history_snapshot"]
                )
            )

        self._assert_rehashed_rejected(
            primary,
            name="byte-identical-alternate-summary-primary.json",
            mutate=repoint_identical,
        )

    def test_alternate_prerequisite_paths_are_rejected_for_v1_and_v2(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(primary)
        overlay = self._overlay(primary, summary)

        alternate_v1 = self.locks_dir / "alternate-source.lock.json"
        shutil.copy2(self.source_lock_path, alternate_v1)

        def repoint_v1(payload: dict) -> None:
            payload["prerequisites"][0]["lock_path"] = str(
                alternate_v1.resolve()
            )
            payload["budget"]["history_snapshot"]["prerequisite_budgets"][0][
                "lock_path"
            ] = str(alternate_v1.resolve())
            payload["budget"]["history_snapshot_sha256"] = (
                materializer.canonical_sha256(
                    payload["budget"]["history_snapshot"]
                )
            )

        self._assert_rehashed_rejected(
            primary,
            name="alternate-v1-prerequisite-primary.json",
            mutate=repoint_v1,
        )

        primary_path = self.locks_dir / "primary.lock.json"
        alternate_v2 = self.locks_dir / "alternate-primary.lock.json"
        shutil.copy2(primary_path, alternate_v2)

        def repoint_v2(payload: dict) -> None:
            reference = payload["prerequisites"][0]
            reference["lock_path"] = str(alternate_v2.resolve())
            snapshot = payload["budget"]["history_snapshot"][
                "prerequisite_budgets"
            ][0]
            snapshot["lock_path"] = str(alternate_v2.resolve())
            payload["budget"]["history_snapshot_sha256"] = (
                materializer.canonical_sha256(
                    payload["budget"]["history_snapshot"]
                )
            )

        self._assert_rehashed_rejected(
            overlay,
            name="alternate-v2-prerequisite-overlay.json",
            mutate=repoint_v2,
        )

        def forge_v2_hash(payload: dict) -> None:
            forged_sha = "f" * 64
            reference = payload["prerequisites"][0]
            old_sha = reference["lock_payload_sha256"]
            reference["lock_payload_sha256"] = forged_sha
            snapshot = payload["budget"]["history_snapshot"][
                "prerequisite_budgets"
            ][0]
            self.assertEqual(snapshot["lock_payload_sha256"], old_sha)
            snapshot["lock_payload_sha256"] = forged_sha
            payload["budget"]["history_snapshot"][
                "prerequisite_budgets"
            ].sort(key=lambda row: row["lock_payload_sha256"])
            payload["budget"]["history_snapshot_sha256"] = (
                materializer.canonical_sha256(
                    payload["budget"]["history_snapshot"]
                )
            )

        self._assert_rehashed_rejected(
            overlay,
            name="forged-v2-prerequisite-hash-overlay.json",
            mutate=forge_v2_hash,
        )

    def test_origin_artifacts_are_pinned_below_trusted_kernel_directory(self) -> None:
        primary = self._primary()
        origin = primary["origins"][0]
        source_root = self.artifacts / origin["kernel_slug"]
        alternate_artifacts = self.root / "alternate-artifacts"
        alternate_root = alternate_artifacts / origin["kernel_slug"]
        shutil.copytree(source_root, alternate_root)
        completion_path = alternate_root / "notebook_completed.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["code_bundle_sha256"] = "f" * 64
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        config_path = next(alternate_root.rglob("training_config.json"))
        iid_path = next(alternate_root.rglob("iid_validation_predictions.parquet"))

        def repoint(payload: dict) -> None:
            changed = payload["origins"][0]
            changed["code_bundle_sha256"] = "f" * 64
            changed["expected_source_sha256"] = "f" * 64
            changed["completion_artifact_path"] = str(completion_path.resolve())
            changed["completion_sha256"] = materializer.file_sha256(
                completion_path
            )
            changed["training_config_artifact_path"] = str(config_path.resolve())
            changed["training_config_artifact_sha256"] = materializer.file_sha256(
                config_path
            )
            changed["iid_predictions_artifact_path"] = str(iid_path.resolve())
            changed["iid_predictions_sha256"] = materializer.file_sha256(iid_path)
            changed["iid_predictions_relative_path"] = str(
                iid_path.relative_to(alternate_root)
            )
            self._refresh_decision_evidence(payload)

        self._assert_rehashed_rejected(
            primary,
            name="alternate-anchor-artifact-root.json",
            mutate=repoint,
        )

        for relative_path, name in (
            (origin["iid_predictions_artifact_path"], "absolute"),
            ("../iid_validation_predictions.parquet", "dotdot"),
        ):
            self._assert_rehashed_rejected(
                primary,
                name=f"unsafe-{name}-iid-relative-path.json",
                mutate=lambda payload, value=relative_path: (
                    payload["origins"][0].__setitem__(
                        "iid_predictions_relative_path", value
                    ),
                    self._refresh_decision_evidence(payload),
                ),
            )

    def test_confirmation_baseline_cannot_repoint_its_artifact_authority(self) -> None:
        chain = self._full_chain()
        confirmation = self._confirmation(chain)
        baseline = next(
            origin
            for origin in confirmation["origins"]
            if origin["experiment"] == self.baseline_row["experiment"]
        )
        source_root = self.artifacts / baseline["kernel_slug"]
        alternate_root = (
            self.root / "alternate-baseline-artifacts" / baseline["kernel_slug"]
        )
        shutil.copytree(source_root, alternate_root)
        completion_path = alternate_root / "notebook_completed.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["code_bundle_sha256"] = "e" * 64
        completion_path.write_text(json.dumps(completion), encoding="utf-8")

        def repoint(payload: dict) -> None:
            changed = next(
                origin
                for origin in payload["origins"]
                if origin["experiment"] == self.baseline_row["experiment"]
            )
            changed["code_bundle_sha256"] = "e" * 64
            changed["expected_source_sha256"] = "e" * 64
            changed["completion_artifact_path"] = str(completion_path.resolve())
            changed["completion_sha256"] = materializer.file_sha256(
                completion_path
            )
            changed["training_config_artifact_path"] = str(
                next(alternate_root.rglob("training_config.json")).resolve()
            )
            changed["iid_predictions_artifact_path"] = str(
                next(
                    alternate_root.rglob("iid_validation_predictions.parquet")
                ).resolve()
            )
            self._refresh_decision_evidence(payload)

        self._assert_rehashed_rejected(
            confirmation,
            name="alternate-confirmation-baseline-artifacts.json",
            mutate=repoint,
        )

    def test_fixed_manifest_controls_replay_and_archive_integrity(self) -> None:
        primary = self._primary()
        trusted = self.trusted_contexts[primary["lock_payload_sha256"]]
        primary_path = self.locks_dir / "primary.lock.json"
        manifest_path = materializer.trusted_provenance_manifest_path(primary_path)
        manifest_text = manifest_path.read_text(encoding="utf-8")
        self.assertEqual(
            manifest_text,
            materializer.canonical_json_dumps(trusted) + "\n",
        )
        self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o444)
        self.assertEqual(
            materializer._write_trusted_provenance_once(
                primary_path, trusted, plan=self.plan
            ),
            manifest_path,
        )
        self.assertEqual(manifest_path.read_text(encoding="utf-8"), manifest_text)

        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError,
            "caller-supplied trusted provenance",
        ):
            materializer.read_lock(primary_path, plan=self.plan)
        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError,
            "caller-supplied trusted provenance",
        ):
            materializer.load_provenance_lock(primary_path, plan=self.plan)

        invalid_manifest = deepcopy(trusted)
        invalid_manifest["artifacts_dir"] = str(self.root / "other-artifacts")
        invalid_manifest_path = self.root / "invalid-trusted-manifest.json"
        invalid_manifest_path.write_text(
            materializer.canonical_json_dumps(invalid_manifest) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError,
            "identity/hash differs",
        ):
            materializer.load_trusted_provenance(
                invalid_manifest_path, plan=self.plan
            )

        noncanonical_manifest_path = self.root / "noncanonical-trusted-manifest.json"
        noncanonical_manifest_path.write_text(
            json.dumps(trusted, indent=2), encoding="utf-8"
        )
        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError,
            "not canonical",
        ):
            materializer.load_trusted_provenance(
                noncanonical_manifest_path, plan=self.plan
            )

        alternate_summary = self._write_bound_summary(
            deepcopy(self.source_summary), "alternate-replay-source.json"
        )
        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError,
            "authorities differ",
        ):
            materializer.materialize_loss_primary_lock(
                plan=self.plan,
                summary=self.source_summary,
                summary_path=alternate_summary,
                artifacts_dir=self.artifacts,
                prerequisite_locks=[self.source_lock],
                prerequisite_lock_paths=[self.source_lock_path],
                history_documents=[],
                history_document_paths=[],
                output_path=primary_path,
            )

        snapshot_path = Path(trusted["source_documents"][0]["snapshot_path"])
        self.assertEqual(snapshot_path.stat().st_mode & 0o777, 0o444)
        snapshot_path.chmod(0o644)
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["runs"][0]["iid_macro_ap"] = 0.123
        snapshot_path.write_text(
            materializer.canonical_json_dumps(snapshot) + "\n",
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError,
            "snapshot path/hash differs",
        ):
            materializer.read_lock(
                primary_path,
                plan=self.plan,
                trusted_provenance=trusted,
            )

    def test_actual_38_union_cannot_be_hidden_by_repointing_confirmation_history(self) -> None:
        chain = self._full_chain()
        confirmation = self._confirmation(chain)
        self.assertEqual(confirmation["budget"]["resulting_unique_kernels"], 37)
        historical_slug = next(
            slug
            for slug in confirmation["budget"]["prior_unique_kernel_slugs"]
            if slug.startswith("prior-kernel-")
        )
        actual_extra_slug = "actual-union-kernel-38"
        actual_union = set(confirmation["budget"]["all_unique_kernel_slugs_after"])
        actual_union.add(actual_extra_slug)
        self.assertEqual(len(actual_union), 38)

        forged_summary = deepcopy(chain["loss_final"])
        forged_summary["runs"].append(
            {"completed": True, "kernel_slug": actual_extra_slug}
        )
        forged_path = self._write_bound_summary(
            forged_summary, "forged-confirmation-history.json"
        ).resolve()

        def hide_one_and_repoint(payload: dict) -> None:
            self._coherently_delete_historical_slug(payload, historical_slug)
            budget = payload["budget"]
            history = budget["history_snapshot"]
            document = next(
                row
                for row in history["source_documents"]
                if row["document_sha256"]
                == payload["decision_inputs_summary_sha256"]
            )
            document["document_path"] = str(forged_path)
            document["document_sha256"] = materializer._summary_document_sha(
                forged_summary
            )
            document["kernel_slugs"] = sorted(
                set(document["kernel_slugs"]) | {actual_extra_slug}
            )
            payload["decision_inputs_summary_sha256"] = document[
                "document_sha256"
            ]
            for key in (
                "prior_unique_kernel_slugs",
                "all_unique_kernel_slugs_after",
            ):
                budget[key] = sorted(set(budget[key]) | {actual_extra_slug})
            history["prior_unique_kernel_slugs"] = sorted(
                set(history["prior_unique_kernel_slugs"]) | {actual_extra_slug}
            )
            budget["prior_unique_kernels"] = len(
                budget["prior_unique_kernel_slugs"]
            )
            budget["resulting_unique_kernels"] = len(
                budget["all_unique_kernel_slugs_after"]
            )
            budget["history_snapshot_sha256"] = materializer.canonical_sha256(
                history
            )
            self.assertEqual(budget["resulting_unique_kernels"], 37)

        self._assert_rehashed_rejected(
            confirmation,
            name="forged-actual-38-confirmation.json",
            mutate=hide_one_and_repoint,
        )

    def test_prediction_and_loss_hook_provenance_are_revalidated(self) -> None:
        primary = self._primary()
        summary, rows = self._primary_summary(primary)
        broken = deepcopy(summary)
        broken_row = next(
            row
            for row in broken["runs"]
            if row["loss_variant"] == "balanced_binary_bce"
        )
        broken_row["iid_predictions_sha256"] = "0" * 64
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "IID predictions SHA differs"
        ):
            self._overlay(primary, broken, name="bad-iid-overlay.lock.json")

        focal = rows["focal_bce_gamma2_scale4"]
        completion_path = self.artifacts / focal["kernel_slug"] / "notebook_completed.json"
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        completion["loss_hook_sha256"] = "0" * 64
        completion_path.write_text(json.dumps(completion), encoding="utf-8")
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "completion loss hook differs"
        ):
            self._overlay(primary, summary, name="bad-hook-overlay.lock.json")

    def test_candidate_source_sampling_weights_and_data_contract_are_revalidated(self) -> None:
        primary = self._primary()
        self.assertEqual(
            {
                variant["expected_source_sha256"]
                for variant in primary["resolved_stage"]["variants"]
            },
            {primary["expected_source_sha256"]},
        )
        summary, rows = self._primary_summary(primary)
        target = rows["balanced_binary_bce"]
        completion_path = (
            self.artifacts / target["kernel_slug"] / "notebook_completed.json"
        )
        original = json.loads(completion_path.read_text(encoding="utf-8"))

        def mixed_source(completion: dict) -> None:
            completion["code_bundle_sha256"] = "f" * 64

        def sampled(completion: dict) -> None:
            completion["training_report"]["training_sampling"] = "category_label"

        def externally_weighted(completion: dict) -> None:
            completion["training_report"]["training_loss_weight_min"] = 0.5

        def changed_data(completion: dict) -> None:
            completion["train_data"]["train_pairs"] = 306_668

        for name, mutate in {
            "mixed-source": mixed_source,
            "sampling": sampled,
            "external-weights": externally_weighted,
            "data-report": changed_data,
        }.items():
            with self.subTest(name=name):
                changed = deepcopy(original)
                mutate(changed)
                completion_path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaises(materializer.AdaptiveMaterializationError):
                    self._overlay(
                        primary,
                        summary,
                        name=f"bad-{name}-overlay.lock.json",
                    )
                completion_path.write_text(json.dumps(original), encoding="utf-8")

    def test_rehashed_origin_training_contract_tamper_is_rejected(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(primary)
        overlay = self._overlay(primary, summary)

        def mutate(lock: dict) -> None:
            origin = next(
                row
                for row in lock["origins"]
                if row["loss_variant"] == "balanced_binary_bce"
            )
            origin["frozen_training_contract"]["training_report"][
                "training_loss_weighting"
            ] = "external"
            origin["frozen_training_contract_sha256"] = (
                materializer.canonical_sha256(origin["frozen_training_contract"])
            )
            self._refresh_decision_evidence(lock)

        self._assert_rehashed_rejected(
            overlay,
            name="tampered-origin-training-contract.json",
            mutate=mutate,
        )

    def test_replay_rejects_different_prerequisite_lock(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(primary)
        lock = self._overlay(primary, summary)
        changed_primary = deepcopy(primary)
        unhashed = dict(changed_primary)
        unhashed.pop("lock_payload_sha256")
        changed_primary["decision"]["seed"] = 7
        unhashed = dict(changed_primary)
        unhashed.pop("lock_payload_sha256", None)
        changed_primary["lock_payload_sha256"] = materializer.canonical_sha256(unhashed)
        summary_path = self.root / "summaries" / "overlay.lock.json.summary.json"
        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError, "different immutable prerequisites"
        ):
            materializer.materialize_loss_overlay_lock(
                plan=self.plan,
                summary=summary,
                summary_path=summary_path,
                artifacts_dir=self.artifacts,
                prerequisite_locks=[changed_primary],
                prerequisite_lock_paths=[self.locks_dir / "primary.lock.json"],
                history_documents=[],
                history_document_paths=[],
                output_path=self.locks_dir / "overlay.lock.json",
            )
        self.assertEqual(lock["mode"], "loss_overlay")

    def test_rehashed_unsafe_or_noncanonical_lock_is_rejected(self) -> None:
        lock = self._primary()
        path = self.locks_dir / "primary.lock.json"
        trusted = self.trusted_contexts[lock["lock_payload_sha256"]]
        payload = deepcopy(lock)
        payload["resolved_stage"]["variants"][0]["resolved_config"]["sampling"] = (
            "category_label"
        )
        unhashed = dict(payload)
        unhashed.pop("lock_payload_sha256")
        payload["lock_payload_sha256"] = materializer.canonical_sha256(unhashed)
        path.write_text(materializer.canonical_json_dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaises(materializer.ExistingAdaptiveLockConflictError):
            materializer.read_lock(
                path, plan=self.plan, trusted_provenance=trusted
            )

        path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
        with self.assertRaisesRegex(
            materializer.ExistingAdaptiveLockConflictError, "not canonical"
        ):
            materializer.read_lock(
                path, plan=self.plan, trusted_provenance=trusted
            )

    def test_execution_summary_requires_exact_lock_set_and_unique_run_ids(self) -> None:
        primary = self._primary()
        summary, _ = self._primary_summary(primary)
        bad = deepcopy(summary)
        bad["execution_lock_sha256s"] = []
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "exactly the required"
        ):
            materializer.validate_execution_summary(
                bad, plan=self.plan, required_locks=[primary]
            )
        duplicate = deepcopy(summary)
        duplicate["runs"].append(deepcopy(duplicate["runs"][0]))
        with self.assertRaisesRegex(
            materializer.AdaptiveMaterializationError, "duplicate run IDs"
        ):
            materializer.validate_execution_summary(
                duplicate, plan=self.plan, required_locks=[primary]
            )


if __name__ == "__main__":
    unittest.main()
