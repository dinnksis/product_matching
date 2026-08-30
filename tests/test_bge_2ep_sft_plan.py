from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "configs" / "bge_2ep_sft_hparam_search_v1.json"
DOC_PATH = ROOT / "docs" / "bge-2ep-sft-hparam-search.md"


class Bge2epSftPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        cls.stages = {stage["name"]: stage for stage in cls.plan["stages"]}

    def test_campaign_is_design_only_and_uses_separate_bge_baseline(self) -> None:
        self.assertEqual(self.plan["schema_version"], 1)
        self.assertEqual(self.plan["campaign"], "bge_2ep_sft_oodtrain_v1")
        self.assertEqual(self.plan["execution_support"]["status"], "design_only")
        self.assertFalse(self.plan["execution_support"]["runnable"])
        self.assertEqual(self.plan["reporting"]["primary_sheet"], "sft_exps")
        self.assertIn("new BGE baseline", self.plan["reporting"]["baseline_policy"])
        self.assertIn("not paired", self.plan["reporting"]["cross_campaign_comparison"])

    def test_checkpoint_identity_and_architecture_are_frozen(self) -> None:
        checkpoint = self.plan["initial_checkpoint"]
        self.assertEqual(checkpoint["path"], "model/pretrain_bge_2ep")
        self.assertEqual(checkpoint["parameter_count"], 567755777)
        self.assertEqual(
            checkpoint["architecture"], "XLMRobertaForSequenceClassification"
        )
        self.assertEqual(
            checkpoint["config"],
            {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "vocab_size": 250002,
                "max_position_embeddings": 8194,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1,
                "classifier_dropout": None,
            },
        )
        self.assertEqual(
            checkpoint["files_sha256"]["model.safetensors"],
            "c21ccfcd5de310ca0328620bf8ba09e838dbe3f6394be656bd7fec16ad8377d1",
        )
        self.assertEqual(set(checkpoint["files_sha256"]), {
            "config.json",
            "model.safetensors",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
        })

    def test_former_ood_is_fully_promoted_to_twenty_category_train(self) -> None:
        policy = self.plan["data_policy"]
        sources = policy["train_sources_in_exact_concat_order"]
        self.assertEqual(
            [source["role"] for source in sources],
            ["original_human_train", "former_ood_promoted_to_human_train"],
        )
        self.assertEqual([source["rows"] for source in sources], [306669, 41171])
        self.assertEqual([source["positives"] for source in sources], [80136, 9155])
        self.assertEqual(
            sources[1]["categories"], ["Бытовая техника", "Одежда"]
        )
        augmented = policy["augmented_train"]
        self.assertEqual(augmented["rows"], 347840)
        self.assertEqual(augmented["positives"], 89291)
        self.assertEqual(augmented["categories"], 20)
        self.assertAlmostEqual(
            augmented["positive_rate"], 89291 / 347840, places=15
        )
        self.assertIn(
            "no item ID overlap with IID or hard validation",
            augmented["integrity_checks"],
        )

    def test_only_iid_selects_and_ood_is_a_non_metric_sentinel(self) -> None:
        validation = self.plan["data_policy"]["validation"]
        self.assertEqual(validation["iid"]["rows"], 12000)
        self.assertEqual(validation["iid"]["role"], "primary_selection")
        self.assertEqual(validation["hard"]["rows"], 5814)
        self.assertEqual(validation["hard"]["role"], "diagnostic_only")
        self.assertFalse(validation["ood"]["evaluate"])
        self.assertEqual(validation["ood"]["sheet_metric_value"], -1)
        self.assertEqual(validation["ood"]["sheet_delta_value"], -1)
        self.assertIsNone(validation["ood"]["statistical_fields"])
        self.assertEqual(
            self.plan["reporting"]["required_ood_fields"],
            {"ood_macro_ap": -1, "ood_delta": -1},
        )
        self.assertIsNone(
            self.plan["reporting"]["required_ood_significance_fields"]
        )

    def test_baseline_recipe_uses_safe_two_t4_geometry(self) -> None:
        recipe = self.plan["baseline_recipe"]
        expected = {
            "epochs": 1,
            "batch_size": 8,
            "gradient_accumulation": 12,
            "world_size": 2,
            "effective_batch": 192,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.05,
            "max_length": 384,
            "label_smoothing": 0.0,
            "classifier_dropout": 0.1,
            "max_grad_norm": 0.5,
            "loss_variant": "bce",
            "seed": 42,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(recipe[key], value)
        self.assertEqual(
            recipe["batch_size"]
            * recipe["world_size"]
            * recipe["gradient_accumulation"],
            recipe["effective_batch"],
        )
        self.assertTrue(recipe["gradient_checkpointing"])
        self.assertTrue(recipe["full_fine_tuning"])
        self.assertEqual(recipe["sampling"], "none")
        self.assertEqual(recipe["loss_weighting"], "none")

    def test_memory_preflight_preserves_effective_batch_and_requires_rebaseline(self) -> None:
        preflight = self.plan["physical_batch_preflight"]
        for name in ("default", "optional_promotion", "fallback_after_oom"):
            point = preflight[name]
            self.assertEqual(
                point["batch_size"] * 2 * point["gradient_accumulation"],
                192,
            )
            self.assertTrue(point["gradient_checkpointing"])
        self.assertEqual(
            preflight["optional_promotion"]["minimum_free_vram_headroom_gib"],
            1.0,
        )
        self.assertIn("requires a new baseline", preflight["freeze_rule"])

    def test_lr_stage_is_three_point_log_line_with_one_conditional_boundary(self) -> None:
        stage = self.stages["lr_log_line"]
        self.assertEqual(
            stage["axis"]["learning_rate"], [1e-5, 2e-5, 4e-5]
        )
        points = {
            variant["overrides"]["learning_rate"]: variant
            for variant in stage["variants"]
        }
        self.assertEqual(set(points), {1e-5, 2e-5, 4e-5})
        self.assertEqual(points[2e-5]["role"], "new_campaign_baseline")
        self.assertEqual(
            points[2e-5]["experiment"],
            "bge2_ood_sft_e1_lr2e5_baseline_v1",
        )
        boundary = stage["boundary_extension"]
        self.assertEqual(boundary["max_new_runs"], 1)
        self.assertEqual(boundary["trigger_gain_over_adjacent_point"], 0.002)
        self.assertEqual(boundary["delta_operator"], "strictly_greater_than")
        self.assertEqual(boundary["lower"]["extension"], 5e-6)
        self.assertEqual(boundary["upper"]["extension"], 8e-5)

    def test_epoch_line_stops_at_three_and_has_runtime_gate(self) -> None:
        stage = self.stages["epoch_line"]
        self.assertEqual(stage["reuse_epochs"], [1])
        self.assertEqual(stage["always_run_epochs"], [2])
        self.assertEqual(
            stage["conditional_epoch_3"]["run_if_epoch_2_gain_over_epoch_1"],
            0.002,
        )
        self.assertEqual(
            stage["conditional_epoch_3"][
                "and_projected_pipeline_hours_strictly_less_than"
            ],
            9.0,
        )
        self.assertEqual(stage["maximum_new_runs"], 2)
        self.assertFalse(stage["epoch_4_allowed"])

    def test_only_one_regularization_and_one_balanced_loss_candidate_are_allowed(self) -> None:
        regularization = self.stages["regularization_sanity"]
        self.assertEqual(regularization["axis"]["weight_decay"], [0.01, 0.05])
        self.assertEqual(regularization["maximum_new_runs"], 1)
        self.assertEqual(regularization["run_if_selected_epochs_at_least"], 2)
        self.assertEqual(
            regularization["explicitly_skipped_axes"],
            [
                "effective_batch",
                "warmup_ratio",
                "label_smoothing",
                "classifier_dropout",
                "max_grad_norm",
            ],
        )

        loss = self.stages["special_loss_transfer"]
        self.assertEqual(
            loss["loss_variants"],
            ["bce", "balanced_category_class_sqrt_bce"],
        )
        self.assertEqual(loss["maximum_new_runs"], 1)
        self.assertEqual(loss["accept_candidate_minimum_iid_gain"], 0.002)
        self.assertEqual(
            loss["balance_contract"]["strata"], "20 category x 2 hard classes"
        )
        self.assertAlmostEqual(
            loss["balance_contract"]["expected_augmented_train_weight_min"],
            0.6814300041898296,
        )
        self.assertAlmostEqual(
            loss["balance_contract"]["expected_augmented_train_weight_max"],
            2.753521186636443,
        )
        self.assertFalse(loss["loss_lr_refinement_allowed"])
        self.assertEqual(
            set(loss["excluded_losses"]),
            {
                "balanced_binary_bce",
                "balanced_category_class_bce",
                "focal_bce_gamma2_scale4",
            },
        )

    def test_confirmation_is_one_extra_seed_and_requires_consistent_gain(self) -> None:
        stage = self.stages["confirmation_seed_17"]
        self.assertEqual(stage["seeds"], [17, 42])
        self.assertTrue(stage["seed_42_runs_are_reused"])
        self.assertFalse(stage["seed_2026_allowed"])
        self.assertFalse(stage["seed_ensemble_allowed"])
        self.assertEqual(stage["maximum_new_runs"], 2)
        self.assertEqual(
            stage["conditional_challenger"][
                "seed_42_minimum_iid_gain_over_matched_tuned_bce"
            ],
            0.002,
        )
        self.assertEqual(
            stage["acceptance"]["positive_delta_required_on_every_seed"],
            [17, 42],
        )
        self.assertEqual(stage["acceptance"]["minimum_mean_iid_delta"], 0.002)
        self.assertEqual(stage["ood_metric_value"], -1)

    def test_kernel_budget_is_a_hard_ten_and_optional_slow_stages_are_pruned(self) -> None:
        budget = self.plan["budget"]
        maximums = budget["maximum_by_stage_including_reused_anchors"]
        self.assertEqual(sum(maximums.values()), 10)
        self.assertEqual(budget["maximum_total_kernels"], 10)
        self.assertEqual(
            (budget["typical_total_kernels_low"], budget["typical_total_kernels_high"]),
            (6, 7),
        )
        runtime = self.plan["runtime_policy"]
        self.assertEqual(runtime["execution"], "sequential_only")
        self.assertTrue(runtime["parallel_kernels_forbidden"])
        self.assertEqual(runtime["internal_soft_kernel_limit_hours"], 9.0)
        self.assertEqual(
            runtime["prune_optional_stages_if_baseline_hours_strictly_greater_than"],
            4.0,
        )
        self.assertEqual(
            runtime["stages_pruned_by_slow_baseline"],
            ["regularization_sanity", "special_loss_transfer"],
        )
        self.assertFalse(runtime["ods_or_submission_runtime_work"])

    def test_selection_contract_uses_iid_holm_and_practical_tie(self) -> None:
        selection = self.plan["selection_protocol"]
        self.assertEqual(selection["primary_metric"], "iid_macro_ap")
        self.assertTrue(selection["hard_is_diagnostic_only"])
        self.assertTrue(selection["ood_is_unavailable"])
        self.assertEqual(selection["practical_tie_margin"], 0.002)
        self.assertEqual(selection["delta_operator"], "strictly_greater_than")
        self.assertEqual(
            selection["stage_iid_correction"],
            "holm_all_planned_non_anchor_stage_variants",
        )
        self.assertTrue(selection["fresh_start_each_run"])
        self.assertTrue(selection["optimizer_and_scheduler_reset_each_run"])
        self.assertTrue(selection["continuation_from_candidate_checkpoint_forbidden"])

    def test_documentation_carries_non_negotiable_contract(self) -> None:
        document = DOC_PATH.read_text(encoding="utf-8")
        for required in (
            "bge_2ep_sft_oodtrain_v1",
            "347 840",
            "ood_macro_ap` и `ood_delta` всегда равны `-1`",
            "8 × 2 GPU × accumulation 12",
            "`1e-5`, baseline `2e-5` и `4e-5`",
            "balanced_category_class_sqrt_bce",
            "Жёсткий максимум",
            "**10**",
            "Seed 2026 и seed ensemble не используются",
            "generator, validator, Kaggle launcher",
        ):
            with self.subTest(required=required):
                self.assertIn(required, document)


if __name__ == "__main__":
    unittest.main()
