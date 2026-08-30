from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    path = scripts / "launch_soft_positive_tier_a_kaggle_experiment.py"
    spec = importlib.util.spec_from_file_location("soft_positive_a_launcher_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SoftPositiveTierAKaggleFlowTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.launcher = load_launcher()

    def test_pinned_catalog_and_schedule_cover_every_rule_twenty_times(self) -> None:
        rules, _ = self.launcher.verify_catalog()
        donors = pd.read_parquet(self.launcher.DONORS)
        schedule = self.launcher.build_balanced_rule_schedule(
            donors,
            rules,
            count=self.launcher.RAW_COUNT,
            seed=self.launcher.EXPECTED_SEED,
            two_rule_fraction=0.0,
            semantic_signature_limit=(
                self.launcher.EXPECTED_SEMANTIC_SIGNATURE_LIMIT
            ),
            label_one_fraction=1.0,
        )
        summary = schedule.summary()
        self.assertEqual(
            schedule.schedule_sha256,
            "b171cea4202cce2b9c6ce212656909bd51269c8f7d24ce5f8af58670d6b90f42",
        )
        self.assertEqual(summary["planned_target_counts"], {"0": 0, "1": 6_500})
        self.assertEqual(summary["eligible_rules"], 325)
        self.assertEqual(set(summary["primary_rule_usage"].values()), {20})
        self.assertEqual(summary["scheduled_two_rule_tasks"], 0)

    def test_global_card_count_is_category_agnostic_and_order_insensitive(self) -> None:
        with tempfile.TemporaryDirectory(prefix="soft-a-dedup-") as raw:
            source = Path(raw)
            items = pd.DataFrame(
                [
                    {
                        "id": 1,
                        "name": "Кольцо Ёлка",
                        "attributes": json.dumps(
                            {"Тип": "Кольцо", "Цвет": "Зелёный"},
                            ensure_ascii=False,
                        ),
                        "category": "Галантерея и аксессуары",
                    },
                    {
                        "id": -1,
                        "name": "Кольцо синее",
                        "attributes": json.dumps(
                            {"Тип": "Кольцо", "Цвет": "Синий"},
                            ensure_ascii=False,
                        ),
                        "category": "Галантерея и аксессуары",
                    },
                    {
                        "id": 2,
                        "name": "кольцо елка",
                        "attributes": json.dumps(
                            {"Цвет": "зеленый", "Тип": "кольцо"},
                            ensure_ascii=False,
                        ),
                        "category": "Ювелирные изделия",
                    },
                    {
                        "id": -2,
                        "name": "Кольцо красное",
                        "attributes": json.dumps(
                            {"Тип": "Кольцо", "Цвет": "Красный"},
                            ensure_ascii=False,
                        ),
                        "category": "Ювелирные изделия",
                    },
                ]
            )
            pairs = pd.DataFrame(
                [
                    {"id1": 1, "id2": -1, "target": 1},
                    {"id1": 2, "id2": -2, "target": 1},
                ]
            )
            items.to_parquet(source / "items.parquet", index=False)
            pairs.to_parquet(source / "pairs.parquet", index=False)
            self.assertEqual(
                self.launcher.maximum_globally_unique_pair_count(source), 1
            )


if __name__ == "__main__":
    unittest.main()
