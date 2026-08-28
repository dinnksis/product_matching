#!/usr/bin/env python3
"""Evaluate frozen label-free specialist routing rules without fitting a router."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_selective_specialists import SPLITS, load_split
from src.deterministic_specialist_routing import (
    POLICIES,
    ROUTE_BUDGETS,
    SCORE_MODES,
    estimate_private_t4_runtime,
    expert_assignment,
    routed_scores,
    routing_priority,
    top_budget_mask,
)
from src.selective_specialist_analysis import safe_macro_average_precision


EXPERIMENT = "deterministic_specialist_routing_v1"
PRIVATE_PAIRS = 275_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "preds")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=ROOT / "artifacts" / "catboost2_trust_router_v1" / "validation_feature_cache",
    )
    parser.add_argument(
        "--runtime-measurements",
        type=Path,
        default=ROOT / "reports" / "inference_backend_benchmark_v1" / "winner_summary.csv",
    )
    parser.add_argument(
        "--experiment1-summary",
        type=Path,
        default=ROOT / "reports" / "selective_specialist_oracle_v1" / "oracle_budget_summary.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "reports" / EXPERIMENT
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def macro_ap(frame: pd.DataFrame, score: np.ndarray) -> float:
    value, _ = safe_macro_average_precision(
        frame["target"].to_numpy(dtype=np.float64),
        score,
        frame["category"].astype(str).to_numpy(),
    )
    return value


def baseline_rows(frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
    rows = []
    for split, frame in frames.items():
        bge = frame["bge_probability"].to_numpy(dtype=np.float64)
        minilm = frame["minilm_probability"].to_numpy(dtype=np.float64)
        rumodern = frame["rumodernbert_probability"].to_numpy(dtype=np.float64)
        baseline = macro_ap(frame, bge)
        scores = {
            "full_bge": (bge, 0.0, 0.0, "BGE"),
            "full_bge_minilm_50_50": ((bge + minilm) / 2.0, 1.0, 0.0, "BGE+MiniLM"),
            "full_bge_rumodernbert_50_50": (
                (bge + rumodern) / 2.0,
                0.0,
                1.0,
                "BGE+RuModernBERT",
            ),
            "full_triple_equal": (
                (bge + minilm + rumodern) / 3.0,
                1.0,
                1.0,
                "BGE+MiniLM+RuModernBERT",
            ),
        }
        for name, (score, mini_coverage, ru_coverage, specialist) in scores.items():
            current = macro_ap(frame, score)
            rows.append(
                {
                    "split": split,
                    "routing": name,
                    "specialist": specialist,
                    "route_budget": mini_coverage + ru_coverage,
                    "score_mode": "baseline_fixed",
                    "route_coverage": min(1.0, mini_coverage + ru_coverage),
                    "minilm_coverage": mini_coverage,
                    "rumodernbert_coverage": ru_coverage,
                    "macro_ap": current,
                    "baseline_bge_macro_ap": baseline,
                    "delta_vs_bge": current - baseline,
                    "routing_uses_labels": False,
                }
            )
    return rows


def routing_rows(frames: dict[str, pd.DataFrame]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    equivalence: list[dict[str, Any]] = []
    for split, frame in frames.items():
        baseline = macro_ap(frame, frame["bge_probability"].to_numpy(dtype=np.float64))
        priorities = {
            kind: routing_priority(frame, kind)
            for kind in ("uncertainty_abs", "uncertainty_entropy", "domain_conflict")
        }
        masks: dict[tuple[str, float], np.ndarray] = {}
        for kind, priority in priorities.items():
            for budget in ROUTE_BUDGETS:
                masks[(kind, budget)] = top_budget_mask(
                    priority, budget, frame["id1"], frame["id2"]
                )
        for budget in ROUTE_BUDGETS:
            equal = np.array_equal(
                masks[("uncertainty_abs", budget)],
                masks[("uncertainty_entropy", budget)],
            )
            equivalence.append(
                {
                    "split": split,
                    "route_budget": budget,
                    "abs_entropy_masks_equal": equal,
                }
            )
            if not equal:
                raise AssertionError(f"Abs/entropy route masks differ for {split}/{budget}")

        for policy in POLICIES:
            for budget in ROUTE_BUDGETS:
                routed = masks[(policy.priority, budget)]
                assignment = expert_assignment(frame, routed, policy.expert)
                mini_coverage = float(np.mean(assignment == "minilm"))
                ru_coverage = float(np.mean(assignment == "rumodernbert"))
                for score_mode in SCORE_MODES:
                    score = routed_scores(frame, assignment, score_mode)
                    current = macro_ap(frame, score)
                    rows.append(
                        {
                            "split": split,
                            "routing": policy.name,
                            "priority": policy.priority,
                            "specialist": policy.expert,
                            "route_budget": budget,
                            "score_mode": score_mode,
                            "route_coverage": mini_coverage + ru_coverage,
                            "minilm_coverage": mini_coverage,
                            "rumodernbert_coverage": ru_coverage,
                            "macro_ap": current,
                            "baseline_bge_macro_ap": baseline,
                            "delta_vs_bge": current - baseline,
                            "routed_pairs": int(routed.sum()),
                            "routing_uses_labels": False,
                        }
                    )
    return rows, equivalence


def main_table(
    routing: pd.DataFrame,
    baselines: pd.DataFrame,
    runtime: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    all_rows = pd.concat([baselines, routing], ignore_index=True, sort=False)
    records = []
    keys = ["routing", "specialist", "route_budget", "score_mode"]
    for key, part in all_rows.groupby(keys, sort=False, dropna=False):
        by_split = part.set_index("split")
        if not {"iid", "hard", "ood"}.issubset(by_split.index):
            continue
        iid = by_split.loc["iid"]
        timing = estimate_private_t4_runtime(
            PRIVATE_PAIRS,
            float(iid["minilm_coverage"]),
            float(iid["rumodernbert_coverage"]),
            runtime,
        )
        oracle_expert = {
            "minilm": "minilm",
            "rumodernbert": "rumodernbert",
            "dynamic_conflict": "best_expert",
        }.get(str(key[1]))
        oracle_values: dict[str, float] = {}
        for split in ("iid", "hard", "ood"):
            if oracle_expert is None:
                gain = float("nan")
            else:
                match = oracle.loc[
                    oracle["split"].eq(split)
                    & oracle["route_expert"].eq(oracle_expert)
                    & np.isclose(oracle["route_budget"], float(key[2]))
                ]
                if len(match) != 1:
                    raise ValueError(
                        f"Missing Experiment 1 oracle row for {split}/{oracle_expert}/{key[2]}"
                    )
                gain = float(match.iloc[0]["macro_ap_gain"])
            delta = float(by_split.loc[split, "delta_vs_bge"])
            oracle_values[f"{split}_oracle_gain"] = gain
            oracle_values[f"{split}_oracle_capture"] = (
                delta / gain if np.isfinite(gain) and gain > 0 else float("nan")
            )
        records.append(
            {
                "routing": key[0],
                "specialist": key[1],
                "coverage": float(iid["route_coverage"]),
                "minilm_coverage": float(iid["minilm_coverage"]),
                "rumodernbert_coverage": float(iid["rumodernbert_coverage"]),
                "score_mode": key[3],
                "iid_ap": float(iid["macro_ap"]),
                "iid_delta": float(iid["delta_vs_bge"]),
                "hard_ap": float(by_split.loc["hard", "macro_ap"]),
                "hard_delta": float(by_split.loc["hard", "delta_vs_bge"]),
                "ood_ap": float(by_split.loc["ood", "macro_ap"]),
                "ood_delta": float(by_split.loc["ood", "delta_vs_bge"]),
                **timing,
                **oracle_values,
                "runtime_basis": "one_T4_275k_throughput_extrapolation_using_IID_expert_mix",
                "oof_approved": False,
            }
        )
    return pd.DataFrame(records)


def build_report(table: pd.DataFrame, equivalence: pd.DataFrame) -> str:
    baseline = table.loc[table["routing"].str.startswith("full_")].copy()
    candidate_names = {"uncertainty_abs_minilm", "domain_conflict_minilm"}
    candidate_view = table.loc[
        table["routing"].isin(candidate_names) & table["coverage"].between(0.099, 0.101)
    ]
    lines = [
        "# Deterministic specialist routing",
        "",
        "Маршрутизация не использует IID/Hard/OOD labels. Labels читаются только после "
        "фиксации mask для расчёта macro AP.",
        "",
        "## Leakage status",
        "",
        "Совместимых train/OOF predictions для BGE, MiniLM и RuModernBERT нет. Поэтому "
        "ни одна стратегия или blend weight не может быть утверждена как OOF-selected. "
        "Результаты ниже — frozen validation benchmark заранее объявленных правил, а не "
        "основание для production tuning.",
        "",
        f"`abs(p-0.5)` и entropy дали одинаковые route masks во всех {len(equivalence)} "
        "проверках split × budget.",
        "Empirically difficult score region и specialist slices из Experiment 1 не "
        "использовались: они были найдены по IID/Hard/OOD labels и нарушили бы текущий запрет.",
        "",
        "## Full-model comparisons",
        "",
        "| routing | IID AP | Hard AP | OOD AP | T4 private estimate | runtime vs BGE |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in baseline.itertuples(index=False):
        lines.append(
            f"| {row.routing} | {row.iid_ap:.6f} | {row.hard_ap:.6f} | "
            f"{row.ood_ap:.6f} | {row.estimated_private_t4_minutes:.1f} min | "
            f"{row.runtime_multiplier_vs_bge:.3f}x |"
        )
    lines.extend(
        [
            "",
            "## Predeclared 10% MiniLM protocol candidates",
            "",
            "| routing | score | IID AP / delta | Hard AP / delta | OOD AP / delta | runtime vs BGE |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in candidate_view.itertuples(index=False):
        lines.append(
            f"| {row.routing} | {row.score_mode} | {row.iid_ap:.6f} / {row.iid_delta:+.6f} | "
            f"{row.hard_ap:.6f} / {row.hard_delta:+.6f} | "
            f"{row.ood_ap:.6f} / {row.ood_delta:+.6f} | "
            f"{row.runtime_multiplier_vs_bge:.3f}x |"
        )
    capture = candidate_view.loc[
        candidate_view["score_mode"].eq("blend_specialist_25")
    ]
    lines.extend(
        [
            "",
            "For the predeclared 25% specialist blend, captured Experiment 1 oracle headroom is:",
            "",
        ]
    )
    for row in capture.itertuples(index=False):
        lines.append(
            f"- `{row.routing}`: IID {row.iid_oracle_capture:.1%}, "
            f"Hard {row.hard_oracle_capture:.1%}, OOD {row.ood_oracle_capture:.1%}."
        )
    lines.extend(
        [
            "",
            "## Runtime interpretation",
            "",
            "T4 estimates use the measured one-T4 throughput and 275k private pairs. They are "
            "not H100 deadline predictions. The reliable portable quantity is the runtime "
            "multiplier relative to BGE; actual end-to-end H100 timing is still required.",
            "",
            "## Selection status",
            "",
            "Approved strategies: **0**. Two protocol candidates for a future neural OOF run "
            "are `uncertainty_abs_minilm` at 10% and `domain_conflict_minilm` at 10%. "
            "They are chosen in advance for simplicity and MiniLM runtime, not from the "
            "validation AP table. Their score mode remains unresolved until OOF predictions exist.",
            "",
            "See `main_table.csv` for every policy/budget/fixed score mode and "
            "`routing_results.csv` for split-level rows.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runtime = pd.read_csv(args.runtime_measurements)
    oracle = pd.read_csv(args.experiment1_summary)
    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, Any] = {}
    for split, filename in SPLITS.items():
        frame, split_sources = load_split(
            args.predictions_root, args.feature_cache, split, filename
        )
        frames[split] = frame
        sources.update(split_sources)

    route_rows, equivalence_rows = routing_rows(frames)
    routes = pd.DataFrame(route_rows)
    baselines = pd.DataFrame(baseline_rows(frames))
    equivalence = pd.DataFrame(equivalence_rows)
    table = main_table(routes, baselines, runtime, oracle)
    candidates = pd.DataFrame(
        [
            {
                "routing": "uncertainty_abs_minilm",
                "budget": 0.10,
                "score_mode": "requires_neural_oof_selection",
                "status": "protocol_candidate_not_approved",
                "selection_basis": "predeclared simplicity and measured MiniLM runtime",
            },
            {
                "routing": "domain_conflict_minilm",
                "budget": 0.10,
                "score_mode": "requires_neural_oof_selection",
                "status": "protocol_candidate_not_approved",
                "selection_basis": "predeclared domain prior and measured MiniLM runtime",
            },
        ]
    )

    routes.to_csv(args.output_dir / "routing_results.csv", index=False)
    baselines.to_csv(args.output_dir / "full_model_baselines.csv", index=False)
    table.to_csv(args.output_dir / "main_table.csv", index=False)
    equivalence.to_csv(args.output_dir / "uncertainty_equivalence.csv", index=False)
    candidates.to_csv(args.output_dir / "protocol_candidates.csv", index=False)
    (args.output_dir / "REPORT.md").write_text(
        build_report(table, equivalence), encoding="utf-8"
    )

    manifest = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "completed_at_utc": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "training_performed": False,
        "submission_performed": False,
        "google_sheets_modified": False,
        "learned_router_used": False,
        "routing_uses_validation_labels": False,
        "validation_labels_used_only_for_final_metrics": True,
        "compatible_neural_oof_predictions_available": False,
        "approved_strategy_count": 0,
        "route_budgets": list(ROUTE_BUDGETS),
        "score_modes": list(SCORE_MODES),
        "policies": [policy.__dict__ for policy in POLICIES],
        "domain_conflict_coefficients": {
            "bge_uncertainty": 1.0,
            "identifier_conflict": 0.25,
            "any_numeric_conflict": 0.15,
            "title_dissimilarity": 0.10,
            "low_attribute_overlap": 0.05,
        },
        "runtime": {
            "private_pairs": PRIVATE_PAIRS,
            "measurement_path": str(args.runtime_measurements.resolve()),
            "measurement_sha256": sha256_file(args.runtime_measurements),
            "hardware": "one Tesla T4",
            "warning": "Throughput extrapolation, not an H100 deadline verdict.",
        },
        "experiment1_oracle": {
            "path": str(args.experiment1_summary.resolve()),
            "sha256": sha256_file(args.experiment1_summary),
            "use": "post-routing headroom comparison only; never used to construct masks",
        },
        "sources": sources,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
