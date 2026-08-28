#!/usr/bin/env python3
"""Frozen BGE+MiniLM coverage sweep using an existing benefit router."""

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
from scripts.train_benefit_routers import bge_prediction_view
from src.benefit_router import router_feature_frame
from src.deterministic_specialist_routing import (
    domain_conflict_priority,
    top_budget_mask,
    uncertainty_abs,
)
from src.selective_specialist_analysis import safe_macro_average_precision


METHODS = ("learned_classification", "uncertainty", "conflict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "minilm_coverage_sweep.json"
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def macro_ap(frame: pd.DataFrame, score: np.ndarray) -> float:
    return safe_macro_average_precision(
        frame["target"].to_numpy(dtype=np.int8),
        np.asarray(score, dtype=np.float64),
        frame["category"].astype(str).to_numpy(),
    )[0]


def blended_score(frame: pd.DataFrame, routed: np.ndarray, weight: float) -> np.ndarray:
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    mini = frame["minilm_probability"].to_numpy(dtype=np.float64)
    result = bge.copy()
    result[routed] = (1.0 - weight) * bge[routed] + weight * mini[routed]
    return result


def priority(frame: pd.DataFrame, method: str) -> np.ndarray:
    if method == "learned_classification":
        return frame["benefit_minilm_classification"].to_numpy(dtype=np.float64)
    if method == "uncertainty":
        return uncertainty_abs(frame["bge_probability"])
    if method == "conflict":
        return domain_conflict_priority(frame)
    raise ValueError(f"Unknown method: {method}")


def load_oof(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    artifact_dir = resolve(config["router_artifact_dir"])
    oof_path = artifact_dir / "router_oof_predictions.parquet"
    pairs_path = resolve(config["human_train_pairs_path"])
    features_path = resolve(config["train_feature_cache_path"])
    oof = pd.read_parquet(oof_path)
    required = {
        "id1", "id2", "target", "category", "bge_probability",
        "minilm_probability", "benefit_minilm_classification",
    }
    missing = required - set(oof.columns)
    if missing:
        raise ValueError(f"Router OOF is missing columns: {sorted(missing)}")
    pairs = pd.read_parquet(pairs_path, columns=["id1", "id2", "target"])
    cheap = pd.read_parquet(features_path)
    if len(oof) != len(pairs) or len(oof) != len(cheap):
        raise ValueError("OOF, human pairs, and cheap features differ in size")
    if not pairs[["id1", "id2", "target"]].reset_index(drop=True).equals(
        oof[["id1", "id2", "target"]].reset_index(drop=True)
    ):
        raise ValueError("Router OOF is not aligned to human train")
    frame = pd.concat(
        [oof.reset_index(drop=True), cheap.drop(columns="category").reset_index(drop=True)],
        axis=1,
    )
    sources = {
        "router_oof": {"path": str(oof_path.resolve()), "sha256": sha256_file(oof_path)},
        "human_pairs": {"path": str(pairs_path.resolve()), "sha256": sha256_file(pairs_path)},
        "cheap_features": {"path": str(features_path.resolve()), "sha256": sha256_file(features_path)},
    }
    return frame, sources


def select_oof_policies(config: dict[str, Any], frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    bge_score = frame["bge_probability"].to_numpy(dtype=np.float64)
    bge_ap = macro_ap(frame, bge_score)
    all_mask = np.ones(len(frame), dtype=bool)
    full_candidates = []
    for weight in config["specialist_weights"]:
        score = blended_score(frame, all_mask, float(weight))
        full_candidates.append((macro_ap(frame, score), float(weight)))
    full_ap, full_weight = max(full_candidates, key=lambda item: (item[0], -item[1]))

    rows: list[dict[str, Any]] = []
    budgets_by_method = {
        "learned_classification": config["learned_budgets"],
        "uncertainty": config["baseline_budgets"],
        "conflict": config["baseline_budgets"],
    }
    for method, budgets in budgets_by_method.items():
        routing_priority = priority(frame, method)
        for budget in budgets:
            routed = top_budget_mask(
                routing_priority, float(budget), frame["id1"], frame["id2"]
            )
            candidates = []
            for weight in config["specialist_weights"]:
                score = blended_score(frame, routed, float(weight))
                candidates.append((macro_ap(frame, score), float(weight)))
            ap, weight = max(candidates, key=lambda item: (item[0], -item[1]))
            rows.append(
                {
                    "method": method,
                    "coverage": float(budget),
                    "specialist_weight": weight,
                    "score_mode": "replace" if weight == 1.0 else f"blend_minilm_{weight:.2f}",
                    "oof_macro_ap": ap,
                    "oof_delta_vs_bge": ap - bge_ap,
                    "oof_delta_vs_full_ensemble": ap - full_ap,
                    "routed_pairs": int(routed.sum()),
                    "parameters_selected_on_oof": True,
                }
            )
    anchor = {
        "bge_oof_ap": bge_ap,
        "full_ensemble_oof_ap": full_ap,
        "full_ensemble_specialist_weight": full_weight,
    }
    return pd.DataFrame(rows), anchor


def validation_features(
    config: dict[str, Any],
    split: str,
    filename: str,
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical: list[str],
) -> pd.DataFrame:
    reserved = {
        "id1", "id2", "target", "category", "bge_probability",
        "minilm_probability", "bge_token_length_max",
    }
    cheap_columns = [column for column in frame.columns if column not in reserved]
    cheap = frame[["category", *cheap_columns]].copy()
    bge_path = resolve(config["predictions_root"]) / "preds_bge" / filename
    bge_raw = pd.read_parquet(
        bge_path,
        columns=[
            "id1", "id2", "score", "logit", "score_order_gap",
            "token_length_ab", "token_length_ba",
        ],
    )
    if not bge_raw[["id1", "id2"]].reset_index(drop=True).equals(
        frame[["id1", "id2"]].reset_index(drop=True)
    ):
        raise ValueError(f"Raw BGE predictions differ on {split}")
    features, actual_categorical = router_feature_frame(cheap, bge_prediction_view(bge_raw))
    if actual_categorical != categorical:
        raise ValueError(f"Categorical schema differs on {split}")
    missing = set(feature_columns) - set(features.columns)
    extra = set(features.columns) - set(feature_columns)
    if missing or extra:
        raise ValueError(f"Feature schema differs on {split}: missing={missing}, extra={extra}")
    return features[feature_columns]


def evaluate_validation(
    config: dict[str, Any], policies: pd.DataFrame, anchor: dict[str, float]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    from catboost import CatBoostClassifier, Pool

    artifact_dir = resolve(config["router_artifact_dir"])
    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    feature_columns = list(manifest["feature_columns"])
    categorical = list(manifest["categorical_columns"])
    model_path = artifact_dir / "models" / "router_minilm_classification.cbm"
    model = CatBoostClassifier()
    model.load_model(model_path)
    rows: list[dict[str, Any]] = []
    sources: dict[str, Any] = {
        "router_model": {"path": str(model_path.resolve()), "sha256": sha256_file(model_path)}
    }
    for split, filename in SPLITS.items():
        frame, split_sources = load_split(
            resolve(config["predictions_root"]),
            resolve(config["validation_feature_cache_dir"]),
            split,
            filename,
            specialists=("minilm",),
        )
        sources.update(split_sources)
        features = validation_features(
            config, split, filename, frame, feature_columns, categorical
        )
        frame["benefit_minilm_classification"] = model.predict_proba(
            Pool(features, cat_features=categorical)
        )[:, 1]
        bge = frame["bge_probability"].to_numpy(dtype=np.float64)
        all_mask = np.ones(len(frame), dtype=bool)
        full = blended_score(
            frame, all_mask, float(anchor["full_ensemble_specialist_weight"])
        )
        baselines = {
            "bge": (bge, 0.0, 0.0),
            "full_ensemble": (
                full, 1.0, float(anchor["full_ensemble_specialist_weight"])
            ),
        }
        for method, (score, coverage, weight) in baselines.items():
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "coverage": coverage,
                    "specialist_weight": weight,
                    "macro_ap": macro_ap(frame, score),
                }
            )
        for policy in policies.itertuples(index=False):
            routed = top_budget_mask(
                priority(frame, policy.method),
                float(policy.coverage),
                frame["id1"],
                frame["id2"],
            )
            score = blended_score(frame, routed, float(policy.specialist_weight))
            rows.append(
                {
                    "split": split,
                    "method": policy.method,
                    "coverage": float(policy.coverage),
                    "specialist_weight": float(policy.specialist_weight),
                    "macro_ap": macro_ap(frame, score),
                }
            )
    return pd.DataFrame(rows), sources


def build_quality_table(
    policies: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    anchor = validation.loc[validation["method"].eq("full_ensemble")].set_index("split")
    baseline = validation.loc[validation["method"].isin(["bge", "full_ensemble"])].copy()
    records: list[dict[str, Any]] = []
    for policy in policies.itertuples(index=False):
        part = validation.loc[
            validation["method"].eq(policy.method)
            & np.isclose(validation["coverage"], float(policy.coverage))
        ].set_index("split")
        values = {split: float(part.loc[split, "macro_ap"]) for split in SPLITS}
        full_values = {split: float(anchor.loc[split, "macro_ap"]) for split in SPLITS}
        mean_ap = float(np.mean(list(values.values())))
        full_mean = float(np.mean(list(full_values.values())))
        deltas = {split: values[split] - full_values[split] for split in SPLITS}
        records.append(
            {
                "method": policy.method,
                "coverage": float(policy.coverage),
                "specialist_weight": float(policy.specialist_weight),
                "score_mode": policy.score_mode,
                "oof_macro_ap": float(policy.oof_macro_ap),
                "iid_macro_ap": values["iid"],
                "hard_macro_ap": values["hard"],
                "ood_macro_ap": values["ood"],
                "mean_macro_ap": mean_ap,
                "iid_delta_vs_full": deltas["iid"],
                "hard_delta_vs_full": deltas["hard"],
                "ood_delta_vs_full": deltas["ood"],
                "mean_delta_vs_full": mean_ap - full_mean,
                "worst_split_delta_vs_full": min(deltas.values()),
                "minilm_stage_fraction": float(policy.coverage),
                "minilm_stage_saved_pct": 100.0 * (1.0 - float(policy.coverage)),
                "absolute_private_runtime": np.nan,
                "runtime_basis": "coverage_only; final H100 stage timings unavailable",
                "is_previous_best_learned_15pct": bool(
                    policy.method == "learned_classification"
                    and np.isclose(policy.coverage, 0.15)
                ),
            }
        )
    table = pd.DataFrame(records)
    baseline_records = []
    for method, part in baseline.groupby("method"):
        by_split = part.set_index("split")
        values = [float(by_split.loc[split, "macro_ap"]) for split in SPLITS]
        baseline_records.append(
            {
                "method": method,
                "coverage": 0.0 if method == "bge" else 1.0,
                "iid_macro_ap": values[0],
                "hard_macro_ap": values[1],
                "ood_macro_ap": values[2],
                "mean_macro_ap": float(np.mean(values)),
            }
        )
    return table, pd.DataFrame(baseline_records)


def operating_points(
    config: dict[str, Any], table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    learned = table.loc[table["method"].eq("learned_classification")].sort_values("coverage")
    points = []
    for tolerance in config["mean_loss_tolerances"]:
        for scope, metric in (
            ("mean_ap", "mean_delta_vs_full"),
            ("every_split", "worst_split_delta_vs_full"),
        ):
            eligible = learned.loc[learned[metric] >= -float(tolerance)]
            if eligible.empty:
                points.append(
                    {"scope": scope, "tolerance": tolerance, "status": "not_reached"}
                )
                continue
            row = eligible.iloc[0]
            points.append(
                {
                    "scope": scope,
                    "tolerance": float(tolerance),
                    "status": "reached",
                    "minimum_coverage": float(row["coverage"]),
                    "specialist_weight": float(row["specialist_weight"]),
                    "mean_macro_ap": float(row["mean_macro_ap"]),
                    "mean_delta_vs_full": float(row["mean_delta_vs_full"]),
                    "worst_split_delta_vs_full": float(row["worst_split_delta_vs_full"]),
                    "minilm_stage_saved_pct": float(row["minilm_stage_saved_pct"]),
                }
            )
    pareto_mask = []
    for row in learned.itertuples(index=False):
        dominated = (
            (learned["coverage"] <= row.coverage)
            & (learned["mean_macro_ap"] >= row.mean_macro_ap)
            & (
                (learned["coverage"] < row.coverage)
                | (learned["mean_macro_ap"] > row.mean_macro_ap)
            )
        ).any()
        pareto_mask.append(not dominated)
    return pd.DataFrame(points), learned.loc[pareto_mask].copy()


def build_report(
    table: pd.DataFrame, baselines: pd.DataFrame, points: pd.DataFrame, pareto: pd.DataFrame
) -> str:
    learned = table.loc[table["method"].eq("learned_classification")]
    lines = [
        "# MiniLM coverage sweep",
        "",
        "All routing masks and blend weights were selected on human-train neural OOF. "
        "IID/Hard/OOD labels were loaded only after policy selection.",
        "",
        "Absolute runtime is intentionally omitted: the final production implementation uses "
        "SentenceTransformers, SDPA, FP16, max_length=192, batch=1024 and one-direction inference, "
        "while the frozen quality predictions use the earlier 384-token symmetric protocol.",
        "",
        "| coverage | MiniLM weight | IID | Hard | OOD | mean | delta vs full mean | MiniLM stage saved |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in learned.itertuples(index=False):
        lines.append(
            f"| {row.coverage:.0%} | {row.specialist_weight:.0%} | {row.iid_macro_ap:.6f} | "
            f"{row.hard_macro_ap:.6f} | {row.ood_macro_ap:.6f} | {row.mean_macro_ap:.6f} | "
            f"{row.mean_delta_vs_full:+.6f} | {row.minilm_stage_saved_pct:.0f}% |"
        )
    lines.extend(["", "## Operating points", ""])
    for row in points.itertuples(index=False):
        if row.status != "reached":
            lines.append(f"- {row.scope}, loss <= {row.tolerance:.3f}: not reached")
        else:
            lines.append(
                f"- {row.scope}, loss <= {row.tolerance:.3f}: {row.minimum_coverage:.0%} coverage, "
                f"mean delta {row.mean_delta_vs_full:+.6f}, worst split {row.worst_split_delta_vs_full:+.6f}."
            )
    lines.extend(
        [
            "",
            "Pareto learned coverages: "
            + ", ".join(f"{value:.0%}" for value in pareto["coverage"]),
            "",
            "Runtime conversion after final H100 logs: T = T_fixed + T_BGE + T_MiniLM_load "
            "+ coverage * T_MiniLM_inference.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or resolve(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    oof, sources = load_oof(config)
    policies, anchor = select_oof_policies(config, oof)
    policies.to_csv(output_dir / "oof_policy_selection.csv", index=False)
    (output_dir / "frozen_policy.json").write_text(
        json.dumps(
            {"anchor": anchor, "policies": policies.to_dict("records")},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    # Protocol boundary: validation labels are not read before the files above exist.
    validation, validation_sources = evaluate_validation(config, policies, anchor)
    validation.to_csv(output_dir / "split_metrics.csv", index=False)
    table, baselines = build_quality_table(policies, validation)
    table.to_csv(output_dir / "quality_runtime_table.csv", index=False)
    baselines.to_csv(output_dir / "baselines.csv", index=False)
    points, pareto = operating_points(config, table)
    points.to_csv(output_dir / "operating_points.csv", index=False)
    pareto.to_csv(output_dir / "pareto_learned.csv", index=False)
    (output_dir / "REPORT.md").write_text(
        build_report(table, baselines, points, pareto), encoding="utf-8"
    )
    manifest = {
        "status": "complete",
        "experiment": "minilm_coverage_sweep_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "training_performed": False,
        "submission_performed": False,
        "oof_parameters_frozen_before_validation": True,
        "runtime_protocol": config["runtime_protocol"],
        "quality_protocol": "saved max_length=384 symmetric probability predictions",
        "production_parity_warning": "final submit uses max_length=192 one-direction raw-logit rank ensemble",
        "sources": {**sources, **validation_sources},
        "outputs": {
            "table": "quality_runtime_table.csv",
            "operating_points": "operating_points.csv",
            "pareto": "pareto_learned.csv",
            "report": "REPORT.md",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "output_dir": str(output_dir),
        "full_ensemble_weight": anchor["full_ensemble_specialist_weight"],
        "operating_points": points.to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
