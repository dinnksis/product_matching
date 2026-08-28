#!/usr/bin/env python
"""Cross-fitted rule-gated CatBoost-1 negative-router experiment."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.catboost1_early_exit import wilson_upper
from src.catboost1_rule_portfolio import (
    apply_policy,
    build_candidate_matrix,
    build_veto_masks,
    select_baseline_policy,
    select_portfolio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/catboost1_rule_portfolio.json"
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smoke", action="store_true", help="Integration check; metrics are invalid")
    return parser.parse_args()


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def configure_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("catboost1_rule_portfolio")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(output_dir / "run.log", encoding="utf-8"),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def summarize_mask(
    route: str,
    selection_factor: float,
    risk_limit: float,
    mask: np.ndarray,
    target: np.ndarray,
    baseline: np.ndarray,
) -> dict[str, Any]:
    accepted = int(mask.sum())
    errors = int(target[mask].sum())
    ucb = float(wilson_upper(errors, accepted)) if accepted else 1.0
    return {
        "route": route,
        "selection_factor": selection_factor,
        "risk_limit": risk_limit,
        "accepted": accepted,
        "coverage": accepted / len(mask),
        "errors": errors,
        "empirical_error": errors / accepted if accepted else 0.0,
        "error_ucb_95": ucb,
        "passes_target_risk": bool(accepted and ucb < risk_limit),
        "verified_coverage": accepted / len(mask) if accepted and ucb < risk_limit else 0.0,
        "added_vs_baseline": int((mask & ~baseline).sum()),
        "removed_vs_baseline": int((baseline & ~mask).sum()),
        "net_accepted_vs_baseline": accepted - int(baseline.sum()),
    }


def write_results(summary: pd.DataFrame, path: Path, smoke: bool) -> None:
    headers = [
        "route", "selection_factor", "risk_limit", "accepted", "coverage",
        "errors", "error_ucb_95", "passes_target_risk", "added_vs_baseline",
        "removed_vs_baseline", "net_accepted_vs_baseline",
    ]
    lines = [
        "# CatBoost-1 rule-portfolio results",
        "",
        "SMOKE MODE: числа не являются результатом эксперимента." if smoke else
        "Все числа — product/component-disjoint cross-fit: правила и пороги каждого held fold выбраны без его меток.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in summary[headers].itertuples(index=False, name=None):
        values = []
        for header, value in zip(headers, row):
            if header in {"coverage", "error_ucb_95"}:
                values.append(f"{float(value):.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    lines.extend([
        "",
        "`added_vs_baseline` — новые пары, а не общий объём срабатывания правил.",
        "Статистики discovery/validation из 60k-каталога не читались; использованы только concept/relation templates.",
        "Google Sheets не изменялась.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    output_dir = args.output_dir or resolve(config["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "_smoke"
    logger = configure_logging(output_dir)
    started = time.perf_counter()
    if args.smoke:
        logger.warning("SMOKE MODE: output metrics are not experiment results")

    source_dir = resolve(config["catboost1_source_dir"])
    cache_dir = source_dir / "feature_cache"
    oof = pd.read_parquet(source_dir / "oof_predictions.parquet")
    base = pd.read_parquet(cache_dir / "base_features.parquet")
    global_rules = sparse.load_npz(cache_dir / "clean_global_rules.npz").tocsr()
    category_rules = sparse.load_npz(cache_dir / "clean_category_rules.npz").tocsr()
    definitions = pd.read_parquet(
        cache_dir / "clean_rule_definitions.parquet",
        columns=["rule_id", "canonical_rule", "concept", "relation", "rule_role"],
    )
    category_vocabulary = json.loads(
        (cache_dir / "category_rule_vocabulary.json").read_text(encoding="utf-8")
    )
    if not (len(oof) == len(base) == global_rules.shape[0] == category_rules.shape[0]):
        raise ValueError("CatBoost OOF predictions and feature cache are not aligned")
    if not np.array_equal(oof["category"].astype(str).to_numpy(), base["category"].astype(str).to_numpy()):
        raise ValueError("Category order mismatch between OOF and feature cache")

    score_column = str(config["score_column"])
    required_oof = {"id1", "id2", "target", "category", "fold", score_column}
    missing = required_oof - set(oof.columns)
    if missing:
        raise ValueError(f"Missing OOF columns: {sorted(missing)}")
    if definitions["rule_role"].ne("RULE_CANDIDATE").any():
        raise ValueError("Feature cache contains non-candidate rule roles")
    if not set(definitions["relation"]).issubset({"different_value", "specificity_difference"}):
        raise ValueError("Feature cache contains forbidden rule relations")

    if args.smoke:
        limit = int(config["smoke_pairs_per_fold"])
        selected = np.concatenate([
            np.flatnonzero(oof["fold"].to_numpy() == fold)[:limit]
            for fold in sorted(oof["fold"].unique())
        ])
        selected.sort()
        oof = oof.iloc[selected].reset_index(drop=True)
        base = base.iloc[selected].reset_index(drop=True)
        global_rules = global_rules[selected]
        category_rules = category_rules[selected]

    logger.info(
        "Building label-free candidates from %s clean 60k templates over %s rows",
        f"{len(definitions):,}", f"{len(oof):,}",
    )
    candidates, catalog = build_candidate_matrix(
        base, global_rules, category_rules, definitions, category_vocabulary
    )
    catalog.to_parquet(output_dir / "rule_candidate_catalog.parquet", index=False)
    catalog.groupby(["source", "scope"], as_index=False).agg(
        candidates=("candidate_index", "size"),
        observations=("label_free_support", "sum"),
    ).to_csv(output_dir / "rule_candidate_source_summary.csv", index=False)
    logger.info(
        "Candidate matrix: rows=%s columns=%s observations=%s",
        f"{candidates.shape[0]:,}", f"{candidates.shape[1]:,}", f"{candidates.nnz:,}",
    )

    target = oof["target"].to_numpy(dtype=np.int8)
    scores = oof[score_column].to_numpy(dtype=np.float64)
    folds = oof["fold"].to_numpy(dtype=np.int8)
    risk_limits = [float(value) for value in config["risk_limits"]]
    selection_factors = [float(value) for value in config["selection_risk_factors"]]
    route_masks: dict[tuple[str, float, float], np.ndarray] = {}
    route_reasons: dict[tuple[str, float, float], np.ndarray] = {}
    policy_rows: list[dict[str, Any]] = []

    for risk in risk_limits:
        baseline_mask = np.zeros(len(oof), dtype=bool)
        baseline_reason = np.full(len(oof), "", dtype=object)
        for held_fold in sorted(np.unique(folds)):
            train = folds != held_fold
            held = ~train
            policy = select_baseline_policy(scores[train], target[train], risk)
            selected = (scores[held] < policy.score_threshold)
            baseline_mask[held] = selected
            baseline_reason[np.flatnonzero(held)[selected]] = "catboost_threshold"
            policy_rows.append({
                "route": "C1A_baseline", "selection_factor": 1.0,
                "target_risk": risk, "selection_risk": risk,
                "held_fold": int(held_fold), **policy.to_dict(),
            })
        route_masks[("C1A_baseline", 1.0, risk)] = baseline_mask
        route_reasons[("C1A_baseline", 1.0, risk)] = baseline_reason

        for factor in selection_factors:
            route = "rule_portfolio_nominal" if factor == 1.0 else f"rule_portfolio_guarded_{factor:g}"
            accepted = np.zeros(len(oof), dtype=bool)
            reasons = np.full(len(oof), "", dtype=object)
            for held_fold in sorted(np.unique(folds)):
                train_idx = np.flatnonzero(folds != held_fold)
                held_idx = np.flatnonzero(folds == held_fold)
                train_base = base.iloc[train_idx].reset_index(drop=True)
                held_base = base.iloc[held_idx].reset_index(drop=True)
                logger.info(
                    "Selecting %s risk=%.4g held_fold=%d on %s calibration rows",
                    route, risk, held_fold, f"{len(train_idx):,}",
                )
                policy = select_portfolio(
                    scores[train_idx], target[train_idx], folds[train_idx],
                    candidates[train_idx], catalog, build_veto_masks(train_base),
                    risk_limit=risk * factor,
                    score_caps=[float(value) for value in config["score_caps"]],
                    seed_fractions=[float(value) for value in config["baseline_seed_fractions"]],
                    minimum_support=int(config["minimum_gate_support"]),
                    minimum_folds=int(config["minimum_calibration_folds"]),
                    maximum_gates=int(config["maximum_gates"]),
                )
                held_mask, held_reasons = apply_policy(
                    policy, scores[held_idx], candidates[held_idx], build_veto_masks(held_base)
                )
                accepted[held_idx] = held_mask
                reasons[held_idx] = held_reasons
                policy_rows.append({
                    "route": route, "selection_factor": factor,
                    "target_risk": risk, "selection_risk": risk * factor,
                    "held_fold": int(held_fold), **policy.to_dict(),
                })
            route_masks[(route, factor, risk)] = accepted
            route_reasons[(route, factor, risk)] = reasons

    summary_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    accepted_oof = oof[["id1", "id2", "target", "category", "fold", score_column]].copy()
    for (route, factor, risk), mask in route_masks.items():
        baseline = route_masks[("C1A_baseline", 1.0, risk)]
        summary_rows.append(summarize_mask(route, factor, risk, mask, target, baseline))
        safe_name = f"accepted__{route}__{factor:g}__{risk:g}".replace(".", "p")
        accepted_oof[safe_name] = mask
        accepted_oof[f"reason__{route}__{factor:g}__{risk:g}".replace(".", "p")] = route_reasons[(route, factor, risk)]
        for category in sorted(pd.unique(oof["category"])):
            category_mask = oof["category"].astype(str).eq(str(category)).to_numpy()
            selected = mask & category_mask
            count = int(selected.sum())
            errors = int(target[selected].sum())
            category_rows.append({
                "route": route, "selection_factor": factor, "risk_limit": risk,
                "category": str(category), "category_pairs": int(category_mask.sum()),
                "accepted": count, "coverage": count / max(1, int(category_mask.sum())),
                "errors": errors, "empirical_error": errors / count if count else 0.0,
            })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["risk_limit", "route", "selection_factor"], kind="stable"
    )
    summary.to_csv(output_dir / "crossfit_summary.csv", index=False)
    pd.DataFrame(category_rows).to_csv(output_dir / "crossfit_category_details.csv", index=False)
    accepted_oof.to_parquet(output_dir / "crossfit_accepted_pairs.parquet", index=False)
    (output_dir / "fold_policies.json").write_text(
        json.dumps(policy_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_results(summary, output_dir / "RESULTS.md", args.smoke)

    leakage_audit = {
        "human_train_only": True,
        "iid_hard_ood_loaded": False,
        "neural_predictions_loaded": False,
        "catboost2_loaded": False,
        "rule_definition_columns_read": [
            "rule_id", "canonical_rule", "concept", "relation", "rule_role",
        ],
        "forbidden_mined_columns_read": [],
        "forbidden_examples": [
            "global_support", "global_effect", "discovery_effect_class",
            "validation_support", "validation_stability", "generation_status",
            "allowed_categories",
        ],
        "held_fold_used_for_rule_or_threshold_selection": False,
        "google_sheets_written": False,
    }
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    manifest = {
        "experiment": "catboost1_rule_portfolio_v1",
        "smoke": args.smoke,
        "source_dir": str(source_dir.resolve()),
        "score_column": score_column,
        "rows": len(oof),
        "candidate_columns": candidates.shape[1],
        "candidate_observations": candidates.nnz,
        "risk_limits": risk_limits,
        "selection_risk_factors": selection_factors,
        "python": platform.python_version(),
        "elapsed_seconds": time.perf_counter() - started,
        "google_sheets_written": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    logger.info("Completed in %.1f minutes", manifest["elapsed_seconds"] / 60)


if __name__ == "__main__":
    main()

