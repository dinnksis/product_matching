#!/usr/bin/env python3
"""Train component-disjoint benefit routers, freeze choices on OOF, then evaluate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_selective_specialists import SPLITS, load_split
from src.benefit_router import (
    assert_component_disjoint,
    assert_pair_ids_disjoint,
    benefit_targets,
    deterministic_random_priority,
    router_feature_frame,
)
from src.deterministic_specialist_routing import (
    domain_conflict_priority,
    estimate_private_t4_runtime,
    top_budget_mask,
    uncertainty_abs,
)
from src.selective_specialist_analysis import safe_macro_average_precision


AVAILABLE_SPECIALISTS = ("minilm", "rumodernbert")
TARGET_KINDS = ("classification", "regression")
EXPERIMENT = "benefit_router_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "benefit_router.json"
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use two folds, at most 6000 train rows, and 10 CatBoost iterations.",
    )
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


def macro_ap(target: np.ndarray, score: np.ndarray, category: np.ndarray) -> float:
    return safe_macro_average_precision(target, score, category)[0]


def checked_oof(path: Path, model: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {model} neural OOF predictions: {path}\n"
            "Run/download the three benefit-router OOF Kaggle notebooks first."
        )
    frame = pd.read_parquet(path)
    required = {
        "id1", "id2", "target", "category", "component_id", "fold",
        "oof_row_index", "score",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{model} OOF is missing columns: {sorted(missing)}")
    if frame[["id1", "id2"]].duplicated().any():
        raise ValueError(f"{model} OOF contains duplicate pairs")
    score = frame["score"].to_numpy(dtype=np.float64)
    if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
        raise ValueError(f"{model} OOF contains invalid probabilities")
    return frame.sort_values("oof_row_index", kind="stable").reset_index(drop=True)


def load_train(
    config: dict[str, Any], smoke: bool, specialists: tuple[str, ...]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    predictions = {
        model: checked_oof(resolve(config["oof_predictions"][model]), model)
        for model in ("bge", *specialists)
    }
    reference = predictions["bge"].copy()
    keys = ["id1", "id2", "target", "category", "component_id", "fold"]
    for specialist in specialists:
        current = predictions[specialist]
        for column in keys:
            if not np.array_equal(
                reference[column].astype(str).to_numpy(),
                current[column].astype(str).to_numpy(),
            ):
                raise ValueError(f"{specialist} OOF differs from BGE in {column}")
        reference[f"{specialist}_probability"] = current["score"].to_numpy(dtype=np.float32)
    reference = reference.rename(columns={"score": "bge_probability"})

    pairs = pd.read_parquet(resolve(config["human_train_pairs_path"]))
    cheap = pd.read_parquet(resolve(config["train_feature_cache_path"]))
    if len(pairs) != len(cheap):
        raise ValueError("Human pairs and cheap features differ in size")
    reference = pairs[["id1", "id2", "target"]].merge(
        reference,
        on=["id1", "id2", "target"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    if reference["bge_probability"].isna().any():
        raise ValueError("Neural OOF pair set differs from human train")
    assert_component_disjoint(reference["fold"], reference["component_id"])
    assert_pair_ids_disjoint(reference[["id1", "id2", "fold"]])

    if smoke:
        selected_folds = sorted(reference["fold"].unique())[:2]
        mask = reference["fold"].isin(selected_folds).to_numpy()
        chosen = np.flatnonzero(mask)[:6000]
        reference = reference.iloc[chosen].reset_index(drop=True)
        cheap = cheap.iloc[chosen].reset_index(drop=True)
        remap = {fold: index for index, fold in enumerate(sorted(reference["fold"].unique()))}
        reference["fold"] = reference["fold"].map(remap).astype(np.int8)
    return reference, cheap


def bge_prediction_view(frame: pd.DataFrame) -> pd.DataFrame:
    score_column = "bge_probability" if "bge_probability" in frame else "score"
    if score_column not in frame:
        raise ValueError("BGE prediction frame has neither bge_probability nor score")
    result = pd.DataFrame({"score": frame[score_column]})
    for source in (
        "logit",
        "score_order_gap",
        "token_length_ab",
        "token_length_ba",
    ):
        if source in frame:
            result[source] = frame[source]
    if "bge_token_length_max" in frame and "token_length_ab" not in result:
        result["token_length_ab"] = frame["bge_token_length_max"]
        result["token_length_ba"] = frame["bge_token_length_max"]
    return result


def catboost_parameters(config: dict[str, Any], kind: str, smoke: bool) -> dict[str, Any]:
    parameters = dict(config["catboost"])
    if smoke:
        parameters["iterations"] = 10
        parameters["verbose"] = False
    parameters.update(
        {
            "loss_function": "Logloss" if kind == "classification" else "RMSE",
            "task_type": "CPU",
            "thread_count": int(config["threads"]),
            "random_seed": int(config["seed"]),
            "allow_writing_files": False,
        }
    )
    return parameters


def fit_routers(
    config: dict[str, Any],
    frame: pd.DataFrame,
    features: pd.DataFrame,
    categorical: list[str],
    output_dir: Path,
    smoke: bool,
    specialists: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[tuple[str, str], Any], pd.DataFrame]:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool

    folds = frame["fold"].to_numpy(dtype=np.int8)
    targets: dict[tuple[str, str], np.ndarray] = {}
    target_summary = []
    for specialist in specialists:
        regression, classification = benefit_targets(
            frame["target"],
            frame["bge_probability"],
            frame[f"{specialist}_probability"],
            classification_margin=float(config["classification_margin_logloss"]),
        )
        targets[(specialist, "regression")] = regression
        targets[(specialist, "classification")] = classification
        target_summary.append(
            {
                "specialist": specialist,
                "rows": len(frame),
                "positive_benefit_rate": float(np.mean(regression > 0)),
                "classification_positive_rate": float(np.mean(classification)),
                "mean_benefit": float(np.mean(regression)),
            }
        )

    predictions = frame[
        ["id1", "id2", "target", "category", "component_id", "fold", "bge_probability", *[
            f"{specialist}_probability" for specialist in specialists
        ]]
    ].copy()
    models: dict[tuple[str, str], Any] = {}
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for specialist in specialists:
        for kind in TARGET_KINDS:
            target = targets[(specialist, kind)]
            oof = np.full(len(frame), np.nan, dtype=np.float64)
            model_class = CatBoostClassifier if kind == "classification" else CatBoostRegressor
            for fold in sorted(np.unique(folds)):
                train_idx = np.flatnonzero(folds != fold)
                valid_idx = np.flatnonzero(folds == fold)
                model = model_class(**catboost_parameters(config, kind, smoke))
                model.fit(
                    Pool(features.iloc[train_idx], target[train_idx], cat_features=categorical)
                )
                held_pool = Pool(features.iloc[valid_idx], cat_features=categorical)
                if kind == "classification":
                    oof[valid_idx] = model.predict_proba(held_pool)[:, 1]
                else:
                    oof[valid_idx] = model.predict(held_pool)
            if not np.isfinite(oof).all():
                raise RuntimeError(f"Incomplete router OOF: {specialist}/{kind}")
            predictions[f"benefit_{specialist}_{kind}"] = oof.astype(np.float32)
            full_model = model_class(**catboost_parameters(config, kind, smoke))
            full_model.fit(Pool(features, target, cat_features=categorical))
            full_model.save_model(models_dir / f"router_{specialist}_{kind}.cbm")
            models[(specialist, kind)] = full_model
    return predictions, models, pd.DataFrame(target_summary)


def route_priority(frame: pd.DataFrame, method: str, specialist: str | None, seed: int) -> np.ndarray:
    if method == "random":
        return deterministic_random_priority(frame["id1"], frame["id2"], seed)
    if method == "uncertainty":
        return uncertainty_abs(frame["bge_probability"])
    if method == "simple_slice":
        return domain_conflict_priority(frame)
    if method.startswith("learned_") and specialist is not None:
        kind = method.removeprefix("learned_")
        return frame[f"benefit_{specialist}_{kind}"].to_numpy(dtype=np.float64)
    raise ValueError(f"Unsupported routing method: {method}")


def blended_score(
    frame: pd.DataFrame, mask: np.ndarray, specialist: str, weight: float
) -> np.ndarray:
    bge = frame["bge_probability"].to_numpy(dtype=np.float64)
    specialist_score = frame[f"{specialist}_probability"].to_numpy(dtype=np.float64)
    result = bge.copy()
    result[mask] = (1.0 - weight) * bge[mask] + weight * specialist_score[mask]
    return result


def select_oof_policies(
    config: dict[str, Any], frame: pd.DataFrame, specialists: tuple[str, ...]
) -> pd.DataFrame:
    target = frame["target"].to_numpy(dtype=np.int8)
    category = frame["category"].astype(str).to_numpy()
    baseline = macro_ap(target, frame["bge_probability"].to_numpy(), category)
    rows: list[dict[str, Any]] = []
    methods = ("random", "uncertainty", "simple_slice", "learned_classification", "learned_regression")
    for method in methods:
        for specialist in specialists:
            priority = route_priority(frame, method, specialist, int(config["seed"]))
            for budget in config["budgets"]:
                mask = top_budget_mask(priority, float(budget), frame["id1"], frame["id2"])
                candidates = []
                for weight in config["blend_weights"]:
                    score = blended_score(frame, mask, specialist, float(weight))
                    candidates.append((macro_ap(target, score, category), float(weight)))
                ap, weight = max(candidates, key=lambda value: (value[0], -value[1]))
                rows.append(
                    {
                        "method": method,
                        "specialist": specialist,
                        "budget": float(budget),
                        "blend_weight": weight,
                        "oof_macro_ap": ap,
                        "oof_delta_vs_bge": ap - baseline,
                        "route_coverage": float(mask.mean()),
                    }
                )

    # Multi-expert policies use the larger predicted benefit and one shared OOF-selected blend.
    for kind in TARGET_KINDS if len(specialists) > 1 else ():
        method = f"learned_multi_{kind}"
        mini = frame[f"benefit_minilm_{kind}"].to_numpy(dtype=np.float64)
        ru = frame[f"benefit_rumodernbert_{kind}"].to_numpy(dtype=np.float64)
        priority = np.maximum(mini, ru)
        expert = np.where(mini >= ru, "minilm", "rumodernbert")
        for budget in config["budgets"]:
            mask = top_budget_mask(priority, float(budget), frame["id1"], frame["id2"])
            candidates = []
            for weight in config["blend_weights"]:
                score = frame["bge_probability"].to_numpy(dtype=np.float64).copy()
                for specialist in specialists:
                    selected = mask & (expert == specialist)
                    specialist_score = frame[f"{specialist}_probability"].to_numpy(dtype=np.float64)
                    score[selected] = (1.0 - float(weight)) * score[selected] + float(weight) * specialist_score[selected]
                candidates.append((macro_ap(target, score, category), float(weight)))
            ap, weight = max(candidates, key=lambda value: (value[0], -value[1]))
            rows.append(
                {
                    "method": method,
                    "specialist": "dynamic_one_expert",
                    "budget": float(budget),
                    "blend_weight": weight,
                    "oof_macro_ap": ap,
                    "oof_delta_vs_bge": ap - baseline,
                    "route_coverage": float(mask.mean()),
                    "minilm_coverage": float(np.mean(mask & (expert == "minilm"))),
                    "rumodernbert_coverage": float(np.mean(mask & (expert == "rumodernbert"))),
                }
            )
    return pd.DataFrame(rows)


def predict_router_scores(
    models: dict[tuple[str, str], Any],
    features: pd.DataFrame,
    categorical: list[str],
    specialists: tuple[str, ...],
) -> pd.DataFrame:
    from catboost import Pool

    pool = Pool(features, cat_features=categorical)
    result = pd.DataFrame(index=features.index)
    for specialist in specialists:
        for kind in TARGET_KINDS:
            model = models[(specialist, kind)]
            if kind == "classification":
                score = model.predict_proba(pool)[:, 1]
            else:
                score = model.predict(pool)
            result[f"benefit_{specialist}_{kind}"] = score.astype(np.float32)
    return result


def evaluate_validation(
    config: dict[str, Any],
    models: dict[tuple[str, str], Any],
    train_feature_columns: list[str],
    categorical: list[str],
    policies: pd.DataFrame,
    specialists: tuple[str, ...],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    records = []
    sources: dict[str, Any] = {}
    for split, filename in SPLITS.items():
        frame, split_sources = load_split(
            resolve(config["predictions_root"]),
            resolve(config["validation_feature_cache_dir"]),
            split,
            filename,
            specialists=specialists,
        )
        sources.update(split_sources)
        reserved = {
            "id1", "id2", "target", "category", "bge_probability",
            "minilm_probability", "rumodernbert_probability", "bge_token_length_max",
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
            raise ValueError(f"Raw BGE prediction order differs on {split}")
        features, validation_categorical = router_feature_frame(
            cheap, bge_prediction_view(bge_raw)
        )
        if validation_categorical != categorical:
            raise ValueError(f"Categorical router columns differ on {split}")
        missing = set(train_feature_columns) - set(features.columns)
        extra = set(features.columns) - set(train_feature_columns)
        if missing or extra:
            raise ValueError(f"Router feature schema differs on {split}: missing={missing}, extra={extra}")
        features = features[train_feature_columns]
        predicted = predict_router_scores(models, features, categorical, specialists)
        for column in predicted:
            frame[column] = predicted[column].to_numpy()
        target = frame["target"].to_numpy(dtype=np.int8)
        category = frame["category"].astype(str).to_numpy()
        baseline = macro_ap(target, frame["bge_probability"].to_numpy(), category)
        bge = frame["bge_probability"].to_numpy(dtype=np.float64)
        full_scores = {"full_bge": (bge, 0.0, 0.0, "BGE")}
        if "minilm" in specialists:
            full_scores["full_bge_minilm_50_50"] = (
                0.5 * bge + 0.5 * frame["minilm_probability"].to_numpy(dtype=np.float64),
                1.0, 0.0, "BGE+MiniLM",
            )
        if "rumodernbert" in specialists:
            full_scores["full_bge_rumodernbert_50_50"] = (
                0.5 * bge + 0.5 * frame["rumodernbert_probability"].to_numpy(dtype=np.float64),
                0.0, 1.0, "BGE+RuModernBERT",
            )
        if set(AVAILABLE_SPECIALISTS).issubset(specialists):
            full_scores["full_triple_equal"] = (
                (
                    bge
                    + frame["minilm_probability"].to_numpy(dtype=np.float64)
                    + frame["rumodernbert_probability"].to_numpy(dtype=np.float64)
                ) / 3.0,
                1.0, 1.0, "BGE+MiniLM+RuModernBERT",
            )
        for method, (score, mini_coverage, ru_coverage, specialist) in full_scores.items():
            ap = macro_ap(target, score, category)
            records.append(
                {
                    "split": split,
                    "method": method,
                    "specialist": specialist,
                    "specialist_budget": mini_coverage + ru_coverage,
                    "blend_weight": np.nan,
                    "route_coverage": min(1.0, mini_coverage + ru_coverage),
                    "minilm_coverage": mini_coverage,
                    "rumodernbert_coverage": ru_coverage,
                    "macro_ap": ap,
                    "delta_vs_bge": ap - baseline,
                }
            )

        for policy in policies.itertuples(index=False):
            method = str(policy.method)
            budget = float(policy.budget)
            weight = float(policy.blend_weight)
            if method.startswith("learned_multi_"):
                kind = method.removeprefix("learned_multi_")
                mini = frame[f"benefit_minilm_{kind}"].to_numpy(dtype=np.float64)
                ru = frame[f"benefit_rumodernbert_{kind}"].to_numpy(dtype=np.float64)
                priority = np.maximum(mini, ru)
                expert = np.where(mini >= ru, "minilm", "rumodernbert")
                mask = top_budget_mask(priority, budget, frame["id1"], frame["id2"])
                score = frame["bge_probability"].to_numpy(dtype=np.float64).copy()
                for specialist in specialists:
                    chosen = mask & (expert == specialist)
                    specialist_score = frame[f"{specialist}_probability"].to_numpy(dtype=np.float64)
                    score[chosen] = (1.0 - weight) * score[chosen] + weight * specialist_score[chosen]
                mini_coverage = float(np.mean(mask & (expert == "minilm")))
                ru_coverage = float(np.mean(mask & (expert == "rumodernbert")))
            else:
                specialist = str(policy.specialist)
                priority = route_priority(frame, method, specialist, int(config["seed"]))
                mask = top_budget_mask(priority, budget, frame["id1"], frame["id2"])
                score = blended_score(frame, mask, specialist, weight)
                mini_coverage = float(mask.mean()) if specialist == "minilm" else 0.0
                ru_coverage = float(mask.mean()) if specialist == "rumodernbert" else 0.0
            ap = macro_ap(target, score, category)
            records.append(
                {
                    "split": split,
                    "method": method,
                    "specialist": str(policy.specialist),
                    "specialist_budget": budget,
                    "blend_weight": weight,
                    "route_coverage": mini_coverage + ru_coverage,
                    "minilm_coverage": mini_coverage,
                    "rumodernbert_coverage": ru_coverage,
                    "macro_ap": ap,
                    "delta_vs_bge": ap - baseline,
                }
            )
    return pd.DataFrame(records), sources


def make_main_table(
    validation: pd.DataFrame, policies: pd.DataFrame, runtime: pd.DataFrame, private_pairs: int
) -> pd.DataFrame:
    rows = []
    keys = ["method", "specialist", "specialist_budget", "blend_weight"]
    for key, part in validation.groupby(keys, sort=False, dropna=False):
        by_split = part.set_index("split")
        if not {"iid", "hard", "ood"}.issubset(by_split.index):
            continue
        iid = by_split.loc["iid"]
        timing = estimate_private_t4_runtime(
            private_pairs,
            float(iid["minilm_coverage"]),
            float(iid["rumodernbert_coverage"]),
            runtime,
        )
        selected_rows = policies.loc[
            policies["method"].eq(key[0])
            & policies["specialist"].eq(key[1])
            & np.isclose(policies["budget"], float(key[2]))
        ]
        is_baseline = str(key[0]).startswith("full_")
        selected = None if is_baseline else selected_rows.iloc[0]
        rows.append(
            {
                "method": key[0],
                "specialist": key[1],
                "specialist_budget": key[2],
                "blend_weight_oof": key[3],
                "oof_ap": np.nan if selected is None else float(selected["oof_macro_ap"]),
                "oof_gain": np.nan if selected is None else float(selected["oof_delta_vs_bge"]),
                "iid_ap": float(iid["macro_ap"]),
                "iid_gain": float(iid["delta_vs_bge"]),
                "hard_ap": float(by_split.loc["hard", "macro_ap"]),
                "hard_gain": float(by_split.loc["hard", "delta_vs_bge"]),
                "ood_ap": float(by_split.loc["ood", "macro_ap"]),
                "ood_gain": float(by_split.loc["ood", "delta_vs_bge"]),
                **timing,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    specialists = tuple(config.get("specialists", AVAILABLE_SPECIALISTS))
    if not specialists or not set(specialists).issubset(AVAILABLE_SPECIALISTS):
        raise ValueError(
            f"specialists must be a non-empty subset of {AVAILABLE_SPECIALISTS}, got {specialists}"
        )
    output_dir = args.output_dir or resolve(config["output_dir"])
    if args.smoke and args.output_dir is None:
        output_dir = output_dir / "_smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    train, cheap = load_train(config, args.smoke, specialists)
    features, categorical = router_feature_frame(cheap, bge_prediction_view(train))
    router_oof, models, target_summary = fit_routers(
        config, train, features, categorical, output_dir, args.smoke, specialists
    )
    selection_frame = pd.concat(
        [
            router_oof,
            cheap.drop(columns=["category"], errors="ignore").reset_index(drop=True),
        ],
        axis=1,
    )
    policies = select_oof_policies(config, selection_frame, specialists)
    # This is the protocol boundary: all route rankings and blend weights are frozen before
    # any IID/Hard/OOD labels are loaded by evaluate_validation().
    policies.to_csv(output_dir / "oof_policy_selection.csv", index=False)
    router_oof.to_parquet(output_dir / "router_oof_predictions.parquet", index=False)
    target_summary.to_csv(output_dir / "benefit_target_summary.csv", index=False)

    validation, sources = evaluate_validation(
        config, models, list(features.columns), categorical, policies, specialists
    )
    validation.to_csv(output_dir / "validation_routing_results.csv", index=False)
    runtime_path = resolve(config["runtime_measurements"])
    table = make_main_table(
        validation,
        policies,
        pd.read_csv(runtime_path),
        int(config["private_pairs"]),
    )
    table.to_csv(output_dir / "main_table.csv", index=False)

    learned = policies.loc[policies["method"].str.startswith("learned_")].copy()
    simple = policies.loc[policies["method"].isin(["uncertainty", "simple_slice"])].copy()
    comparisons = learned.merge(
        simple.groupby(["budget"], as_index=False)["oof_macro_ap"]
        .max().rename(columns={"oof_macro_ap": "best_simple_oof_ap"}),
        on=["budget"],
        how="left",
    )
    comparisons["gain_over_simple"] = (
        comparisons["oof_macro_ap"] - comparisons["best_simple_oof_ap"]
    )
    comparisons.to_csv(output_dir / "learned_vs_simple_oof.csv", index=False)
    best_advantage = float(comparisons["gain_over_simple"].max(skipna=True))
    learned_accepted = best_advantage >= float(config["min_learned_gain_over_simple"])
    eligible = learned if learned_accepted else simple
    recommended_policies = eligible.sort_values("oof_delta_vs_bge", ascending=False).head(2)
    recommended = table.merge(
        recommended_policies[["method", "specialist", "budget"]],
        left_on=["method", "specialist", "specialist_budget"],
        right_on=["method", "specialist", "budget"],
        how="inner",
    ).drop(columns="budget")
    recommended.to_csv(output_dir / "recommended_strategies.csv", index=False)

    manifest = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "smoke": args.smoke,
        "training_source": "human train neural OOF only",
        "component_disjoint_router_oof": True,
        "validation_loaded_after_oof_policy_freeze": True,
        "specialist_scores_used_as_router_features": False,
        "specialist_scores_used_for_benefit_targets": True,
        "both_specialists_per_pair": False,
        "specialists": list(specialists),
        "learned_router_accepted": learned_accepted,
        "best_learned_oof_gain_over_simple": best_advantage,
        "min_required_gain_over_simple": float(config["min_learned_gain_over_simple"]),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_columns": list(features.columns),
        "categorical_columns": categorical,
        "source_hashes": {
            model: sha256_file(resolve(config["oof_predictions"][model]))
            for model in ("bge", *specialists)
        },
        "runtime_measurements_sha256": sha256_file(runtime_path),
        "validation_sources": sources,
        "outputs": {
            "main_table": "main_table.csv",
            "oof_policy_selection": "oof_policy_selection.csv",
            "recommended_strategies": "recommended_strategies.csv",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "status": "complete",
        "output_dir": str(output_dir),
        "recommended": recommended[["method", "specialist", "specialist_budget", "blend_weight_oof", "oof_gain"]].to_dict("records"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
