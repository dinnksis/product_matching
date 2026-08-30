#!/usr/bin/env python3
"""Analyze saved BGE/MiniLM/RuModernBERT predictions without training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.selective_specialist_analysis import (
    ROUTE_BUDGETS,
    SCORE_MODES,
    SPECIALISTS,
    add_pairwise_proxies,
    compact_oracle_summary,
    oracle_routing_rows,
    pairwise_summary,
    slice_metric_rows,
    stable_slice_table,
)


SPLITS = {
    "iid": "iid_validation_predictions.parquet",
    "hard": "hard_validation_predictions.parquet",
    "ood": "ood_validation_predictions.parquet",
}
PREDICTION_COLUMNS = (
    "id1",
    "id2",
    "target",
    "category_1",
    "score",
    "token_length_ab",
    "token_length_ba",
)
EXPERIMENT_VERSION = "selective_specialist_oracle_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path, default=ROOT / "preds")
    parser.add_argument(
        "--feature-cache",
        type=Path,
        default=ROOT / "artifacts" / "catboost2_trust_router_v1" / "validation_feature_cache",
    )
    parser.add_argument("--items", type=Path, default=ROOT / "data" / "items_human.parquet")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "reports" / EXPERIMENT_VERSION,
    )
    parser.add_argument("--slice-min-size", type=int, default=20)
    parser.add_argument("--stable-min-size", type=int, default=100)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(
    predictions_root: Path,
    feature_cache: Path,
    split: str,
    filename: str,
    specialists: tuple[str, ...] = SPECIALISTS,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    sources: dict[str, dict[str, Any]] = {}
    frames: dict[str, pd.DataFrame] = {}
    for model in ("bge", *specialists):
        path = predictions_root / f"preds_{model}" / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[f"{split}/{model}"] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        frame = pd.read_parquet(path, columns=list(PREDICTION_COLUMNS))
        if frame[["id1", "id2"]].duplicated().any():
            raise ValueError(f"{model}/{split} has duplicate pairs")
        if not frame["target"].isin([0.0, 1.0]).all():
            raise ValueError(f"{model}/{split} has non-binary labels")
        score = frame["score"].to_numpy(dtype=np.float64)
        if not np.isfinite(score).all() or np.any((score < 0) | (score > 1)):
            raise ValueError(f"{model}/{split} has invalid scores")
        frames[model] = frame

    reference = frames["bge"].rename(
        columns={"category_1": "category", "score": "bge_probability"}
    )
    result = reference[
        [
            "id1",
            "id2",
            "target",
            "category",
            "bge_probability",
            "token_length_ab",
            "token_length_ba",
        ]
    ].copy()
    result["bge_token_length_max"] = result[
        ["token_length_ab", "token_length_ba"]
    ].max(axis=1)
    result = result.drop(columns=["token_length_ab", "token_length_ba"])
    for specialist in specialists:
        current = frames[specialist][["id1", "id2", "target", "category_1", "score"]].rename(
            columns={
                "target": f"target_{specialist}",
                "category_1": f"category_{specialist}",
                "score": f"{specialist}_probability",
            }
        )
        result = result.merge(
            current,
            on=["id1", "id2"],
            how="inner",
            validate="one_to_one",
            sort=False,
        )
        if not np.array_equal(result["target"], result[f"target_{specialist}"]):
            raise ValueError(f"{specialist}/{split} labels differ from BGE")
        if not np.array_equal(
            result["category"].astype(str), result[f"category_{specialist}"].astype(str)
        ):
            raise ValueError(f"{specialist}/{split} categories differ from BGE")
        result = result.drop(columns=[f"target_{specialist}", f"category_{specialist}"])
    if len(result) != len(reference):
        raise ValueError(f"{split} prediction pair sets differ")

    pair_path = feature_cache / f"{split}_pairs.parquet"
    base_path = feature_cache / f"{split}_base.parquet"
    if not pair_path.is_file() or not base_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen feature cache for {split}: {pair_path}, {base_path}"
        )
    pairs = pd.read_parquet(pair_path, columns=["id1", "id2", "target"])
    features = pd.read_parquet(base_path)
    if len(pairs) != len(features):
        raise ValueError(f"{split} cached pairs/features have different lengths")
    if not pairs[["id1", "id2"]].equals(
        result[["id1", "id2"]].reset_index(drop=True)
    ) or not np.array_equal(
        pairs["target"].to_numpy(), result["target"].to_numpy()
    ):
        raise ValueError(f"{split} feature cache is not row-aligned with predictions")
    if features["category"].astype(str).tolist() != result["category"].astype(str).tolist():
        raise ValueError(f"{split} cached categories differ from predictions")
    duplicate_columns = set(result) & set(features)
    duplicate_columns.discard("category")
    if duplicate_columns:
        raise ValueError(f"Unexpected duplicate feature columns: {sorted(duplicate_columns)}")
    features = features.drop(columns=["category"]).reset_index(drop=True)
    result = pd.concat([result.reset_index(drop=True), features], axis=1)
    sources[f"{split}/feature_pairs"] = {
        "path": str(pair_path.resolve()),
        "bytes": pair_path.stat().st_size,
    }
    sources[f"{split}/feature_base"] = {
        "path": str(base_path.resolve()),
        "bytes": base_path.stat().st_size,
    }
    return result, sources


def representative_examples(
    frames: dict[str, pd.DataFrame],
    items_path: Path,
    *,
    per_direction: int = 12,
) -> pd.DataFrame:
    required_ids = pd.unique(
        pd.concat([frame[["id1", "id2"]] for frame in frames.values()])
        .to_numpy()
        .reshape(-1)
    )
    items = pd.read_parquet(items_path, columns=["id", "name"])
    items = items.loc[items["id"].isin(required_ids)].set_index("id", verify_integrity=True)["name"]
    rows = []
    for split, frame in frames.items():
        for specialist in SPECIALISTS:
            proxied = add_pairwise_proxies(frame, specialist)
            for direction, ascending in (("specialist_better", False), ("bge_better", True)):
                chosen = proxied.loc[
                    proxied["absolute_error_status"].eq(direction)
                ].sort_values("absolute_error_gain", ascending=ascending, kind="stable")
                chosen = chosen.head(per_direction)
                for row in chosen.itertuples(index=False):
                    rows.append(
                        {
                            "split": split,
                            "specialist": specialist,
                            "direction": direction,
                            "id1": row.id1,
                            "id2": row.id2,
                            "category": row.category,
                            "target": row.target,
                            "bge_probability": row.bge_probability,
                            "specialist_probability": getattr(
                                row, f"{specialist}_probability"
                            ),
                            "absolute_error_gain": row.absolute_error_gain,
                            "logloss_gain": row.logloss_gain,
                            "title_1": str(items.loc[row.id1]),
                            "title_2": str(items.loc[row.id2]),
                            "title_token_set": row.title_token_set,
                            "brand_match": row.brand_match,
                            "brand_conflict": row.brand_conflict,
                            "model_code_match": row.model_code_match,
                            "model_code_conflict": row.model_code_conflict,
                            "title_code_match": row.title_code_match,
                            "title_code_conflict": row.title_code_conflict,
                            "primary_conflict_type": row.primary_conflict_type,
                        }
                    )
    return pd.DataFrame(rows)


def compact_budget_markdown(summary: pd.DataFrame, split: str) -> list[str]:
    selected = summary.loc[summary["split"] == split].copy()
    pivot = selected.pivot(index="route_budget", columns="route_expert", values="macro_ap_gain")
    lines = [
        f"### {split.upper()}",
        "",
        "| route budget | oracle MiniLM gain | oracle RuModern gain | oracle best-expert gain |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for budget, row in pivot.iterrows():
        lines.append(
            f"| {budget:.0%} | {row.get('minilm', math.nan):+.6f} | "
            f"{row.get('rumodernbert', math.nan):+.6f} | "
            f"{row.get('best_expert', math.nan):+.6f} |"
        )
    return lines + [""]


def expert_overlap_summary(corrections: pd.DataFrame) -> pd.DataFrame:
    statuses = corrections.pivot(
        index=["split", "id1", "id2"],
        columns="specialist",
        values="absolute_error_status",
    ).reset_index()
    rows = []
    for split, part in statuses.groupby("split", sort=True):
        minilm_better = part["minilm"].eq("specialist_better")
        rumodern_better = part["rumodernbert"].eq("specialist_better")
        rows.append(
            {
                "split": split,
                "n": len(part),
                "minilm_better": int(minilm_better.sum()),
                "rumodernbert_better": int(rumodern_better.sum()),
                "both_better": int((minilm_better & rumodern_better).sum()),
                "minilm_only": int((minilm_better & ~rumodern_better).sum()),
                "rumodernbert_only": int((rumodern_better & ~minilm_better).sum()),
                "neither_better": int((~minilm_better & ~rumodern_better).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_report(
    pairwise: pd.DataFrame,
    compact: pd.DataFrame,
    stable: pd.DataFrame,
    overlap: pd.DataFrame,
) -> str:
    lines = [
        "# Selective specialist analysis: BGE → MiniLM / RuModernBERT",
        "",
        "Эксперимент использует только сохранённые predictions и label-free признаки. "
        "Обучение, submission и Google Sheets sync не выполнялись.",
        "",
        "## Важные ограничения",
        "",
        "- Train/OOF neural predictions для этих трёх checkpoint отсутствуют. Поэтому "
        "binary correctness использует заранее фиксированный threshold `0.5`, а не "
        "подобранный на IID threshold.",
        "- Oracle использует label и ранжирует пары по уменьшению `|score-label|`. Это "
        "label-aware upper-bound политики directional routing, но не доказанный "
        "комбинаторный максимум macro AP.",
        "- Все fixed probability weights заданы заранее: 25/50/75% specialist. Они не "
        "подбирались на IID/Hard/OOD.",
        "- `mean_normalized_rank` — диагностический прежний ensemble-вариант; для "
        "реального selective inference полный specialist rank заранее неизвестен.",
        "- OOD category identity не используется как production slice.",
        "",
        "## Одиночные модели и pairwise proxies",
        "",
        "| split | specialist | BGE macro AP | specialist macro AP | Δ AP | binary net @0.5 | mean logloss gain | mean MAE gain |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.specialist} | {row.bge_macro_ap:.6f} | "
            f"{row.specialist_macro_ap:.6f} | {row.macro_ap_delta:+.6f} | "
            f"{row.binary_net_correction:+d} | "
            f"{row.bge_logloss-row.specialist_logloss:+.6f} | "
            f"{row.bge_mae-row.specialist_mae:+.6f} |"
        )
    lines.extend(
        [
            "",
            "## Пересечение полезных corrections по MAE proxy",
            "",
            "| split | MiniLM better | RuModern better | both | MiniLM only | RuModern only |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in overlap.itertuples(index=False):
        lines.append(
            f"| {row.split} | {row.minilm_better} | {row.rumodernbert_better} | "
            f"{row.both_better} | {row.minilm_only} | {row.rumodernbert_only} |"
        )
    lines.extend(["", "## Oracle routing upper bound", ""])
    for split in ("iid", "hard", "ood"):
        lines.extend(compact_budget_markdown(compact, split))

    candidates = stable.loc[stable["production_candidate"]].head(20)
    lines.extend(
        [
            "## Стабильные label-free slices",
            "",
            "Строгий флаг требует минимум 100 строк, положительный logloss/MAE gain, "
            "неотрицательную binary net correction и положительный AP delta, когда AP "
            "определён, одновременно на IID и Hard.",
            "",
            "| specialist | slice | value | IID N | Hard N | IID ΔAP | Hard ΔAP | IID logloss gain | Hard logloss gain | OOD better |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in candidates.itertuples(index=False):
        lines.append(
            f"| {row.specialist} | {row.slice_type} | {row.slice_value} | "
            f"{row.iid_n:.0f} | {row.hard_n:.0f} | {row.iid_macro_ap_delta:+.6f} | "
            f"{row.hard_macro_ap_delta:+.6f} | {row.iid_logloss_gain:+.6f} | "
            f"{row.hard_logloss_gain:+.6f} | {bool(row.ood_better)} |"
        )
    if candidates.empty:
        lines.append("| — | — | Строгих стабильных slices не найдено | — | — | — | — | — | — | — |")
    lines.extend(
        [
            "",
            "## Артефакты",
            "",
            "- `pairwise_summary.csv` — общие pairwise corrections;",
            "- `pairwise_corrections.parquet` — per-example proxy labels;",
            "- `expert_overlap_summary.csv` — пересечение полезных corrections двух specialists;",
            "- `slice_metrics.csv` и `stable_slices.csv` — все slices и межsplit устойчивость;",
            "- `oracle_routing_results.csv` — полный перебор budgets/policies/fixed scores;",
            "- `oracle_budget_summary.csv` — основной компактный label-aware результат только для probability scores;",
            "- `oracle_budget_summary_all_modes.csv` — диагностический вариант, где дополнительно разрешён normalized-rank blend;",
            "- `representative_examples.csv` — наиболее сильные исправления и регрессии;",
            "- `manifest.json` — источники, hashes и ограничения.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.slice_min_size <= 0 or args.stable_min_size <= 0:
        raise ValueError("Slice size thresholds must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    sources: dict[str, dict[str, Any]] = {}
    pairwise_rows: list[dict[str, Any]] = []
    correction_parts: list[pd.DataFrame] = []
    slice_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for split, filename in SPLITS.items():
        frame, split_sources = load_split(
            args.predictions_root, args.feature_cache, split, filename
        )
        frames[split] = frame
        sources.update(split_sources)
        oracle_rows.extend(oracle_routing_rows(frame, split))
        for specialist in SPECIALISTS:
            proxied = add_pairwise_proxies(frame, specialist)
            pairwise_rows.append(pairwise_summary(proxied, specialist, split))
            slice_rows.extend(
                slice_metric_rows(
                    proxied,
                    specialist,
                    split,
                    min_size=args.slice_min_size,
                )
            )
            correction_parts.append(
                proxied[
                    [
                        "id1",
                        "id2",
                        "target",
                        "category",
                        "bge_probability",
                        f"{specialist}_probability",
                        "binary_status",
                        "logloss_status",
                        "absolute_error_status",
                        "logloss_gain",
                        "absolute_error_gain",
                    ]
                ].assign(split=split, specialist=specialist)
            )

    pairwise = pd.DataFrame(pairwise_rows)
    corrections = pd.concat(correction_parts, ignore_index=True)
    slices = pd.DataFrame(slice_rows)
    stable = stable_slice_table(slices, min_size=args.stable_min_size)
    oracle = pd.DataFrame(oracle_rows)
    compact_all_modes = compact_oracle_summary(oracle)
    compact = compact_oracle_summary(
        oracle.loc[oracle["score_mode"].ne("mean_normalized_rank")]
    )
    examples = representative_examples(frames, args.items)
    overlap = expert_overlap_summary(corrections)

    pairwise.to_csv(args.output_dir / "pairwise_summary.csv", index=False)
    corrections.to_parquet(
        args.output_dir / "pairwise_corrections.parquet", index=False, compression="zstd"
    )
    slices.to_csv(args.output_dir / "slice_metrics.csv", index=False)
    stable.to_csv(args.output_dir / "stable_slices.csv", index=False)
    oracle.to_csv(args.output_dir / "oracle_routing_results.csv", index=False)
    compact.to_csv(args.output_dir / "oracle_budget_summary.csv", index=False)
    compact_all_modes.to_csv(
        args.output_dir / "oracle_budget_summary_all_modes.csv", index=False
    )
    overlap.to_csv(args.output_dir / "expert_overlap_summary.csv", index=False)
    examples.to_csv(args.output_dir / "representative_examples.csv", index=False)

    completed_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    summary = {
        "status": "complete",
        "experiment": EXPERIMENT_VERSION,
        "completed_at_utc": completed_at,
        "training_performed": False,
        "submission_performed": False,
        "google_sheets_modified": False,
        "splits": {name: len(frame) for name, frame in frames.items()},
        "specialists": list(SPECIALISTS),
        "route_budgets": list(ROUTE_BUDGETS),
        "score_modes": list(SCORE_MODES),
        "primary_oracle_summary_excludes": ["mean_normalized_rank"],
        "oracle_policies": ["global_directional", "category_balanced_directional"],
        "binary_threshold": 0.5,
        "binary_threshold_source": "fixed_not_fitted",
        "oof_neural_thresholds_available": False,
        "limitations": [
            "No train/OOF predictions exist for all three neural checkpoints.",
            "Oracle selection uses labels and is not a production router.",
            "Directional oracle is not a proof of the exact combinatorial AP optimum.",
            "Mean normalized rank requires full specialist score distribution.",
            "OOD category identity is descriptive only and forbidden for production rules.",
        ],
        "sources": sources,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in args.output_dir.iterdir()
            if path.is_file() and path.name not in {"manifest.json", "REPORT.md"}
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "REPORT.md").write_text(
        build_report(pairwise, compact, stable, overlap), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
