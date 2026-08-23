"""Build label-free candidate rules, then attach labels for pilot statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTIONS = ROOT / "artifacts" / "qwen_semantic_extraction_v1_4_sanitized50" / "sanitized_pairs.jsonl"
DEFAULT_LABELS = ROOT / "data" / "qwen_semantic_pilot_v1" / "pilot_labels.parquet"
DEFAULT_DISCOVERY = ROOT / "data" / "rule_discovery_split_v1" / "split_assignments.parquet"
DEFAULT_OUTPUT = ROOT / "reports" / "qwen_pilot_rules_v1_4_50"
RELATION_GROUPS = {
    "missing_a": "missing_one_side",
    "missing_b": "missing_one_side",
    "subset": "specificity_difference",
    "more_specific": "specificity_difference",
}
RELATION_TEXT = {
    "different_value": "has different explicit values",
    "missing_one_side": "is explicitly known on only one side",
    "specificity_difference": "has a subset or specificity difference",
    "compatible": "has compatible values",
    "incompatible": "has incompatible values",
    "conflicting_sources": "has conflicting title/attribute sources",
    "unknown": "has an unresolved semantic relation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze candidate rules from a Qwen pilot.")
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--discovery-assignments", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument(
        "--concept-map",
        type=Path,
        default=None,
        help="Optional label-free CSV/Parquet with source_concept and target_concept.",
    )
    parser.add_argument(
        "--effect-baseline",
        choices=("pilot", "discovery"),
        default="pilot",
        help="Reference prevalence for diagnostic effects; pilot avoids case-control sampling bias.",
    )
    parser.add_argument(
        "--sampling-design",
        choices=("balanced_label_aware", "prevalence_random"),
        default="balanced_label_aware",
        help="Sampling design used only for honest interpretation in the report.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--posterior-draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def stable_seed(seed: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(map(str, (seed,) + parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def canonical_relation(relation: str) -> str:
    return RELATION_GROUPS.get(relation, relation)


def rule_id(concept: str, relation: str) -> str:
    return "rule_" + hashlib.sha256(f"{concept}|{relation}".encode("utf-8")).hexdigest()[:16]


def raw_value(side: Any) -> Any:
    if not isinstance(side, dict):
        return None
    value = side.get("value")
    return value.get("raw_value") if isinstance(value, dict) else None


def load_concept_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    resolved = path.resolve()
    frame = pd.read_parquet(resolved) if resolved.suffix.lower() == ".parquet" else pd.read_csv(resolved)
    required = {"source_concept", "target_concept"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Concept map must contain {sorted(required)}")
    frame = frame[list(required)].dropna().drop_duplicates()
    if frame["source_concept"].duplicated().any():
        raise ValueError("Concept map contains conflicting source_concept rows")
    return dict(zip(frame["source_concept"].astype(str), frame["target_concept"].astype(str)))


def build_label_free_assignments(
    path: Path, concept_map: dict[str, str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            # Deliberately do not access source["human_label"] here.
            differences = source.get("differences", [])
            missing = source.get("missing_information", [])
            facts = differences + missing
            pair_rows.append(
                {
                    "pair_id": str(source["pair_id"]),
                    "category": str(source["category"]),
                    "difference_count": len(differences),
                    "semantic_fact_count": len(facts),
                }
            )
            seen: set[tuple[str, str]] = set()
            for fact in facts:
                original_concept = str(fact["concept"])
                concept = concept_map.get(original_concept, original_concept)
                source_relation = str(fact["relation"])
                relation = canonical_relation(source_relation)
                key = (concept, relation)
                if key in seen:
                    continue
                seen.add(key)
                assignments.append(
                    {
                        "pair_id": str(source["pair_id"]),
                        "category": str(source["category"]),
                        "rule_id": rule_id(concept, relation),
                        "canonical_rule": f"{concept} {RELATION_TEXT.get(relation, relation)}",
                        "concept": concept,
                        "original_concept": original_concept,
                        "relation": relation,
                        "source_relation": source_relation,
                        "value_a": json.dumps(fact.get("value_a"), ensure_ascii=False),
                        "value_b": json.dumps(fact.get("value_b"), ensure_ascii=False),
                        "difference_count": len(differences),
                        "semantic_fact_count": len(facts),
                    }
                )
    return pd.DataFrame(assignments), pd.DataFrame(pair_rows)


def logit(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 1e-9, 1 - 1e-9)
    return np.log(values / (1 - values))


def beta_summary(positive: int, negative: int) -> dict[str, float]:
    alpha, beta = positive + 0.5, negative + 0.5
    mean = alpha / (alpha + beta)
    variance = alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))
    return {"mean": mean, "sd": math.sqrt(variance), "alpha": alpha, "beta": beta}


def effect_posterior(
    positive: int,
    negative: int,
    baseline_positive: int,
    baseline_negative: int,
    draws: int,
    seed: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    rule_p = rng.beta(positive + 0.5, negative + 0.5, draws)
    base_p = rng.beta(baseline_positive + 0.5, baseline_negative + 0.5, draws)
    effect = logit(rule_p) - logit(base_p)
    low, median, high = np.quantile(effect, [0.025, 0.5, 0.975])
    return {
        "effect_mean": float(effect.mean()),
        "effect_median": float(median),
        "effect_low_95": float(low),
        "effect_high_95": float(high),
        "effect_sd": float(effect.std(ddof=1)),
        "probability_positive_effect": float((effect > 0).mean()),
        "probability_negative_effect": float((effect < 0).mean()),
    }


def global_effect_posterior(
    group: pd.DataFrame,
    baselines: pd.DataFrame,
    draws: int,
    seed: int,
) -> dict[str, float]:
    positive = int(group["human_label"].sum())
    negative = len(group) - positive
    rng = np.random.default_rng(seed)
    rule_p = rng.beta(positive + 0.5, negative + 0.5, draws)
    baseline_mix = np.zeros(draws)
    for category, category_group in group.groupby("category"):
        baseline = baselines.loc[category]
        sampled = rng.beta(
            float(baseline["positive_count"]) + 0.5,
            float(baseline["negative_count"]) + 0.5,
            draws,
        )
        baseline_mix += len(category_group) / len(group) * sampled
    effect = logit(rule_p) - logit(baseline_mix)
    low, median, high = np.quantile(effect, [0.025, 0.5, 0.975])
    return {
        "global_effect": float(effect.mean()),
        "global_effect_median": float(median),
        "global_effect_low_95": float(low),
        "global_effect_high_95": float(high),
        "global_effect_sd": float(effect.std(ddof=1)),
        "global_probability_positive": float((effect > 0).mean()),
        "global_probability_negative": float((effect < 0).mean()),
    }


def scope_candidate(category_rows: pd.DataFrame, global_row: dict[str, Any]) -> str:
    if len(category_rows) < 2:
        return "UNCERTAIN"
    positive = category_rows["category_effect_low_95"] > 0
    negative = category_rows["category_effect_high_95"] < 0
    if positive.any() and negative.any():
        return "HETEROGENEOUS"
    # With this deliberately small, label-aware pilot we retain direction and
    # heterogeneity signals but do not promote them to scope decisions.
    return "UNCERTAIN"


def empirical_bayes_shrinkage(category_rows: pd.DataFrame, global_effect: float) -> pd.DataFrame:
    result = category_rows.copy()
    if len(result) < 2:
        result["heterogeneity_tau2"] = 0.0
        result["category_shrinkage_weight"] = 0.0
        result["category_effect_shrunk"] = global_effect
        return result
    effects = result["category_effect_median"].to_numpy(float)
    variances = np.square(result["category_effect_sd"].to_numpy(float))
    inverse = 1 / np.maximum(variances, 1e-12)
    center = float(np.average(effects, weights=inverse))
    observed = float(np.average(np.square(effects - center), weights=inverse))
    sampling = float(np.average(variances, weights=inverse))
    tau2 = max(0.0, observed - sampling)
    weights = tau2 / (tau2 + variances)
    result["heterogeneity_tau2"] = tau2
    result["category_shrinkage_weight"] = weights
    result["category_effect_shrunk"] = global_effect + weights * (effects - global_effect)
    return result


def quantiles(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in columns:
        for quantile in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
            rows.append({"metric": column, "quantile": quantile, "value": frame[column].quantile(quantile)})
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    """Render a small DataFrame without pandas' optional tabulate dependency."""
    if frame.empty:
        return "_Нет примеров._"

    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            value = f"{float(value):.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    columns = [str(column) for column in frame.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(render(value) for value in row) + " |"
        for row in frame.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def main() -> None:
    args = parse_args()
    if args.posterior_draws < 1000:
        raise ValueError("posterior-draws must be at least 1000")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    concept_map = load_concept_map(args.concept_map)
    assignments, pairs = build_label_free_assignments(args.extractions.resolve(), concept_map)
    assignments.to_parquet(output_dir / "rule_assignments_label_free.parquet", index=False)
    assignments.to_csv(output_dir / "rule_assignments_label_free.csv", index=False, encoding="utf-8-sig")

    # Labels are loaded only after all semantic rule boundaries have been fixed.
    labels = pd.read_parquet(args.labels.resolve(), columns=["pair_id", "human_label"])
    labels["pair_id"] = labels["pair_id"].astype(str)
    selected_labels = labels[labels["pair_id"].isin(pairs["pair_id"])].copy()
    if len(selected_labels) != len(pairs) or selected_labels["pair_id"].duplicated().any():
        raise ValueError("Pilot labels do not match extracted pair IDs one-to-one")
    labeled = assignments.merge(selected_labels, on="pair_id", how="left", validate="many_to_one")

    discovery = pd.read_parquet(
        args.discovery_assignments.resolve(), columns=["category", "target", "split"]
    )
    discovery = discovery[discovery["split"] == "rule_discovery"].copy()
    discovery["human_label"] = discovery["target"].astype(int)
    baseline = discovery.groupby("category")["human_label"].agg(["size", "sum"])
    baseline = baseline.rename(columns={"size": "pair_count", "sum": "positive_count"})
    baseline["negative_count"] = baseline["pair_count"] - baseline["positive_count"]
    baseline["category_baseline"] = (baseline["positive_count"] + 0.5) / (baseline["pair_count"] + 1)
    pilot_baseline = selected_labels.merge(pairs[["pair_id", "category"]], on="pair_id", validate="one_to_one").groupby("category")["human_label"].agg(["size", "sum"])
    pilot_baseline = pilot_baseline.rename(columns={"size": "pilot_pairs", "sum": "pilot_positive"})
    pilot_baseline["pilot_category_baseline"] = pilot_baseline["pilot_positive"] / pilot_baseline["pilot_pairs"]
    baseline = baseline.join(pilot_baseline, how="left")
    baseline["pilot_negative"] = baseline["pilot_pairs"] - baseline["pilot_positive"]
    if args.effect_baseline == "pilot":
        effect_baseline = baseline.copy()
        effect_baseline["positive_count"] = effect_baseline["pilot_positive"]
        effect_baseline["negative_count"] = effect_baseline["pilot_negative"]
        effect_baseline["effect_reference_baseline"] = (
            effect_baseline["pilot_positive"] + 0.5
        ) / (effect_baseline["pilot_pairs"] + 1)
    else:
        effect_baseline = baseline.copy()
        effect_baseline["effect_reference_baseline"] = effect_baseline["category_baseline"]
    baseline.reset_index().to_csv(output_dir / "category_baselines.csv", index=False, encoding="utf-8-sig")

    global_rows: list[dict[str, Any]] = []
    category_rows_all: list[pd.DataFrame] = []
    for rid, group in labeled.groupby("rule_id", sort=True):
        first = group.iloc[0]
        positive = int(group["human_label"].sum())
        support = len(group)
        clean = group.drop_duplicates("pair_id")
        global_effect = global_effect_posterior(
            clean,
            effect_baseline,
            args.posterior_draws,
            stable_seed(args.seed, rid, "global"),
        )
        global_row: dict[str, Any] = {
            "rule_id": rid,
            "canonical_rule": first["canonical_rule"],
            "concept": first["concept"],
            "relation": first["relation"],
            "global_support": support,
            "global_positive": positive,
            "global_negative": support - positive,
            "category_count": int(group["category"].nunique()),
            "single_difference_support": int((clean["semantic_fact_count"] == 1).sum()),
            "max_2_differences_support": int((clean["semantic_fact_count"] <= 2).sum()),
            "strong_multi_difference_support": int((clean["semantic_fact_count"] > 2).sum()),
            **global_effect,
            "example_pair_ids": json.dumps(clean["pair_id"].head(5).tolist(), ensure_ascii=False),
        }
        category_rows: list[dict[str, Any]] = []
        for category, category_group in clean.groupby("category"):
            category_positive = int(category_group["human_label"].sum())
            category_support = len(category_group)
            base = baseline.loc[category]
            effect_base = effect_baseline.loc[category]
            effect = effect_posterior(
                category_positive,
                category_support - category_positive,
                int(effect_base["positive_count"]),
                int(effect_base["negative_count"]),
                args.posterior_draws,
                stable_seed(args.seed, rid, category),
            )
            category_rows.append(
                {
                    "rule_id": rid,
                    "category": category,
                    "category_support": category_support,
                    "category_positive": category_positive,
                    "category_negative": category_support - category_positive,
                    "category_baseline": float(base["category_baseline"]),
                    "pilot_category_baseline": float(base["pilot_category_baseline"]),
                    "effect_reference": args.effect_baseline,
                    "effect_reference_baseline": float(effect_base["effect_reference_baseline"]),
                    "category_effect": effect["effect_mean"],
                    "category_effect_median": effect["effect_median"],
                    "category_effect_low_95": effect["effect_low_95"],
                    "category_effect_high_95": effect["effect_high_95"],
                    "category_effect_sd": effect["effect_sd"],
                    "probability_positive_effect": effect["probability_positive_effect"],
                    "probability_negative_effect": effect["probability_negative_effect"],
                }
            )
        category_frame = empirical_bayes_shrinkage(pd.DataFrame(category_rows), global_row["global_effect"])
        global_row["heterogeneity_tau2"] = float(category_frame["heterogeneity_tau2"].iloc[0])
        global_row["scope_candidate"] = scope_candidate(category_frame, global_row)
        global_row["posterior_direction_signal"] = (
            "POSITIVE" if global_row["global_effect_low_95"] > 0 else
            "NEGATIVE" if global_row["global_effect_high_95"] < 0 else
            "NO_CLEAR_DIRECTION"
        )
        global_row["effect_class_placeholder"] = "UNCERTAIN"
        global_row["uncertainty"] = global_row["global_effect_high_95"] - global_row["global_effect_low_95"]
        global_row["stability_placeholder"] = "NOT_CHECKED_RULE_INTERNAL_VALIDATION"
        global_rows.append(global_row)
        category_rows_all.append(category_frame)

    global_frame = pd.DataFrame(global_rows)
    category_frame = pd.concat(category_rows_all, ignore_index=True)
    rule_table = category_frame.merge(global_frame, on="rule_id", how="left", validate="many_to_one")
    global_frame.to_parquet(output_dir / "global_rule_summary.parquet", index=False)
    global_frame.to_csv(output_dir / "global_rule_summary.csv", index=False, encoding="utf-8-sig")
    rule_table.to_parquet(output_dir / "pilot_rule_table.parquet", index=False)
    rule_table.to_csv(output_dir / "pilot_rule_table.csv", index=False, encoding="utf-8-sig")
    distributions = quantiles(global_frame, ["global_support", "category_count", "global_effect", "uncertainty", "single_difference_support", "max_2_differences_support", "heterogeneity_tau2"])
    distributions.to_csv(output_dir / "rule_distributions.csv", index=False, encoding="utf-8-sig")

    def example_rows(frame: pd.DataFrame, kind: str, count: int = 8) -> pd.DataFrame:
        if kind == "positive":
            return frame.sort_values(["global_effect_median", "global_support"], ascending=[False, False]).head(count)
        if kind == "negative":
            return frame.sort_values(["global_effect_median", "global_support"], ascending=[True, False]).head(count)
        if kind == "neutral":
            return frame.assign(abs_effect=frame["global_effect_median"].abs()).sort_values(["abs_effect", "global_support"], ascending=[True, False]).head(count)
        if kind == "heterogeneous":
            return frame.sort_values(["heterogeneity_tau2", "category_count"], ascending=[False, False]).head(count)
        return frame.sort_values(["global_support", "uncertainty"], ascending=[True, False]).head(count)

    examples = []
    for kind in ("positive", "negative", "neutral", "uncertain", "heterogeneous"):
        part = example_rows(global_frame, kind).copy()
        part.insert(0, "example_group", kind.upper())
        examples.append(part)
    pd.concat(examples, ignore_index=True).to_csv(output_dir / "representative_rule_examples.csv", index=False, encoding="utf-8-sig")

    support_q = global_frame["global_support"].quantile([0, .25, .5, .75, .9, 1]).to_dict()
    category_q = global_frame["category_count"].quantile([0, .5, .9, 1]).to_dict()
    scope_counts = global_frame["scope_candidate"].value_counts().to_dict()
    class_counts = global_frame["effect_class_placeholder"].value_counts().to_dict()
    direction_counts = global_frame["posterior_direction_signal"].value_counts().to_dict()
    singleton_rules = int((global_frame["global_support"] == 1).sum())
    at_most_two_rules = int((global_frame["global_support"] <= 2).sum())
    positive_examples = example_rows(global_frame, "positive", 5)[["canonical_rule", "global_support", "global_positive", "global_negative", "global_effect_median", "uncertainty"]]
    negative_examples = example_rows(global_frame, "negative", 5)[["canonical_rule", "global_support", "global_positive", "global_negative", "global_effect_median", "uncertainty"]]
    neutral_examples = example_rows(global_frame, "neutral", 5)[["canonical_rule", "global_support", "global_positive", "global_negative", "global_effect_median", "uncertainty"]]
    sampling_note = (
        "Pilot sampler был label-aware и примерно балансировал классы; эти пары не "
        "являются вероятностной выборкой discovery. Поэтому effects ниже проверяют работу "
        "pipeline, но не являются финальными оценками правил."
        if args.sampling_design == "balanced_label_aware"
        else
        "Pilot был выбран случайно внутри category quotas без балансировки по label. "
        "Поэтому effects пригодны как предварительные estimates, но остаются pilot-оценками."
    )
    report = f"""# Pilot: semantic differences → candidate rules

## Границы

Использованы только **{len(pairs)}** уже извлечённых пар Qwen из
`RULE_DISCOVERY`. Internal validation, ordinary, hard и OOD не использовались.
Rule definitions (`concept + relation`) и label-free assignments были сохранены
до загрузки labels.

{sampling_note} Диагностический effect
сравнивается с **{args.effect_baseline} category baseline**. Полный discovery и
pilot prevalence сохранены рядом в `category_baselines.csv`.

## Grouping

- `different_value` сохраняется отдельно;
- `missing_a/missing_b` объединены в orientation-invariant `missing_one_side`;
- `subset/more_specific` объединены в `specificity_difference`;
- identity anchors не превращаются в candidate rules;
- human labels не участвуют в grouping.
- label-free concept map: **{str(args.concept_map) if args.concept_map else 'не использовался'}**.

Получено **{len(global_frame)}** candidate rules и **{len(assignments)}**
pair-rule observations. Медианный support: **{support_q[0.5]:.1f}**; p90:
**{support_q[0.9]:.1f}**; максимум: **{support_q[1.0]:.0f}**. Медианное число
categories на rule: **{category_q[0.5]:.1f}**, максимум: **{category_q[1.0]:.0f}**.
Rules с support=1: **{singleton_rules} ({singleton_rules / len(global_frame):.1%})**;
с support<=2: **{at_most_two_rules} ({at_most_two_rules / len(global_frame):.1%})**.

## Uncertainty и shrinkage

Вероятности и log-odds effects считаются через Jeffreys Beta(0.5, 0.5)
posterior. Для category deviations используется empirical-Bayes shrinkage:
межкатегориальная variance оценивается как observed variance минус sampling
variance; маленькие/шумные cells сильнее тянутся к global effect.

Жёсткие min-support/effect thresholds не задавались. Распределение effect classes:
`{json.dumps(class_counts, ensure_ascii=False)}`; scope candidates:
`{json.dumps(scope_counts, ensure_ascii=False)}`. Минимальная ширина 95% effect
interval в этом pilot равна **{global_frame['uncertainty'].min():.2f} log-odds**,
а медианный support равен одному. Отдельно сохранён posterior direction signal,
не являющийся финальным правилом: `{json.dumps(direction_counts, ensure_ascii=False)}`.

### Наиболее положительные direction candidates

{markdown_table(positive_examples)}

### Наиболее отрицательные direction candidates

{markdown_table(negative_examples)}

### Ближайшие к neutral candidates

{markdown_table(neutral_examples)}

Все эти примеры остаются `UNCERTAIN`, поскольку uncertainty очень велика.

## Clean support

Для каждого rule сохранены support в парах с ровно одним semantic difference,
не более чем двумя differences и более чем двумя differences. Это pair-level
counts; несколько rules одной пары не трактуются как независимые доказательства.

## Вывод

Формат representation позволяет детерминированно получить rule table и
category/global effects. Однако даже **{len(pairs)}** label-aware sampled pairs
пока недостаточно для надёжного выбора GLOBAL/CATEGORY_SPECIFIC и естественных
support thresholds: распределение support остаётся крайне разреженным. Перед
увеличением pilot важнее стабилизировать canonical concepts, иначе новые пары
продолжат создавать множество одноразовых formulations.

Internal-validation код на этом шаге не запускался. Поле
`stability_placeholder` оставлено для последующей замороженной проверки без
изменения definitions rules.
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"pairs": len(pairs), "rules": len(global_frame), "observations": len(assignments), "scope_counts": scope_counts, "effect_class_counts": class_counts, "posterior_direction_signals": direction_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
