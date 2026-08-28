# CatBoost-1: leakage-safe early-exit experiment

## Purpose

This experiment compares four cheap CPU CatBoost classifiers for the first stage
of the matching cascade. The selection objective is maximum OOF early-exit
coverage at a strict error upper confidence bound, not maximum standalone AP.

The run reads only `data/validation_splits_v1/human_train_pairs.parquet`. It does
not read IID, hard, or OOD labels/predictions, neural-model predictions, or any
CatBoost-2 artifact. Five common `StratifiedGroupKFold` folds use connected product
components as groups. Every product ID is therefore confined to one side of each
outer split.

## Leakage rules

The Qwen checkpoint contributes only label-free feature templates:

- `accepted_facts.parquet` supplies raw attribute-name to semantic-concept aliases;
- only `rule_id`, `canonical_rule`, `concept`, and `relation` are read from
  `frozen_rule_definitions.parquet`;
- persisted support, effects, probabilities, precision, confidence, category
  effects, and human targets from rule-mining outputs are never read.

For each outer fold, global and category-rule support/evidence are recomputed from
that fold's training labels. Training rows receive leave-one-out statistics;
validation rows receive statistics from the full outer training part. Rule
probabilities use a smoothed fold baseline. This makes every label-derived rule
feature OOF-safe.

## Variants

- `V1_global`: numeric/lexical features and fold-safe global rule evidence; no
  category feature.
- `V2_global_category`: V1 plus category as a CatBoost categorical feature.
- `V3_category_aware`: V2 plus typed conflict signatures, explicit
  category/conflict interactions, and fold-safe category-rule evidence.
- `V4_matching_regimes`: V3 plus a label-free manual matching regime and its
  conflict interaction.

There is no generic numeric-conflict feature. Size, RAM/storage, volume, weight,
pack count, power, dimensions, optical values, model numbers, voltage, frequency,
and capacity each have separate match/conflict/missing/count features. This is
important because, for example, clothing-size variants and electronics-memory
variants have different matching semantics.

## Run locally on CPU

From the repository root in PowerShell:

```powershell
uv sync
uv pip install --python .venv\Scripts\python.exe -r requirements-catboost1.txt
.venv\Scripts\python.exe scripts\run_catboost1_early_exit.py --config configs\catboost1_early_exit.json
```

An optional fast integration check (not a valid experiment) is:

```powershell
.venv\Scripts\python.exe scripts\run_catboost1_early_exit.py --config configs\catboost1_early_exit.json --smoke
```

The normal run writes under `artifacts/catboost1_early_exit_v1/`:

- `RESULTS.md`: compact readable comparison and at most two selected variants;
- `summary.csv`: standalone metrics plus joint coverage at 0.1%, 0.2%, 0.5%;
- `risk_coverage_operating_points.csv`: best negative-only, positive-only, and
  joint operating points with accepted count, errors, empirical error, thresholds,
  and Wilson 95% UCB;
- `risk_coverage_curves.parquet`: complete tie-safe threshold sweeps;
- `oof_predictions.parquet`: pair IDs, target, category, component, fold, and all
  OOF `p_match` columns;
- `standalone_metrics.csv`, `category_ap.csv`, and
  `feature_importance_by_fold.csv`;
- fold CatBoost models, a reproducibility manifest, log, and `COMPLETED` marker.

Google Sheets is intentionally not touched. After inspecting the local schema and
results, a separate explicit sync can be added without coupling training to
credentials or network availability.
