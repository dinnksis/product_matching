# CatBoost-1.1 negative-only router

## Decision being tested

This is a bounded follow-up to the first CatBoost experiment. It asks whether a
cheap model can safely discard confident non-matches before CatBoost-2 and the
cross-encoders. Positive early exit is disabled because the first experiment had
no statistically useful positive tail.

Only `human_train_pairs.parquet` is read. IID, hard, OOD, neural predictions and
CatBoost-2 outputs are not used for model or threshold selection.

## Variants

- `C1A_category_no_rules`: cheap typed features plus category, with all mined-rule
  features removed. This isolates whether target-derived rule evidence was
  hurting the first experiment.
- `C1B_category_clean_rules`: A plus cleaned global rule evidence.
- `C1C_category_clean_rules_posw3`: B with positive rows weighted by three, making
  false negatives more expensive during learning.
- `C1D_regime_clean_rules_posw3`: C plus category/conflict interactions,
  category-specific clean-rule evidence and the label-free matching regime.

All four use identical outer folds and CatBoost hyperparameters.

## Rule policy

The policy follows the frozen rule-mining source rather than stored effects:

- only `rule_role == RULE_CANDIDATE`;
- only `different_value` and `specificity_difference`;
- `missing_one_side` is excluded because the mining code defines it as
  `CONTEXT_ONLY`;
- `unknown` and `conflicting_sources` are excluded because they are
  `REVIEW_BEFORE_USE`;
- persisted discovery support/effect/precision/confidence columns are never read;
- label-derived support and direction are recomputed inside each outer fold;
- training features use four inner product/component-disjoint folds rather than
  leave-one-out encoding;
- rules need fold-local support of 50 globally or 30 inside a category;
- effects use a prior strength of 50 and are clipped per rule;
- evidence is separated into brand, model, size, memory/storage, volume, weight,
  pack count, power, dimensions, optical, color, material and `other` families.

Raw label-free rule counts by relation/family are retained. Numeric conflicts
remain separate; no generic numeric-conflict feature is introduced.

## Evaluation

The primary result is negative-only fold-to-fold threshold transfer:

1. Hold out one outer OOF fold.
2. Select a Wilson-safe threshold using the other four folds.
3. Apply it unchanged to the held fold.
4. Repeat for every fold and aggregate accepted pairs/errors.

This is reported for global, category and matching-regime thresholds. A route
passes only when its own aggregated negative Wilson 95% UCB is below the risk
limit. Positive examples are never pooled into the bound. Reports also expose
category concentration, fold coverage and every calibration threshold.

## CPU run

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-catboost1.txt
.venv\Scripts\python.exe scripts\run_catboost1_negative_router.py --config configs\catboost1_negative_router.json
```

Optional integration check:

```powershell
.venv\Scripts\python.exe scripts\run_catboost1_negative_router.py --config configs\catboost1_negative_router.json --smoke
```

Results are isolated under `artifacts/catboost1_negative_router_v2/`. Start with
`RESULTS.md` and `crossfit_negative_routing.csv`. Google Sheets is not modified.
