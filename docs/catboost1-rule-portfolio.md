# CatBoost-1.2: OOF-safe rule portfolio

## Задача

Эксперимент проверяет, могут ли условия из Qwen rule discovery на 60k парах
расширить отрицательный early-exit поверх `C1A_category_no_rules`. CatBoost
повторно не обучается: используются его сохранённые component-disjoint OOF
предсказания и label-free feature cache.

## Защита от leakage

Из frozen rule registry читаются только:

- `rule_id`;
- `canonical_rule`;
- `concept`;
- `relation`;
- `rule_role`.

Поля discovery/validation `support`, `positive`, `negative`, `effect`,
`precision`, `confidence`, `stability`, `generation_status` и
`allowed_categories` не читаются. Условия правил используются только как
шаблоны. Для каждого held outer fold поддержка, ошибки, направление,
category-scope, score cap, veto и состав портфеля выбираются на остальных
четырёх фолдах.

Итоговый Wilson bound считается на объединении всех held-fold решений, а не на
каждом отдельном правиле. IID, Hard, OOD, neural predictions и CatBoost-2 не
загружаются.

## Кандидаты

- отдельные `concept + relation` из frozen 60k registry;
- те же условия внутри категории;
- `rule + anchor`: одинаковый бренд/model/code, высокая похожесть или exact
  title, подтверждение code-конфликта в title;
- составные доменные конфликты: optical, technology, quantity, dimensions,
  identifiers — отдельно `>=2`, `>=3` и несколько осмысленных сочетаний;
- label-free veto для сильного identity evidence и variant-tolerant конфликтов.

Для защиты от случайных редких находок gate должен иметь минимум 100
calibration-срабатываний и встречаться минимум в трёх calibration folds.

## Запуск

Быстрая проверка интеграции:

```powershell
.venv\Scripts\python.exe scripts\run_catboost1_rule_portfolio.py --config configs\catboost1_rule_portfolio.json --smoke
```

Полный CPU-эксперимент:

```powershell
.venv\Scripts\python.exe scripts\run_catboost1_rule_portfolio.py --config configs\catboost1_rule_portfolio.json
```

Результаты будут в `artifacts/catboost1_rule_portfolio_v1/`. Сначала смотреть:

- `RESULTS.md`;
- `crossfit_summary.csv`;
- `fold_policies.json`;
- `crossfit_accepted_pairs.parquet`;
- `leakage_audit.json`.

`added_vs_baseline` показывает именно новые пары сверх C1A. Google Sheets не
изменяется.
