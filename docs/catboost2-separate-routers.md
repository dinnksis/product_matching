# CatBoost-2 separate NEG/POS trust routers

## Проверяемая гипотеза

Один global trust-router из Experiment 1 может смешивать разные механизмы
ошибок. Этот эксперимент обучает две модели с теми же признаками:

- `Router_NEG` только на C1 predicted negative, target — false negative;
- `Router_POS` только на C1 predicted positive, target — false positive.

Mined rules, embeddings и neural predictions не используются.

## OOF-протокол

Протокол совпадает со строгим Experiment 1. Десять nested C1-моделей создают
inner-OOF scores, не видевшие ни outer held fold, ни компонент оцениваемой
training-строки. На каждом outer fold отдельно обучаются `Router_NEG` и
`Router_POS`. Global-router predictions берутся без изменения из Experiment 1.

Пороги `threshold_neg` и `threshold_pos` различаются, но выбираются совместно:
максимизируется total accepted при едином combined Wilson UCB. Дополнительно
сохраняются coverage, errors и Wilson UCB каждого направления.

Для deployment две полные модели обучаются на genuine C1 OOF predictions.
Train thresholds фиксируются до чтения IID/Hard/OOD predictions и затем
переносятся без изменений.

## Запуск

```powershell
.venv\Scripts\python.exe scripts\run_catboost2_separate_routers.py --config configs\catboost2_separate_routers.json
```

Smoke:

```powershell
.venv\Scripts\python.exe scripts\run_catboost2_separate_routers.py --config configs\catboost2_separate_routers.json --smoke
```

Основные файлы в `artifacts/catboost2_separate_routers_v1/`:

- `RESULTS.md`;
- `crossfit_comparison.csv`;
- `crossfit_separate_threshold_details.csv`;
- `frozen_separate_thresholds.csv`;
- `error_detection_metrics.csv`;
- `iid_hard_ood_comparison.csv`;
- `oof_predictions.parquet`.

`nested_cb1_scores.npy` сохраняется для следующих trust-router экспериментов,
чтобы больше не обучать десять C1-моделей повторно. Google Sheets не изменяется.

