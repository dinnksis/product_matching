# Selective specialist analysis: BGE → MiniLM / RuModernBERT

Эксперимент использует только сохранённые predictions и label-free признаки. Обучение, submission и Google Sheets sync не выполнялись.

## Важные ограничения

- Train/OOF neural predictions для этих трёх checkpoint отсутствуют. Поэтому binary correctness использует заранее фиксированный threshold `0.5`, а не подобранный на IID threshold.
- Oracle использует label и ранжирует пары по уменьшению `|score-label|`. Это label-aware upper-bound политики directional routing, но не доказанный комбинаторный максимум macro AP.
- Все fixed probability weights заданы заранее: 25/50/75% specialist. Они не подбирались на IID/Hard/OOD.
- `mean_normalized_rank` — диагностический прежний ensemble-вариант; для реального selective inference полный specialist rank заранее неизвестен.
- OOD category identity не используется как production slice.

## Одиночные модели и pairwise proxies

| split | specialist | BGE macro AP | specialist macro AP | Δ AP | binary net @0.5 | mean logloss gain | mean MAE gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| iid | minilm | 0.782222 | 0.782508 | +0.000286 | +19 | -0.001240 | -0.001960 |
| iid | rumodernbert | 0.782222 | 0.773296 | -0.008926 | -64 | -0.004257 | -0.017831 |
| hard | minilm | 0.375975 | 0.368075 | -0.007900 | -4 | -0.006095 | -0.001349 |
| hard | rumodernbert | 0.375975 | 0.350399 | -0.025576 | -110 | +0.025543 | -0.020540 |
| ood | minilm | 0.641270 | 0.616523 | -0.024748 | -142 | -0.007684 | -0.009131 |
| ood | rumodernbert | 0.641270 | 0.633022 | -0.008248 | -57 | +0.009778 | -0.016239 |

## Пересечение полезных corrections по MAE proxy

| split | MiniLM better | RuModern better | both | MiniLM only | RuModern only |
| --- | ---: | ---: | ---: | ---: | ---: |
| hard | 1958 | 1631 | 982 | 976 | 649 |
| iid | 3558 | 2573 | 1587 | 1971 | 986 |
| ood | 10586 | 8608 | 5145 | 5441 | 3463 |

## Oracle routing upper bound

### IID

| route budget | oracle MiniLM gain | oracle RuModern gain | oracle best-expert gain |
| ---: | ---: | ---: | ---: |
| 5% | +0.042996 | +0.043657 | +0.053836 |
| 10% | +0.058507 | +0.061596 | +0.076514 |
| 15% | +0.068891 | +0.070718 | +0.091404 |
| 20% | +0.074110 | +0.076795 | +0.100821 |
| 30% | +0.079311 | +0.078893 | +0.112443 |
| 40% | +0.080375 | +0.078893 | +0.114579 |

### HARD

| route budget | oracle MiniLM gain | oracle RuModern gain | oracle best-expert gain |
| ---: | ---: | ---: | ---: |
| 5% | +0.026062 | +0.024691 | +0.029806 |
| 10% | +0.045445 | +0.045647 | +0.054609 |
| 15% | +0.060084 | +0.058353 | +0.077079 |
| 20% | +0.069885 | +0.072742 | +0.094566 |
| 30% | +0.083588 | +0.088190 | +0.122121 |
| 40% | +0.088379 | +0.089199 | +0.140301 |

### OOD

| route budget | oracle MiniLM gain | oracle RuModern gain | oracle best-expert gain |
| ---: | ---: | ---: | ---: |
| 5% | +0.072292 | +0.071736 | +0.090521 |
| 10% | +0.096109 | +0.098430 | +0.130060 |
| 15% | +0.106199 | +0.111259 | +0.150353 |
| 20% | +0.111931 | +0.118199 | +0.161897 |
| 30% | +0.116966 | +0.120949 | +0.173626 |
| 40% | +0.117749 | +0.120949 | +0.176853 |

## Стабильные label-free slices

Строгий флаг требует минимум 100 строк, положительный logloss/MAE gain, неотрицательную binary net correction и положительный AP delta, когда AP определён, одновременно на IID и Hard.

| specialist | slice | value | IID N | Hard N | IID ΔAP | Hard ΔAP | IID logloss gain | Hard logloss gain | OOD better |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| minilm | numeric_volume_state | match | 984 | 452 | +0.002695 | +0.023283 | +0.006843 | +0.007596 | False |

## Артефакты

- `pairwise_summary.csv` — общие pairwise corrections;
- `pairwise_corrections.parquet` — per-example proxy labels;
- `expert_overlap_summary.csv` — пересечение полезных corrections двух specialists;
- `slice_metrics.csv` и `stable_slices.csv` — все slices и межsplit устойчивость;
- `oracle_routing_results.csv` — полный перебор budgets/policies/fixed scores;
- `oracle_budget_summary.csv` — основной компактный label-aware результат только для probability scores;
- `oracle_budget_summary_all_modes.csv` — диагностический вариант, где дополнительно разрешён normalized-rank blend;
- `representative_examples.csv` — наиболее сильные исправления и регрессии;
- `manifest.json` — источники, hashes и ограничения.
