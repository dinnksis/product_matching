# CatBoost-2 trust/error router

## Цель

CatBoost-2 не повторяет классификацию match/non-match. Он оценивает вероятность
ошибки зафиксированного `C1A_category_no_rules`:

```text
q_error = P((p_cb1 >= 0.5) != target)
```

При `q_error < threshold` решение первого CatBoost принимается как early exit.
Один router обслуживает positive и negative decisions; раздельных моделей нет.

## Строгий nested stacking

Обычного обучения второго уровня на одной OOF-колонке недостаточно для строгой
внешней оценки: C1-модель для одной training-строки могла видеть метки будущего
held fold. Поэтому эксперимент использует вложенную схему:

1. Сохранённые пять component-disjoint C1 OOF predictions задают внешний target
   и C1 score на каждом held fold.
2. Для каждой пары из пяти фолдов обучается C1 с тем же frozen config на
   остальных трёх фолдах — всего 10 моделей.
3. Для outer fold `H` строки остальных фолдов получают C1 scores от моделей,
   не видевших ни `H`, ни компонент самой строки.
4. C2 обучается на этих inner-OOF scores и применяется к `H`.

Таким образом, метки outer-held fold не используются ни C1, ни C2 при получении
его `q_error_oof`.

## Признаки

- все 118 frozen C1A cheap features и category;
- `p_cb1`, logit, distance from 0.5, uncertainty, entropy, predicted class;
- без rules, embeddings и neural predictions.

Для IID/Hard/OOD обучаются deployment C1 на всём human train и deployment C2 на
исходных genuine C1 OOF predictions. Пороги фиксируются по train OOF до загрузки
validation features и переносятся без изменений.

## Запуск

Smoke:

```powershell
.venv\Scripts\python.exe scripts\run_catboost2_trust_router.py --config configs\catboost2_trust_router.json --smoke
```

Полный CPU-эксперимент:

```powershell
.venv\Scripts\python.exe scripts\run_catboost2_trust_router.py --config configs\catboost2_trust_router.json
```

Основные результаты появятся в `artifacts/catboost2_trust_router_v1/`:

- `RESULTS.md`;
- `crossfit_risk_coverage.csv`;
- `oof_predictions.parquet`;
- `q_error_calibration.csv`;
- `frozen_train_oof_thresholds.csv`;
- `iid_hard_ood_routing.csv`;
- `iid_predictions.parquet`, `hard_predictions.parquet`, `ood_predictions.parquet`.

Google Sheets не изменяется.
