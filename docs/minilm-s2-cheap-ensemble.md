# MiniLM S2 + дешёвые product-specific признаки

## Протокол

- Данные: 51 470 human-пар из фиксированной component-disjoint validation S2.
- Neural feature: замороженный симметричный score лучшего `S2_VALUES_ONLY`; MiniLM не переобучался.
- Meta-validation: 5-fold OOF с разбиением по объединённым item/model-family components.
- Модели: LogisticRegression и CatBoost (`600` деревьев, depth `6`), без sweep.
- Веса: одинаковый суммарный вес каждой категории; hard examples отдельно не перевзвешивались.
- Primary metric: mean category-wise PR-AUC (macro AP).

## Результат

| Вариант | Macro PR-AUC | Абсолютный прирост | Относительный прирост |
|---|---:|---:|---:|
| MiniLM S2 transformer score | 0.690153 | — | — |
| S2 + LogisticRegression | 0.690708 | +0.000555 | +0.08% |
| S2 + CatBoost | **0.703123** | **+0.012970** | **+1.88%** |

Полный CPU pipeline, включая построение признаков, 5 OOF-фолдов и две финальные
модели, занял 505 секунд. Это не inference benchmark submission.

Локальный end-to-end CPU-check CatBoost-обвязки на всех 51 470 validation-парах
с lexical placeholder вместо GPU forward занял 81,8 секунды и записал все строки
без NaN. Полный competition runtime ещё требует Docker+GPU Check: эти 81,8 с
добавляются к MiniLM inference и чтению данных.

## Hard-looking срезы

| Срез | Пар | S2 | LogisticRegression | CatBoost |
|---|---:|---:|---:|---:|
| Numeric/model/memory conflict-like | 4 761 | 0.746568 | 0.755109 | **0.757331** |
| SKU с одной стороны, human-like title с другой | 7 319 | 0.661729 | 0.660689 | **0.671840** |

На 3 275 hard negatives средний score уменьшился `0.0749 → 0.0685`; на 991
hard positives вырос `0.2485 → 0.2800`. Это согласуется с гипотезой, что явные
числовые/модельные конфликты и SKU дополняют нейронную семантику.

## Что добавил CatBoost

Главный признак — `transformer_score` (importance `43.77`). Затем идут char
TF-IDF cosine (`5.29`), категория (`5.25`), token Jaccard (`4.28`), длины title,
fuzzy similarity и сравнения attribute values. Числовые признаки дают
дополнительный сигнал (`number_unmatched_count` importance `2.50`), но модель
не считает любое несовпавшее число автоматическим конфликтом: отдельно
обрабатываются units, slash-specs и alphanumeric product codes.

## Вывод

Для submission основной кандидат — **MiniLM S2 + CatBoost**. LogisticRegression
оставлен как контрольный дешёвый вариант, но его прирост практически нулевой.
Оценка остаётся локальной: meta-model обучен на leakage-safe OOF внутри
замороженного human holdout, однако отдельного внешнего human holdout после
обучения meta-model нет. Leaderboard submission нужен как проверка переноса.

Артефакты: `artifacts/cheap_ensemble_s2/`.
