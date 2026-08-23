# Pilot: semantic differences → candidate rules

## Границы

Использованы только **60000** уже извлечённых пар Qwen из
`RULE_DISCOVERY`. Internal validation, ordinary, hard и OOD не использовались.
Rule definitions (`concept + relation`) и label-free assignments были сохранены
до загрузки labels.

Pilot был выбран случайно внутри category quotas без балансировки по label. Поэтому effects пригодны как предварительные estimates, но остаются pilot-оценками. Диагностический effect
сравнивается с **discovery category baseline**. Полный discovery и
pilot prevalence сохранены рядом в `category_baselines.csv`.

## Grouping

- `different_value` сохраняется отдельно;
- `missing_a/missing_b` объединены в orientation-invariant `missing_one_side`;
- `subset/more_specific` объединены в `specificity_difference`;
- identity anchors не превращаются в candidate rules;
- human labels не участвуют в grouping.
- label-free concept map: **C:\Users\Professional\product_matching\artifacts\qwen_concept_normalization_v1_checkpoint_60000\concept_normalization_map.parquet**.

Получено **12781** candidate rules и **235318**
pair-rule observations. Медианный support: **1.0**; p90:
**13.0**; максимум: **12159**. Медианное число
categories на rule: **1.0**, максимум: **18**.
Rules с support=1: **6759 (52.9%)**;
с support<=2: **8535 (66.8%)**.

## Uncertainty и shrinkage

Вероятности и log-odds effects считаются через Jeffreys Beta(0.5, 0.5)
posterior. Для category deviations используется empirical-Bayes shrinkage:
межкатегориальная variance оценивается как observed variance минус sampling
variance; маленькие/шумные cells сильнее тянутся к global effect.

Жёсткие min-support/effect thresholds не задавались. Распределение effect classes:
`{"UNCERTAIN": 12781}`; scope candidates:
`{"UNCERTAIN": 12735, "HETEROGENEOUS": 46}`. Минимальная ширина 95% effect
interval в этом pilot равна **0.10 log-odds**,
а медианный support равен одному. Отдельно сохранён posterior direction signal,
не являющийся финальным правилом: `{"NO_CLEAR_DIRECTION": 11751, "NEGATIVE": 535, "POSITIVE": 495}`.

### Наиболее положительные direction candidates

| canonical_rule | global_support | global_positive | global_negative | global_effect_median | uncertainty |
| --- | --- | --- | --- | --- | --- |
| lubricant_type is explicitly known on only one side | 4 | 4 | 0 | 5.42966 | 8.99859 |
| toe_cap_material is explicitly known on only one side | 3 | 3 | 0 | 4.86461 | 9.08613 |
| lubricant_base is explicitly known on only one side | 2 | 2 | 0 | 4.85967 | 9.38605 |
| intended_effect has different explicit values | 2 | 2 | 0 | 4.78413 | 9.09286 |
| insert_color has different explicit values | 3 | 3 | 0 | 4.57174 | 9.04829 |

### Наиболее отрицательные direction candidates

| canonical_rule | global_support | global_positive | global_negative | global_effect_median | uncertainty |
| --- | --- | --- | --- | --- | --- |
| frame_shape has different explicit values | 1016 | 0 | 1016 | -5.86959 | 8.56752 |
| curl_type has different explicit values | 73 | 0 | 73 | -5.23156 | 8.6966 |
| curtain_width has different explicit values | 112 | 0 | 112 | -5.17485 | 8.3959 |
| lash_curl has different explicit values | 59 | 0 | 59 | -4.99826 | 8.53577 |
| cpu_model has different explicit values | 164 | 0 | 164 | -4.99575 | 8.3798 |

### Ближайшие к neutral candidates

| canonical_rule | global_support | global_positive | global_negative | global_effect_median | uncertainty |
| --- | --- | --- | --- | --- | --- |
| brush_size_number is explicitly known on only one side | 12 | 4 | 8 | -6.46404e-05 | 2.48264 |
| part_number_analog has different explicit values | 1 | 0 | 1 | -0.000273674 | 9.69571 |
| technology is explicitly known on only one side | 2 | 0 | 2 | -0.000388569 | 9.31309 |
| bracelet_size_mm has different explicit values | 1 | 0 | 1 | 0.00039302 | 9.33498 |
| frame_hinge_type is explicitly known on only one side | 1 | 0 | 1 | 0.000400548 | 9.34694 |

Все эти примеры остаются `UNCERTAIN`, поскольку uncertainty очень велика.

## Clean support

Для каждого rule сохранены support в парах с ровно одним semantic difference,
не более чем двумя differences и более чем двумя differences. Это pair-level
counts; несколько rules одной пары не трактуются как независимые доказательства.

## Вывод

Формат representation позволяет детерминированно получить rule table и
category/global effects. Однако даже **60000** label-aware sampled pairs
пока недостаточно для надёжного выбора GLOBAL/CATEGORY_SPECIFIC и естественных
support thresholds: распределение support остаётся крайне разреженным. Перед
увеличением pilot важнее стабилизировать canonical concepts, иначе новые пары
продолжат создавать множество одноразовых formulations.

Internal-validation код на этом шаге не запускался. Поле
`stability_placeholder` оставлено для последующей замороженной проверки без
изменения definitions rules.
