# Диагностика class-prior shift для frozen lexical CatBoost

Эксперимент использует сохранённые predictions первого names-only эксперимента
`01_names_lexical`. Модель не переобучалась, scores не изменялись, строки
validation не удалялись. Competition AP считается как среднее из 20
category-wise `sklearn.metrics.average_precision_score`.

Исходный validation:

- 54 887 пар, 14 128 позитивов;
- global positive prevalence: `0.257402`;
- средняя prevalence по 20 категориям: `0.261644`;
- competition macro AP: `0.501538`;
- global AP: `0.595118`;
- macro ROC-AUC: `0.735706`;
- global ROC-AUC: `0.794183`.

Различие между `25.74%` и ожидаемыми примерно `26.2%` объясняется способом
агрегации: первое число — доля позитивов среди всех validation-строк, второе —
невзвешенное среднее долей позитивов по 20 категориям.

## Weighted prevalence

Позитивы всегда имеют вес 1. Для всех негативов используется один вес:

```text
w_neg = p_original * (1 - p_target) /
        (p_target * (1 - p_original))
```

`p_original` здесь равна global prevalence `0.257402`. В таблице `effective
prevalence` является global weighted prevalence. Отдельно приведено среднее
значение weighted prevalence по категориям, поскольку итоговый AP агрегируется
по категориям с одинаковым весом.

| Target prevalence | Negative weight | Effective prevalence | Mean category prevalence | Macro AP | Macro ROC-AUC |
|---:|---:|---:|---:|---:|---:|
| original 25.740% | 1.0000 | 25.740% | 26.164% | 0.501538 | 0.735706 |
| 26.200% | 0.9764 | 26.200% | 26.586% | 0.506294 | 0.735706 |
| 20.000% | 1.3865 | 20.000% | 20.820% | 0.437295 | 0.735706 |
| 15.000% | 1.9642 | 15.000% | 16.001% | 0.371755 | 0.735706 |
| 10.000% | 3.1196 | 10.000% | 10.976% | 0.292402 | 0.735706 |
| 7.500% | 4.2750 | 7.500% | 8.365% | 0.244683 | 0.735706 |
| 6.600% | 4.9052 | 6.600% | 7.406% | 0.225611 | 0.735706 |
| 5.000% | 6.5858 | 5.000% | 5.674% | 0.188414 | 0.735706 |
| 3.000% | 11.2075 | 3.000% | 3.456% | 0.133632 | 0.735706 |

Численный поиск даёт macro AP ровно `0.21` при:

- global effective prevalence: `5.9039%`;
- mean category effective prevalence: `6.6567%`;
- negative weight: `5.5245`;
- global AP: `0.258268`;
- macro ROC-AUC: `0.735706`.

## Bootstrap sanity check

Для каждого уровня сохранены все позитивы, а негативы физически сэмплировались
с replacement отдельно внутри каждой категории. Выполнено 30 повторов с
фиксированными исходными scores.

| Target prevalence | Weighted macro AP | Bootstrap AP mean ± std | Weighted macro ROC-AUC | Bootstrap ROC-AUC mean ± std |
|---:|---:|---:|---:|---:|
| 10.0% | 0.292402 | 0.293154 ± 0.001599 | 0.735706 | 0.735815 ± 0.000688 |
| 7.5% | 0.244683 | 0.245526 ± 0.001062 | 0.735706 | 0.735650 ± 0.000577 |
| 5.0% | 0.188414 | 0.188745 ± 0.000964 | 0.735706 | 0.735625 ± 0.000595 |

Bootstrap подтверждает weighted расчёт в пределах ожидаемого sampling noise.

## Интерпретация

AP зависит от precision, а precision зависит от количества негативов на то же
число позитивов. Поэтому увеличение веса всех негативов снижает precision при
фиксированном recall и снижает AP, хотя scores и ranking не изменяются.

ROC-AUC измеряет относительное ранжирование positive/negative пар. Один и тот же
постоянный вес для всего negative-класса не меняет эти попарные сравнения.
Максимальное изменение macro ROC-AUC в weighting-таблице составило менее
`9e-15`, то есть только floating-point noise.

Вывод: одного class-prior shift математически достаточно, чтобы получить
падение competition macro AP с `0.50` до `0.21` при неизменных predictions.
Это **не означает**, что реальная public prevalence равна `5.9%` или `6.7%`.
Public labels неизвестны, а наблюдаемое падение также может включать изменение
сложности retrieval-кандидатов, covariate shift и другие причины.

Полные артефакты находятся в `reports/prevalence_shift_diagnostic/`:

- `prevalence_weighting_results.csv`;
- `per_category_weighted_metrics.csv`;
- `bootstrap_sanity_summary.csv` и `bootstrap_sanity_repeats.csv`;
- `baseline_per_category.csv`;
- `diagnostic_report.json`;
- `ap_vs_prevalence.png`.
