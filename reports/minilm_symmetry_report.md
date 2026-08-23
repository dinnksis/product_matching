# MiniLM symmetry experiment

Проверены baseline BCE и BCE + random swap. Diagnostic выполнен на уже обученном baseline BCE checkpoint; validation protocol и сериализация совпадают с основными loss-экспериментами.

## Сравнение AP

| run | compute | IID AP | Hard AP | OOD AP | mean symmetry error (IID / hard / OOD) | p99 symmetry error (IID / hard / OOD) |
|---|---|---:|---:|---:|---|---|
| BCE baseline | diagnostic: 274 s, без обучения | 0.789655 | 0.365629 | 0.642225 | 0.02049 / 0.02542 / 0.01950 | 0.17093 / 0.19464 / 0.15483 |
| BCE + random swap | training: 945 s (+7.6% к baseline training wall) | 0.789434 | 0.364978 | 0.640881 | 0.02091 / 0.02599 / 0.02029 | 0.17450 / 0.19042 / 0.16022 |

Дельта random swap относительно BCE: IID -0.000222, Hard -0.000651, OOD -0.001344.

## Diagnostic baseline asymmetry

| split | median | p90 | p95 | p99 | max | Pearson | Spearman | >0.05 | >0.10 | >0.20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| IID | 0.00662 | 0.05761 | 0.08980 | 0.17093 | 0.43357 | 0.99263 | 0.99133 | 11.83% | 4.12% | 0.61% |
| Hard | 0.00854 | 0.07315 | 0.10523 | 0.19464 | 0.54869 | 0.98913 | 0.99216 | 15.86% | 5.61% | 0.98% |
| OOD | 0.00782 | 0.05161 | 0.07977 | 0.15483 | 0.77293 | 0.99111 | 0.97951 | 10.39% | 3.14% | 0.38% |

Mean error: IID 0.02049, Hard 0.02542, OOD 0.01950.

Для baseline AP по ориентациям и среднему score:

| split | AP AB | AP BA | AP average |
|---|---:|---:|---:|
| IID | 0.788109 | 0.787677 | **0.789655** |
| Hard | 0.364345 | **0.368868** | 0.365629 |
| OOD | 0.640068 | 0.640366 | **0.642225** |

Асимметрия не является специфичной проблемой hard negatives: средняя ошибка у baseline positive выше, чем у negative.

| split | positive mean / p99 | negative mean / p99 |
|---|---:|---:|
| IID | 0.03171 / 0.20828 | 0.01655 / 0.15422 |
| Hard | 0.03453 / 0.21295 | 0.02231 / 0.18593 |
| OOD | 0.03110 / 0.18692 | 0.01618 / 0.13918 |

## Решение по explicit symmetry regularization

Третий эксперимент (`BCE + lambda_sym * MSE`, второй forward на 25% batch) **пока не запускать**.

Причины:

1. Baseline действительно не идеально симметричен, особенно на Hard: 15.9% пар имеют ошибку больше 0.05, p99 равен 0.195.
2. Но random swap не исправил asymmetry и ухудшил AP на всех трёх split-ах.
3. Текущий train sampler уже случайно выбирает ориентацию каждой пары, поэтому дополнительный row-level random swap почти не добавляет эффективного сигнала.
4. Asymmetry сильнее выражена у positive, чем у negative; связь именно с hard-negative ошибками не подтверждается.
5. Валидационное усреднение AB/BA уже включено в frozen recipe. Оно полезно для IID/OOD примерно на 0.0015–0.002 AP относительно одной ориентации, но это inference averaging, а не необходимость менять обучение.

Вывод: asymmetry существует как диагностический эффект, но текущие данные не показывают, что consistency training улучшит ranking. Приоритет следует оставить hard-negative/category-specific направлениям. Explicit run имеет смысл возвращать только при отдельной гипотезе о пользе consistency regularization, а не как продолжение этой абляции.
