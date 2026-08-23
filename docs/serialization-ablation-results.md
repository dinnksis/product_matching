# MiniLM serialization ablation: результаты

## Протокол

| Поле | Значение |
|---|---|
| Данные | Только human labels |
| Train screening subset | 120 000 пар |
| Validation | 51 470 пар, 13 661 positive, 26.54% |
| Split | `brand_model_family_holdout`, seed 42 |
| Утечки | 0 общих item ID; 0 общих family signatures |
| Модель | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` |
| Обучение | 1 epoch, BCEWithLogitsLoss, AdamW |
| LR / weight decay / warmup | `2e-5` / `0.01` / `5%` |
| Batch / effective batch | 64 / 64 на один GPU |
| Max length | 256 |
| Validation score | Среднее вероятностей для A→B и B→A |
| Primary metric | Среднее category-wise AP по 20 категориям |

Все четыре run использовали один split, train subset, normalization, seed,
optimizer и training config.

## Результаты

| Serialization | Macro PR-AUC | Δ к S0 | Overall PR-AUC | Macro ROC-AUC | Train | Val pairs/s | Avg tokens | P95 | Доля на 256 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S0_TITLE` | 0.639499 | — | 0.722293 | 0.827813 | 3:27 | 2250 | 42.9 | 77 | 0.0% |
| `S1_KEY_VALUE` | 0.686910 | +0.047411 | 0.754481 | 0.857293 | 9:06 | 427 | 210.3 | 256 | 49.3% |
| **`S2_VALUES_ONLY`** | **0.690153** | **+0.050654** | 0.753521 | **0.857968** | 7:10 | 577 | 161.2 | 256 | 18.5% |
| `S3_HYBRID` | 0.686600 | +0.047101 | **0.755168** | 0.857381 | 8:53 | 446 | 202.2 | 256 | 42.5% |

`Val pairs/s` уже включает два направления на пару. Это screening throughput на
T4, а не замер конкурсного Docker на H100.

## Сравнение вариантов

| Сравнение | Δ macro PR-AUC S2 | Отношение скорости S2 | Δ средней длины S2 |
|---|---:|---:|---:|
| S2 против S0 | +0.050654 | 0.26× | +118.4 tokens |
| S2 против S1 | +0.003243 | 1.35× | −49.1 tokens |
| S2 против S3 | +0.003553 | 1.30× | −40.9 tokens |

- Атрибуты дали основной прирост: все attribute-варианты улучшили S0 примерно на
  0.047–0.051 macro AP.
- Значения без названий ключей дали лучший macro AP и macro ROC-AUC.
- S1 имеет почти половину validation sequences на пределе 256 tokens. У S2 эта
  доля 18.5%, поэтому он сохраняет больше значений и быстрее S1/S3.
- S2 лучше S0 в 19 из 20 категорий. Между attribute-вариантами разница мала:
  S2 лучше S1 в 7 категориях и S3 в 9 категориях. По категории лучший вариант
  распределён как S1: 7, S2: 7, S3: 6.
- S3 не улучшил S1: автоматические frequent keys увеличили длину, но не дали
  прироста primary metric.
- S3 имеет лучший overall AP, однако победитель определяется заранее
  зафиксированным macro category-wise AP, поэтому выбран S2.

## HYBRID threshold

| Показатель | Значение |
|---|---:|
| Уникальные attribute names в train subset | 24 916 |
| Всего attribute occurrences | 2 996 439 |
| Автоматический item-support threshold | 694 |
| Frequent keys | 508 |
| Покрытие occurrences | 80.004% |

Threshold получен только из train statistics и не подбирался по validation.

## Решение

Победитель screening — `S2_VALUES_ONLY`. Текущий checkpoint обучен на 120 000
пар и пригоден для отдельного probing submission, который измерит перенос на
leaderboard и фактический H100 runtime. Это не финальная модель.

Следующий подтверждающий run: обучить S2 один epoch на всём доступном train pool
после фиксации holdout. Validation нельзя добавлять в обучение до завершения
локального сравнения; для финального leaderboard checkpoint допустимо затем
переобучить S2 на всех 365 654 human-labelled парах с тем же числом optimizer
steps на пример.
