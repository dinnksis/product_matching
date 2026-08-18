# MiniLM loss sweep: итоговый отчёт

Дата: 2026-08-17. Все запуски использовали один и тот же MiniLM checkpoint после 5 эпох pretraining и фиксированный downstream recipe. Все пять результатов синхронизированы с `experiments_v2`.

## Основная таблица

| run | IID AP | Hard AP | OOD AP | hard neg p99 | hard neg >0.9 | inversion rate | notes |
|---|---:|---:|---:|---:|---:|---:|---|
| BCE | **0.789655** | **0.365629** | **0.642225** | 0.9705 | 4.94% | **0.3505** | baseline |
| BCE + RankNet | 0.789156 | 0.363859 | 0.639918 | 0.9690 | 4.96% | 0.3528 | lambda=1 |
| BCE + top-k RankNet | 0.789325 | 0.359828 | 0.638676 | 0.9675 | 4.62% | 0.3594 | lambda=1, top 50% negatives |
| negative-focused focal, gamma=1 | 0.787790 | 0.365416 | 0.638562 | 0.9662 | 4.38% | 0.3523 | |
| negative-focused focal, gamma=2 | 0.785420 | **0.365634** | 0.634771 | **0.9602** | **4.15%** | 0.3537 | |

Inversion rate — доля всех same-category negative-positive сравнений, где `score_negative > score_positive`.

## ROC-AUC

| run | IID | Hard | OOD |
|---|---:|---:|---:|
| BCE | **0.9258** | **0.7021** | 0.8744 |
| BCE + RankNet | 0.9259 | 0.7021 | **0.8786** |
| BCE + top-k RankNet | 0.9250 | 0.6995 | 0.8754 |
| focal gamma=1 | 0.9245 | 0.7001 | 0.8718 |
| focal gamma=2 | 0.9233 | 0.6989 | 0.8698 |

## Batch diagnostics

Одинаковы для всех вариантов благодаря фиксированному random sampler:

- уникальных категорий в batch: 16.22;
- категорий одновременно с positive и negative: 10.06;
- valid same-category comparisons: 121.88;
- batches с weak ranking signal: 15.7%.

Ranking-сигнал присутствует, хотя примерно в каждом шестом диагностическом окне он слабый.

## Вывод

Focal действительно уменьшает верхний хвост negative scores: при gamma=2 hard p99 снижается с 0.9705 до 0.9602, а доля `score > 0.9` — с 4.94% до 4.15%. Но Hard AP практически не растёт, а IID и особенно OOD ухудшаются.

Обычный RankNet не улучшает Hard AP или inversion rate. Top-k RankNet сильнее подавляет tail среди ranking-вариантов, но даёт худший Hard AP.

Гипотеза о том, что качество определяется только небольшим верхним хвостом hard negatives, не подтверждена. Простое подавление хвоста без улучшения category-specific ordering недостаточно. В рамках этого контролируемого sweep лучший общий результат у BCE baseline.

Полные negative-score statistics (`mean`, `p90`, `p95`, `p99`, `p99.9`, `max`, thresholds) рассчитаны из validation prediction parquet в `artifacts/kaggle/minilm_loss_sweep/`.
