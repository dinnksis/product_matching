# E-CUP 2026 — Product Matching

Решение задачи идентификации одинаковых товаров по названиям, категориям и
структурированным атрибутам. Retrieval выполнен организаторами: на вход решения
поступают карточки товаров и готовые пары-кандидаты, на выходе формируется
непрерывный `predict` для каждой пары.

Метрика соревнования — средний по 20 категориям
`sklearn.metrics.average_precision_score` (macro AP). Решение запускается без
интернета на NVIDIA H100 80 GB; лимит Private — 13 минут.

## Итоговая архитектура

Использованы три multilingual cross-encoder модели:

- `BAAI/bge-reranker-v2-m3` — основной backbone;
- `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — быстрый дополнительный эксперт;
- `deepvk/RuModernBERT-base` — русскоязычный дополнительный эксперт.

Все модели получают две независимо сериализованные карточки:

```text
Категория: <category>
Название: <name>
<ключ атрибута>: <значение>
...
```

Пустые атрибуты удаляются, остальные располагаются в детерминированном порядке.
На этапе final inference используется SentenceTransformers `CrossEncoder`, FP16,
SDPA, `max_length=384` для всей пары, batch 1024 и один проход `A → B`. Отказ от
симметричного `A → B` + `B → A` inference был главным runtime-ускорением.

## Подготовленные финальные submissions

| Решение | Вычисления | Итоговый score | Builder |
|---|---|---|---|
| Full BGE + MiniLM | BGE 100%, MiniLM 100% | 50/50 sigmoid probabilities | `scripts/build_bge_minilm_full_oneway_final_submit.py` |
| Full triple | BGE 100%, MiniLM 100%, RuModernBERT 100% | среднее трёх sigmoid probabilities | `scripts/build_bge_minilm_rumodern_full_oneway_submit.py` |
| Routed 40/5 | BGE 100%, MiniLM 40%, RuModernBERT 5% | последовательные 50/50 blends | `scripts/build_bge_minilm_rumodern_fast_oneway_final_40_5_submit.py` |

Также отправлялся BGE-only diagnostic, но он не рассматривается как основной
финальный ансамбль. Full triple и routed 40/5 прошли контейнерный запуск. В
routed-варианте MiniLM выбирается 7-признаковым CatBoost benefit-router, а
RuModernBERT — 14-признаковым sequential router внутри MiniLM subset. Эти routers
обучались на component-disjoint OOF predictions предыдущей версии нейросетевых
checkpoints; это ограничение явно записано в submission manifest.

ZIP-файлы и веса не хранятся в Git. После размещения финальных checkpoints в
`configs/bge_final`, `configs/minilm_final` и `configs/rumodernbert_final` сборка
выполняется командами:

```powershell
.\.venv\Scripts\python.exe scripts\build_bge_minilm_full_oneway_final_submit.py
.\.venv\Scripts\python.exe scripts\build_bge_minilm_rumodern_full_oneway_submit.py
.\.venv\Scripts\python.exe scripts\build_bge_minilm_rumodern_fast_oneway_final_40_5_submit.py
```

Builder поддерживает обычный `model.safetensors`, Hugging Face sharding и
транспортные части `model.safetensors.partNNN` с manifest-проверкой SHA-256.

## Финальное human fine-tuning MiniLM и RuModernBERT

Этот репозиторий фиксирует последний этап обучения двух специалистов на всех
`365 654` human-labelled парах без validation holdout и без фильтрации. Выбор
гиперпараметров был сделан на более ранних component-disjoint экспериментах;
финальный all-human fit не использовался для выбора параметров.

MiniLM стартует с checkpoint после пяти эпох на предоставленной организаторами
LLM-разметке и затем обучается три эпохи на human labels:

| Параметр | Значение |
|---|---:|
| Epochs | 3 |
| Per-GPU batch | 96 |
| GPU / accumulation | 2 / 1 |
| Effective batch | 192 |
| Learning rate | `8e-5` |
| Weight decay / warmup | `0.01` / `0.05` |
| Classifier dropout | `0.1` |
| Max grad norm | `1.0` |

RuModernBERT стартует с checkpoint после трёх эпох competition-pretrain и также
дообучается три эпохи на всех human labels:

| Параметр | Значение |
|---|---:|
| Epochs | 3 |
| Per-GPU batch / accumulation | 24 / 4 |
| GPU / effective batch | 2 / 192 |
| Learning rate | `4e-5` |
| Scheduler | cosine |
| Weight decay / warmup | `0.01` / `0.05` |
| Max grad norm | `0.5` |

Общее для обеих моделей: seed 42, FP16, SDPA, BCE loss, `max_length=384`,
детерминированная сериализация из `src/data_pipeline.py`, sampling/weights/label
smoothing отключены. Точные configs:

- `configs/cross_encoder_minilm_5ep_full_human_final.json`;
- `configs/cross_encoder_rumodernbert_3ep_full_human_final.json`.

Kaggle notebook генерируется из текущего source bundle и перед запуском проверяет
число строк, SHA-256 исходников, locked config, наличие двух GPU и итоговый training
report. Запуск:

```powershell
.\.venv\Scripts\python.exe scripts\run_minilm_5ep_full_human_kaggle.py --no-wait --no-download
.\.venv\Scripts\python.exe scripts\run_rumodernbert_3ep_full_human_kaggle.py --no-wait --no-download
```

Подробный контракт входов, outputs и проверок описан в
[`docs/final-human-finetuning.md`](docs/final-human-finetuning.md).

## Воспроизводимость: границы текущего commit

В текущем состоянии полностью зафиксированы:

- финальный all-human этап MiniLM и RuModernBERT;
- общая сериализация;
- inference трёх итоговых архитектур;
- сборка submission ZIP и проверка checkpoint hashes;
- код обучения CatBoost routers и их configs.

Отдельным командным вкладом должны быть добавлены исходные скрипты и configs
competition-pretrain для MiniLM/RuModernBERT, а также точный training pipeline
финального BGE. Приватные Kaggle checkpoint datasets не считаются единственным
воспроизводимым источником: они являются кэшем результатов предыдущего этапа.

Организаторские parquet и модельные веса не публикуются в Git. Их ожидаемые пути,
размеры и SHA-256 фиксируются генераторами notebooks и submission manifests.

## Лицензии

Все три upstream модели заявлены авторами под Apache License 2.0. Точные ссылки,
назначение моделей и основные runtime-зависимости перечислены в
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Проприетарные LLM API в
финальном training/inference pipeline не используются. 11M probabilistic labels
предоставлены организаторами соревнования; самостоятельная проприетарная
переразметка в финальные checkpoints не входит.

## Структура

```text
configs/       фиксированные experiment configs; веса *_final игнорируются Git
docs/          протоколы, результаты и инструкции
notebooks/     сгенерированные Kaggle notebooks
scripts/       обучение, OOF/router experiments и submission builders
src/           сериализация, splits, features и training utilities
submits/       лёгкие runtime-шаблоны; ZIP и model weights игнорируются Git
tests/         проверки locked training recipes и notebook contracts
```

История основных ансамблевых и routing-экспериментов находится в
[`docs/final-ensemble-architecture.md`](docs/final-ensemble-architecture.md).
