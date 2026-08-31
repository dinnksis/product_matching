# E-CUP 2026 — Product Matching

Репозиторий решения задачи идентификации одинаковых товарных карточек. Retrieval
выполнен организаторами: на вход подаются товары и готовые пары-кандидаты, на
выходе требуется непрерывный `predict` для каждой пары.

Метрика — средний по 20 категориям
`sklearn.metrics.average_precision_score` (macro AP). Решение работает без
интернета на NVIDIA H100 80 GB; лимит Private — 13 минут.

## Итог и главные ссылки

- [Полная история, результаты и ограничения интерпретации](docs/final-results-and-experiment-history.md)
- [Все эксперименты и paired-сравнения в Google Sheets](https://docs.google.com/spreadsheets/d/1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA/edit?usp=sharing)
- [Хронология reranker/cross-encoder экспериментов](docs/reranker-experiments.md)
- [Архитектура ансамблей и selective routing](docs/final-ensemble-architecture.md)
- [Финальное all-human обучение MiniLM и RuModernBERT](docs/final-human-finetuning.md)

Основное направление после controlled-абляций — supervised multilingual
cross-encoder на архитектуре XLM-R/BGE. Финальный подготовленный BGE submission
использует трёхэпоховый H100 checkpoint, plain BCE, LR `2e-5` и
`max_length=384`. MiniLM и RuModernBERT сохранены как дополнительные эксперты.

### Подтверждённые validation-результаты

| модель/рецепт | IID macro AP | hard macro AP | статус |
|---|---:|---:|---|
| MiniLM, 3 ep, LR `8e-5`, BCE | 0.808502 | 0.423286 | лучший MiniLM anchor |
| BGE, 1 ep, LR `2e-5`, BCE | 0.818291 | 0.414717 | BGE baseline |
| BGE, 2 ep, LR `2e-5`, BCE | **0.823461** | 0.437775 | controlled winner |
| BGE, 2 ep, sqrt category×class BCE | 0.822150 | 0.431759 | хуже plain BCE |
| BGE, 3 ep на H100, symmetric eval | **0.824975** | **0.461148** | другой стартовый checkpoint |
| тот же H100 checkpoint, single-pass eval | 0.823629 | 0.462631 | submission-oriented inference |

Former OOD split включён в BGE SFT train, поэтому OOD для BGE не является
валидацией: predictions не строятся, в отчётах записывается `-1`. IID — primary
split, hard — диагностический. Это локальные component-disjoint результаты, не
leaderboard и не hidden-test score.

Трёхэпоховый H100 run стартует с другого checkpoint, чем controlled BGE e1/e2
линия, поэтому его нельзя трактовать как чистую абляцию третьей эпохи.

## Подготовленные inference-варианты

| вариант | вычисления | статус |
|---|---|---|
| BGE 3ep H100 | один BGE forward, longer-card-first | основной single-model bundle |
| Full BGE + MiniLM | обе модели на 100% пар | production runtime anchor |
| Routed 40/5 | BGE 100%, MiniLM 40%, RuModernBERT 5% | итоговый сабмит с CatBoost routing |
| Full triple | BGE + MiniLM + RuModernBERT на 100% | итоговый сабмит — полный ансамбль |

Для current final checkpoints builders используют one-way `A → B`, FP16,
SDPA, batch 1024 и combined `max_length=384`. Full BGE+MiniLM смешивает
probabilities 50/50. В routed 40/5 последовательно применяются compact CatBoost
benefit routers; они обучены на OOF предыдущих checkpoints, поэтому этот вариант
является проверкой переноса routing policy, а не свежим OOF-оптимумом.

Исходники bundles:

- `submits/bge-reranker-v2-m3-3ep-h100/`;
- `submits/bge-minilm-full-oneway-final/`;
- `submits/bge-minilm-rumodern-fast-oneway-40-5-st-final-v1/`;
- `submits/bge-minilm-rumodern-full-oneway-st-final-v1/`.

Веса и ZIP не хранятся в Git. BGE 3ep checkpoint опубликован в private Kaggle
Dataset [`product-matching-bge-3ep-h100-oodtrain`](https://www.kaggle.com/datasets/alexproger23/product-matching-bge-3ep-h100-oodtrain),
version 1. SHA-256 модели:
`d7e899ea3cd305db970aa6f3466eb71a138ad418c74b8b6ac730d1828c4a4ab8`.

## Данные и валидация

| файл | строки | схема |
|---|---:|---|
| `data/items_human.parquet` | 711 304 | `id`, `name`, `attributes`, `category` |
| `data/matches.parquet` | 365 654 | `id1`, `id2`, `target` |

В human labels 93 890 positives. Нет missing IDs, self-pairs, повторов
неупорядоченных пар и cross-category пар. Validation делится по компонентам
графа: карточка не может одновременно находиться в train и validation.

Parquet-файлы игнорируются Git. Подготовка и структура описаны в
[`data/README.md`](data/README.md), результаты EDA — в
[`docs/data-findings.md`](docs/data-findings.md).

## Воспроизведение

Нужны Python 3.11–3.12 и [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make report
```

### BGE на одной H100

Controlled 3ep server workflow:

```bash
scripts/run_bge_3ep_h100.sh /absolute/path/to/checkpoint
```

Подробности: [`docs/bge-3ep-sft-h100.md`](docs/bge-3ep-sft-h100.md). Отдельный
all-human export без holdout описан в
[`docs/bge-3ep-fulltrain-h100.md`](docs/bge-3ep-fulltrain-h100.md).

### Kaggle 2×T4

Локальный `.env` содержит Kaggle credentials и никогда не коммитится. Общий
workflow — [`docs/kaggle-notebook.md`](docs/kaggle-notebook.md). Типичный dry-run
и запуск:

```bash
make kaggle-dry-run NOTEBOOK=notebooks/train.ipynb
make kaggle-run NOTEBOOK=notebooks/train.ipynb
```

Generators собирают notebook и exact code/data bundle, проверяют SHA-256,
private Dataset refs и GPU preflight. Финальные outputs пишутся в
`/kaggle/working` и скачиваются в игнорируемый `artifacts/kaggle/`.

### Final all-human specialists

MiniLM и RuModernBERT дообучаются на всех 365 654 human-labelled парах после
выбора гиперпараметров на component-disjoint holdout:

```bash
.venv/bin/python scripts/run_minilm_5ep_full_human_kaggle.py --dry-run
.venv/bin/python scripts/run_rumodernbert_3ep_full_human_kaggle.py --dry-run
```

Рецепты зафиксированы в
`configs/cross_encoder_minilm_5ep_full_human_final.json` и
`configs/cross_encoder_rumodernbert_3ep_full_human_final.json`.

### Submission builders

```bash
.venv/bin/python scripts/build_bge_3ep_h100_submit.py
.venv/bin/python scripts/build_bge_minilm_full_oneway_final_submit.py
.venv/bin/python scripts/build_bge_minilm_rumodern_full_oneway_submit.py
.venv/bin/python scripts/build_bge_minilm_rumodern_fast_oneway_final_40_5_submit.py
```

Builders принимают single/sharded safetensors и транспортные
`model.safetensors.partNNN`, проверяют manifests и не изменяют source checkpoint.

## Что было исследовано

- lexical/JSON baselines и CatBoost;
- Qwen3 и Jina zero-shot rerankers;
- MiniLM LR/epoch/regularization/loss sweeps;
- BGE LR, epoch и loss ablations;
- RuModernBERT и архитектурное diversity;
- BGE/MiniLM/RuModern ensembles и learned selective routing;
- hard-label audits и human-error curriculum;
- standalone Qwen item generation, rule-first pair generation, near duplicates,
  soft-positive tiers, statistical/semantic atomic rules;
- 11M probabilistic LLM labels и отдельные pretrain/server pipelines.

Data-generation инфраструктура находится в [`item_pipeline/`](item_pipeline/README.md).
Ни одна synthetic ветка не заменила финальный human-supervised BGE recipe.
Точные метрики, runtime и статусы всех запусков собраны в
[Google Sheets](https://docs.google.com/spreadsheets/d/1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA/edit?usp=sharing).

## Контракт submission

CLI должен принять `--items_path`, `--matches_path`, `--output_path` и записать
ровно `id1,id2,predict`, сохранив все пары и непрерывные finite scores. Offline
runtime: 20 CPU, 200 GB RAM, H100 80 GB; лимиты Check/Public/Private —
1/6/13 минут.

Перед публикацией bundle проверяется на:

- отсутствие internet/runtime downloads;
- полный model load без bypass;
- exact pair/order contract и отсутствие NaN/inf;
- writable caches рядом с output;
- soft deadline и полный fallback;
- лицензии и third-party notices.

## Структура репозитория

```text
configs/       locked experiment configs; локальные *_final weights игнорируются
docs/          протоколы, результаты и инструкции
item_pipeline/ генерация карточек и rule-first пар
notebooks/     EDA и сгенерированные Kaggle training notebooks
reports/       компактные summaries, receipts и paired comparisons
scripts/       обучение, audits, launchers и submission builders
src/           сериализация, splits, metrics, routers и training utilities
submits/       runtime source bundles; ZIP и model weights игнорируются
tests/         contract, provenance и regression tests
```

Raw data, checkpoints, credentials, model weights и локальные `artifacts/` не
попадают в Git. Крупные воспроизводимые derived tables, превышающие практические
лимиты GitHub, перечислены в `.gitignore`; их generators и compact summaries
остаются в репозитории.
