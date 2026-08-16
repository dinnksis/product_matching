# Эксперименты на frozen IID, hard и OOD

Единый приватный Kaggle Dataset:
`alexproger23/product-matching-validation-splits-v1`.

Он фиксирует один train и три независимых протокола проверки:

- `human_train_pairs.parquet` — единственный human train, 306 669 пар;
- `human_iid_validation_pairs.parquet` — IID, 12 000 пар;
- `human_hard_validation_pairs.parquet` — hard, 5 814 пар;
- `human_ood_validation_pairs.parquet` — OOD, 41 171 пара только из категорий
  «Одежда» и «Бытовая техника»;
- `human_items.parquet` — тексты и категории для всех human item ID;
- `llm_non_ood_items.parquet` и `llm_non_ood_pairs.parquet` — допустимый источник
  дополнительного train;
- `llm_ood_items.parquet` и `llm_ood_pairs.parquet` — отложенная OOD-часть, её
  нельзя использовать для обучения или подбора гиперпараметров.

Human train, IID, hard и OOD не пересекаются по item ID. Все эксперименты должны
использовать ровно эти файлы: нельзя пересэмплировать validation или переносить
пары между split-ами. Для выбора модели основной показатель — macro AP на IID;
hard показывает устойчивость на специально сложных примерах, OOD — перенос на
две полностью исключённые из train категории.

Оба LLM item-каталога дополнительно очищены от всех item ID, встречающихся в
human IID, hard или OOD. Сборка Dataset завершается ошибкой, если хотя бы одна
LLM-пара касается frozen validation item. Таким образом, для `llm_non_ood_*`
нулевое пересечение гарантируется одновременно на уровне item-строк и пар.

## Готовый MiniLM baseline

Baseline обучает `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` одну эпоху только
на `human_train_pairs.parquet`, без LLM-разметки. После обучения один checkpoint
считается на всех трёх validation-наборах в обоих порядках пары.

Локально проверить будущую отправку, не обращаясь к Kaggle:

```bash
make kaggle-minilm-validation-dry-run
```

Создать notebook без запуска:

```bash
make kaggle-minilm-validation-build
```

Отправить приватный notebook, дождаться окончания и скачать outputs:

```bash
make kaggle-minilm-validation-run
```

Чтобы только отправить job и не ждать локально:

```bash
uv run python scripts/run_minilm_validation_baseline_kaggle.py --no-wait
```

Runner перед выделением GPU проверяет статус Dataset, полный набор файлов и
SHA-256 frozen manifest. Notebook сохраняет:

- `training_report.json` с секцией `validation_splits.{iid,hard,ood}`;
- `iid_validation_predictions.parquet`;
- `hard_validation_predictions.parquet`;
- `ood_validation_predictions.parquet`;
- checkpoint, training config, log и `notebook_completed.json`.

Для другого MiniLM-конфига скопируйте
`configs/cross_encoder_minilm_validation_baseline.json`, измените параметры и
передайте его runner-у:

```bash
uv run python scripts/run_minilm_validation_baseline_kaggle.py \
  --config configs/my_minilm_experiment.json
```

Для другой архитектуры контракт тот же: train берётся только из human train и,
если нужно, `llm_non_ood_*`; финальный report обязан содержать метрики `iid`,
`hard` и `ood`.

## Human fine-tune после LLM pretraining

Эксперимент `minilm_llm_pretrain_human_ft_v1` начинает обучение с локального
checkpoint `model/pretrain_minilm_1ep`, уже прошедшего одну эпоху на разрешённой
non-OOD LLM-разметке. Следующий stage использует только
`human_train_pairs.parquet`, создаёт optimizer и scheduler заново и валидирует
полученную модель на тех же frozen IID, hard и OOD.

Все параметры human fine-tune, кроме начального checkpoint, программно
сверяются с `configs/cross_encoder_minilm_validation_baseline.json`. Текущий
конфиг — `configs/cross_encoder_minilm_llm_pretrain_human_ft.json`; запуск
прервётся до отправки на Kaggle при любом случайном расхождении.

Для Kaggle в отдельный приватный Dataset загружаются только веса, tokenizer,
model config и небольшие provenance-файлы. `optimizer.pt`, `scheduler.pt`,
`rng_state.pt` и `elr_targets.pt` не загружаются, поскольку это новый fine-tune
stage, а не продолжение LLM-optimizer state:

```bash
make kaggle-minilm-pretrain-checkpoint-dry-run
make kaggle-minilm-pretrain-checkpoint
make kaggle-minilm-pretrain-human-ft-dry-run
make kaggle-minilm-pretrain-human-ft-run
```

Последняя команда ждёт завершения, скачивает outputs и добавляет одну компактную
строку с IID/hard/OOD метриками в `experiments_v2`.

Первый запуск завершён 2026-08-16:

| Validation | Human-only baseline | LLM pretrain 1ep → human FT | Δ к baseline |
| --- | ---: | ---: | ---: |
| IID | 0.705238 | 0.770041 | +0.064803 |
| hard | 0.268804 | 0.341307 | +0.072504 |
| OOD | 0.579545 | 0.630826 | +0.051280 |

Эксперимент использовал 306 669 human train-пар и checkpoint Dataset
`alexproger23/product-matching-minilm-llm-pretrain-1ep` v1 с manifest SHA-256
`5d8a6ac3b3e78b4cb61ddec4fd9453e6b6c49894f493e43bfd18a04fc40498ca`.
Kaggle kernel:
`alexproger23/product-matching-minilm-llm-pretrain-human-ft-v1`.

Тот же эксперимент после пяти эпох LLM pretraining запускается так:

```bash
uv run python scripts/run_minilm_llm_pretrain_human_ft_kaggle.py \
  --checkpoint-tag 5ep
```

Он завершён 2026-08-16 с неизменными human fine-tune параметрами и теми же
frozen validation splits:

| Validation | Human-only baseline | 1ep → human FT | 5ep → human FT | Δ 5ep к 1ep |
| --- | ---: | ---: | ---: | ---: |
| IID | 0.705238 | 0.770041 | 0.789388 | +0.019347 |
| hard | 0.268804 | 0.341307 | 0.365501 | +0.024194 |
| OOD | 0.579545 | 0.630826 | 0.642660 | +0.011834 |

Checkpoint Dataset —
`alexproger23/product-matching-minilm-llm-pretrain-5ep` v1 с manifest SHA-256
`354c7006898a9a44a3115c8384f12dbab520cfec7723a675f8ccedb108544533`.
Kaggle kernel:
`alexproger23/product-matching-minilm-5ep-human-ft-v1`.

## Google Sheets

Новые запуски записываются в компактный лист `experiments_v2`: одна строка на
`run_id`. В нём остаются ссылки и идентификаторы воспроизводимости, macro AP и
overall AP для IID/hard/OOD, число train-пар и основные гиперпараметры.

Время validation, throughput, VRAM, padding efficiency, per-category строки и
полный JSON больше не отправляются в таблицу. Эти подробности остаются в
`training_report.json` среди Kaggle outputs. Старые листы `experiments` и
`category_metrics` не удаляются и сохраняют историю прежних запусков.
