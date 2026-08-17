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

## Google Sheets

Новые запуски записываются в компактный лист `experiments_v2`: одна строка на
`run_id`. В нём остаются ссылки и идентификаторы воспроизводимости, macro AP и
overall AP для IID/hard/OOD, число train-пар и основные гиперпараметры.

Время validation, throughput, VRAM, padding efficiency, per-category строки и
полный JSON больше не отправляются в таблицу. Эти подробности остаются в
`training_report.json` среди Kaggle outputs. Старые листы `experiments` и
`category_metrics` не удаляются и сохраняют историю прежних запусков.
