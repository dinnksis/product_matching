# Обучение MiniLM cross-encoder

Основной компактный эксперимент использует
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`: multilingual
`XLMRobertaForSequenceClassification` с 12 слоями, hidden size 384 и одним
ranking logit. Модель обучается целиком через `BCEWithLogitsLoss`; PEFT, LoRA и
`torchao` в этом пути не используются.

## Конфигурация

Все изменяемые гиперпараметры находятся в
`configs/cross_encoder_minilm.json`. Training script также принимает каждый из
них как CLI-флаг; CLI имеет приоритет над JSON.

Основные параметры:

| Ключ | Назначение | Текущее значение |
| --- | --- | ---: |
| `epochs` | Число полных проходов по train | `1` |
| `batch_size` | Batch на одну GPU | `96` |
| `eval_batch_size` | Validation batch на одну GPU | `192` |
| `max_length` | Общая длина пары после tokenizer truncation | `256` |
| `learning_rate` | Peak learning rate | `2e-5` |
| `weight_decay` | AdamW weight decay | `0.01` |
| `warmup_ratio` | Доля optimizer steps на warmup | `0.05` |
| `sampling` | Балансировка sampler | `category_label` |
| `dataloader_workers` | Worker-процессы на одну GPU | `2` |
| `symmetric_validation` | Усреднять порядки A/B и B/A | `true` |
| `log_every` | Интервал training-логов в шагах | `20` |

При двух T4 effective batch равен
`batch_size × 2 × gradient_accumulation`. Если возникает CUDA OOM, сначала
уменьшите `batch_size` до `64`, затем `eval_batch_size`; gradient checkpointing
для этой небольшой модели обычно не нужен.

## Подготовка данных

Из исходных parquet строится component-disjoint split: один товар не может
одновременно попасть в train и validation. Каждая карточка сериализуется как
плоский список строк `название поля: значение`.

```bash
make prepare-human
```

Результат:

```text
prepared/human/
├── items.parquet
├── train_pairs.parquet
├── val_pairs.parquet
└── report.json
```

## Локальный запуск на двух GPU

После установки GPU-зависимостей:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_cross_encoder.py \
  --config configs/cross_encoder_minilm.json \
  --prepared-dir prepared/human \
  --output-dir models/cross_encoder_minilm \
  --token-cache-dir artifacts/token_cache/cross_encoder_minilm
```

Разовый CLI override не требует изменения JSON:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_cross_encoder.py \
  --config configs/cross_encoder_minilm.json \
  --batch-size 64 \
  --learning-rate 1e-5
```

Token cache содержит обе ориентации пары и переиспользуется, пока не изменились
данные, checkpoint или `max_length`. Validation запускается только после всех
эпох.

## Первый запуск на Kaggle или изменение кода

Для Kaggle нужны заполненные `KAGGLE_USERNAME` и `KAGGLE_API_TOKEN` в `.env`.
Когда меняются training scripts, `src/` или requirements, сначала загрузите
новую версию приватного code bundle:

```bash
make kaggle-cross-build
make kaggle-train-data
make kaggle-cross-dry-run
make kaggle-cross-run
```

`kaggle-cross-run` по умолчанию отправляет приватный kernel
`product-matching-minilm-training` на 2×T4 и сразу возвращает управление.
Запуск с ожиданием и автоматическим скачиванием outputs:

```bash
uv run python scripts/run_cross_encoder_kaggle.py --wait
```

## Изменение только гиперпараметров

Если исходный код не менялся, повторно загружать 208 MiB Dataset не требуется:

1. отредактируйте `configs/cross_encoder_minilm.json`;
2. выполните `make kaggle-cross-dry-run`;
3. выполните `make kaggle-cross-run`.

Генератор встраивает актуальный JSON в отдельную ячейку notebook. Эту же ячейку
можно изменить непосредственно в Kaggle UI перед ручным запуском.

## Логи и результаты

Во время tokenization выводятся число обработанных строк и throughput. Каждые
`log_every` шагов обучения выводятся:

- loss и learning rate;
- examples/s и seconds/step;
- ETA текущей эпохи;
- доля ожидания DataLoader и padding efficiency;
- peak VRAM текущей GPU.

Kaggle сохраняет:

```text
/kaggle/working/
├── minilm_cross_encoder/       # веса, tokenizer, config и training_report.json
├── minilm_training.log         # полный stdout/stderr DDP-процесса
└── notebook_completed.json
```

`training_report.json` содержит время обучения и validation, throughput, peak
VRAM обеих GPU, overall AP, macro AP и AP каждой категории.
