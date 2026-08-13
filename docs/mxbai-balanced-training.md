# mxbai xsmall: balanced human + LLM experiment

Эксперимент использует `mixedbread-ai/mxbai-rerank-xsmall-v1` —
`DebertaV2ForSequenceClassification` с 12 слоями, hidden size 384, одним
ranking logit и лимитом архитектуры 512 токенов. Обучение выполняется напрямую
через Transformers и `BCEWithLogitsLoss`.

Checkpoint опубликован как English reranker, тогда как данные карточек в
основном русскоязычные. Поэтому эксперимент проверяет перенос DeBERTa reranker
на русский product matching; превосходство над multilingual MiniLM заранее не
предполагается. Model card и конфигурация:
`https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1`.

## Как формируется один item

Исходная карточка содержит `category`, `name` и JSON-объект `attributes`.
`serialize_product` превращает её в плоский текст, одна строка на поле:

```text
Категория: Бытовая техника
Название: микроволновая печь LG MS-20R42D
бренд: lg
модель: ms20r42d
тип: микроволновая печь
объем, л: 20
цвет товара: белый
...
```

Порядок фиксирован:

1. категория;
2. название;
3. приоритетные атрибуты: бренд, модель, артикул/партномер/SKU, тип/вид,
   размер, объём, вес, цвет, материал, комплектация;
4. остальные атрибуты в алфавитном порядке.

Пустые поля удаляются, пробелы нормализуются, но числа, пунктуация и артикулы
не повреждаются. Атрибуты одного товара ограничены 6000 символами по границе
строки. После этого tokenizer формирует один pair input:

```text
[CLS] item A [SEP] item B [SEP]
```

и применяет `longest_first` truncation до общих 384 токенов. Это 384 токена на
пару, а не на каждый товар.

## Физический баланс данных

Подготовка запускается командой:

```bash
uv run python scripts/prepare_balanced_llm_data.py
```

Алгоритм:

1. human-данные делятся component-disjoint с seed 42;
2. все 310 767 human train-пар сохраняются;
3. максимальная human-группа `категория × класс` содержит 17 490 строк;
4. каждая из 40 групп дополняется LLM-парами ровно до 17 490;
5. сначала выбираются raw LLM targets 0 и 1, затем ближайшие к нужному краю;
6. пары с validation item ID, дубли и пары, уже размеченные human, исключаются;
7. итог случайно перемешивается с фиксированным seed.

Результат:

```text
prepared/mxbai_balanced/
├── items.parquet             # 1 408 129 только нужных карточек
├── train_pairs.parquet       # 699 600 пар
├── val_pairs.parquet         # 54 887, только human
└── report.json
```

В каждой из 20 категорий находится ровно 34 980 train-пар: 17 490 negative и
17 490 positive. Добавлено 388 833 LLM-пары. Из них 118 795 имеют raw target 0,
267 501 — raw target 1, и только 2 537 — raw target 8/9.

Human и LLM различаются через `sample_weight`. До нормализации human имеет вес
1.0, уверенный LLM — 0.35, LLM с target 8/9 — 0.2722. Затем масса loss
выравнивается отдельно для каждой группы `категория × класс`; внутри группы
human остаётся примерно в 2.86 раза важнее уверенного LLM. Поэтому категории и
классы сбалансированы не только по строкам, но и по влиянию на loss.

## Training config

Конфигурация находится в
`configs/cross_encoder_mxbai_xsmall_balanced_llm.json`:

```text
epochs: 1
batch_size: 32 на GPU
gradient_accumulation: 2
eval_batch_size: 64 на GPU
effective batch: 128 на 2×T4
max_length: 384
learning_rate: 1e-5
warmup_ratio: 0.05
sampling: none
label_smoothing: 0.02
attention_implementation: eager
```

Это full fine-tuning всех 70.8 млн параметров. В train случайно выбирается
ориентация пары, а финальная validation считает A/B и B/A и усредняет scores.
Validation запускается только после эпохи и сохраняет
`validation_predictions.parquet`.

## Kaggle

Локальная проверка без внешней загрузки:

```bash
make kaggle-mxbai-build
make kaggle-mxbai-data-dry-run
make kaggle-mxbai-dry-run
```

После явного решения запускать эксперимент:

```bash
uv run python scripts/push_mxbai_training_dataset.py
make kaggle-mxbai-run
```

Команда запуска перед отправкой GPU kernel проверяет, что Dataset имеет статус
`ready` и содержит все четыре prepared-файла и manifest. После отправки она
скачивает metadata опубликованной версии и проверяет, что Kaggle действительно
сохранил `dataset_sources`. Это защищает от запуска с пустым `/kaggle/input`.
При обновлении Dataset скрипт ждёт увеличения `current_version_number`, а не
только общего статуса `ready`: во время обработки новой версии API ещё может
возвращать `ready` для предыдущей. Перед запуском также скачивается удалённый
manifest и его bundle SHA сверяется с notebook. Та же проверка повторяется в
первой ячейке на уже смонтированной версии Dataset.

После успешной validation финальная ячейка добавляет итоговую строку и AP по
категориям в общую Google-таблицу экспериментов. Runner автоматически подключает
private credential Dataset; secret `GOOGLE_SERVICE_ACCOUNT_JSON` можно оставить
как приоритетный необязательный источник. Подробности находятся в
[`docs/kaggle-notebook.md`](kaggle-notebook.md). Ошибка Google Sheets не отменяет
успешно завершённое обучение и сохраняется отдельным pending-отчётом.

При первом запуске построение token cache для 699 600 пар занимает примерно
14–16 минут. Его выполняет только rank 0, а rank 1 в это время ожидает через
CPU/Gloo control-group. Это ожидаемая подготовка, а не зависание GPU. NCCL
используется после неё только для синхронного обучения; timeout подготовительного
этапа установлен в один час.

Используется отдельный приватный Dataset
`product-matching-mxbai-balanced-training` и отдельный kernel
`product-matching-mxbai-xsmall-balanced-training`; существующие MiniLM и Qwen
артефакты не перезаписываются.
