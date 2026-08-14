# Full fine-tuning на LLM-корпусе на одной H100

[`scripts/train_llm_full.py`](../scripts/train_llm_full.py) обучает все параметры
MiniLM cross-encoder на всех `10 043 007` парах из `llm/non_ood_pairs.parquet`.
Ни одна строка не фильтруется: значения `0`, `1/9`, ..., `8/9`, `1` передаются
в supervised BCE как исходные soft targets. Поверх него включена
[Early-Learning Regularization](https://proceedings.neurips.cc/paper_files/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html),
которая не даёт модели за десять эпох просто запомнить шум слабой разметки.

По умолчанию OOD-файлы не используются. Это сохраняет категории «Одежда» и
«Бытовая техника» невиданными при обучении. Флаг `--include-ood` расширяет train
до всех `11 187 780` LLM-пар, но после этого frozen OOD-валидация больше не
является честной.

## Установка на сервере

Нужны Python 3.11–3.12, одна CUDA GPU и достаточно локального SSD для исходных
parquet, mmap-кэша и checkpoints. Для H100 используется BF16.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cross-encoder.txt
```

Проверка окружения:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

## Запуск

Из корня репозитория:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_llm_full.py \
  --data-dir prepared/validation_splits_v1/llm \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --human-validation-dir prepared/validation_splits_v1/human \
  --human-items data/items_human.parquet \
  --output-dir models/minilm_llm_full \
  --cache-dir artifacts/llm_full_cache \
  --epochs 10 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --learning-rate 5e-6 \
  --elr-beta 0.7 \
  --elr-lambda 3.0 \
  --max-length 512 \
  --serialization-variant S1_KEY_VALUE \
  --num-workers 8
```

`batch-size=256` — стартовая настройка для H100 80 GB. После проверки peak VRAM
её можно увеличить; при OOM достаточно уменьшить batch. Gradient accumulation
для одной эпохи по всем строкам не обязателен.

`max_length=512` выбран намеренно. У XLM-R checkpoint из команды это штатный
предел tokenizer (в конфигурации encoder — 514 позиций с учётом служебных
токенов). В имеющихся validation predictions при длине 384 около 21,6% примеров
уже упирались в cap, поэтому 384 для полного корпуса слишком агрессивен. Скрипт
проверяет предел tokenizer и не позволит случайно передать неподдерживаемую
длину.

`5e-6` — более консервативный peak LR для десяти проходов по 10,04 млн пар.
Scheduler делает warmup на первых 3% всех optimizer updates, затем cosine decay
до нуля. Это означает 100 430 070 train-примеров за полный запуск, поэтому перед
ним имеет смысл обязательно выполнить smoke-run.

## Early-Learning Regularization

Для каждого train pair хранится двухкомпонентная EMA прошлых предсказаний
`tᵢ`. Один logit cross-encoder преобразуется в
`pᵢ = [1-sigmoid(logit), sigmoid(logit)]`, после чего используется loss:

```text
BCE(logit, soft_target) + λ · mean(log(1 - <pᵢ, tᵢ>))
tᵢ ← β · tᵢ + (1-β) · stop_gradient(pᵢ)
```

Это binary-эквивалент формул (6) и (9) статьи. Defaults `β=0.7`, `λ=3.0` и
clamp `1e-4` совпадают с
[официальной реализацией](https://github.com/shengliu66/ELR). ELR-history имеет
форму `[10 043 007, 2]`, занимает около 77 MiB FP32 на GPU и сохраняется в
`elr_targets.pt` каждого checkpoint. Без этого файла точный resume невозможен.

## Human validation после каждой эпохи

После каждой эпохи checkpoint оценивается на frozen `iid`, `hard` и `ood` в
обоих порядках пары; итоговый score — среднее вероятностей A/B и B/A. Для всех
трёх наборов сохраняются overall AP, macro AP, AP по категориям, order gap и
predictions parquet.

Human-тексты сериализуются тем же `S1_KEY_VALUE` и с тем же частотным порядком
атрибутов, что и LLM train. Поэтому нужен сырой
`data/items_human.parquet` с колонками `id/name/attributes/category`; старый
`prepared/.../human/items.parquet` содержит только прежний `product_text` и для
этого запуска не используется.

`checkpoint-best` выбирается только по `IID macro AP`. Hard и OOD остаются
диагностическими метриками и не участвуют в выборе модели. Все эпохи всё равно
сохраняются, поэтому кривые трёх протоколов можно сравнить вручную.

## Сериализация товаров

По умолчанию используется точный вариант `S1_KEY_VALUE` из serialization
ablation. Текст имеет вид:

```text
нормализованный title. ключ: значение. ключ: значение
```

Применяются NFKC, casefold, замена `ё` на `е`, схлопывание пробелов и
нормализация единиц (`512 ГБ` → `512 gb`, `1 ТБ` → `1 tb` и т. п.). Вложенные
и списочные значения разворачиваются тем же способом, что в ablation. Категория
в текст не добавляется. Ключи сортируются по глобальной частоте на всех товарах,
на которые ссылаются выбранные train-пары: сначала `occurrences`, затем
`item_support`, затем имя ключа.

Полученный порядок сохраняется в cache как
`attribute_name_frequency.csv`. Если нужно буквально переиспользовать таблицу
из отдельного ablation-run, передайте:

```bash
python scripts/train_llm_full.py \
  --attribute-frequency-csv path/to/attribute_name_frequency.csv
```

Доступны также `S0_TITLE`, `S2_VALUES_ONLY` и `S3_HYBRID`. Для `S3_HYBRID`
нужно дополнительно передать исходный список:

```bash
python scripts/train_llm_full.py \
  --serialization-variant S3_HYBRID \
  --frequent-keys-json path/to/frequent_attribute_names.json
```

Пайплайн сначала:

1. находит все товары, реально используемые парами;
2. строит частотный ранг нормализованных атрибутов;
3. потоково сериализует и токенизирует каждый уникальный товар ровно один раз;
4. записывает токены и pair-index в mmap-кэш;
5. запускает 10 эпох full fine-tuning всех параметров модели в BF16 с ELR;
6. после каждой эпохи считает IID/hard/OOD validation;
7. сохраняет `checkpoint-last`, checkpoint каждой эпохи и лучший по IID.

Ожидаемые результаты:

```text
models/minilm_llm_full/
├── checkpoint-last/
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── elr_targets.pt
│   ├── rng_state.pt
│   └── training_state.json
├── checkpoint-best/
├── checkpoint-epoch-01/
├── ...
├── checkpoint-epoch-10/
├── validation/
│   ├── epoch-01/
│   │   ├── metrics.json
│   │   ├── iid_predictions.parquet
│   │   ├── hard_predictions.parquet
│   │   └── ood_predictions.parquet
│   └── ...
├── model/
│   ├── model.safetensors
│   └── tokenizer.json
├── training_args.json
├── validation_history.json
└── training_report.json
```

Epoch и best directories являются hard-link snapshots `checkpoint-last`: внутри
одной файловой системы неизменившиеся большие файлы физически не дублируются.
Если файловая система не поддерживает hard links, скрипт автоматически
переходит на обычное копирование. При копировании каждого каталога на другой
диск по отдельности экономия тоже пропадёт.

Кэш переиспользуется при повторном запуске с теми же файлами, tokenizer,
`max_length` и параметрами сериализации. Для отдельного построения кэша без GPU
training:

```bash
python scripts/train_llm_full.py --cache-only
```

Для быстрой сквозной проверки перед полным запуском:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_llm_full.py \
  --max-pairs 100000 \
  --output-dir models/minilm_llm_smoke \
  --cache-dir artifacts/llm_full_smoke_cache \
  --batch-size 256 \
  --save-every-updates 0
```

## Возобновление

Скрипт перезаписывает `checkpoint-last` каждые 5000 optimizer updates и после
каждой эпохи. Для продолжения нужны те же data/cache/training параметры:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_llm_full.py \
  --output-dir models/minilm_llm_full \
  --cache-dir artifacts/llm_full_cache \
  --batch-size 256 \
  --resume
```

Resume восстанавливает модель, optimizer, scheduler, ELR temporal targets, номер
эпохи и следующий batch. Порядок пар и ориентация `A/B` также детерминированы.

## Явное включение OOD

Только для финальной модели, когда OOD уже не используется для выбора рецепта:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_llm_full.py \
  --include-ood \
  --output-dir models/minilm_llm_all_categories \
  --cache-dir artifacts/llm_full_all_categories_cache
```

## Последующий human fine-tune

Итоговый checkpoint совместим с human fine-tune, но human train тоже нужно
подавать через тот же `S1_KEY_VALUE` serializer и частотный ранг из LLM cache.
Validation path в этом скрипте уже делает это корректно; старый generic trainer
с готовым `product_text` всё ещё использует прежний формат.
