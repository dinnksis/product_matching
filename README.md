# Product matching

Репозиторий устроен так, чтобы ветка всегда могла быть упакована и отправлена в
контейнерное соревнование. Текущий scorer — быстрый CPU baseline без обучения;
он нужен как проверка полного submission-пайплайна. Обученные reranker/CatBoost
модели подключаются позже через `src/scorer.py`, не меняя CLI.

## Запуск

```bash
python run.py \
  --items_path data/items_human.parquet \
  --matches_path data/matches.parquet \
  --output-path submit.csv
```

Поддерживаемые аргументы:

- `--items_path`, `-i` — parquet с `id`, `name`, `attributes`, `category`;
- `--matches_path`, `-m` — parquet с `id1`, `id2` (лишние столбцы игнорируются);
- `--output-path`, `--output_path`, `-o` — итоговый CSV.

Результат всегда содержит ровно `id1,id2,predict`, сохраняет порядок и число
входных пар. Ошибки схемы, пропущенные товары и нечисловые предсказания приводят
к явной ошибке до записи результата.

## Проверка и упаковка

```bash
python -m unittest discover -s tests -v
python scripts/smoke_test.py
docker build -t product-matching:local .
powershell -ExecutionPolicy Bypass -File scripts/package_submission.ps1
```

Последняя команда создаёт `dist/submission.zip`. Реальные данные, ноутбуки,
кэши и локальные результаты в архив не попадают. Перед отправкой распакуйте
архив в пустую папку и проверьте, что `metadata.json` лежит в его корне.

## Структура

```text
run.py                    официальный entry point
src/scorer.py             заменяемая модель и подготовка текста
src/io.py                 чтение, join и строгая проверка результата
model/config.json         версионируемые параметры текущего scorer
metadata.json             образ и команда запуска платформы
Dockerfile                воспроизводимое окружение
scripts/                  smoke-test и сборка архива
tests/                    быстрые контрактные тесты
notebooks/                EDA/обучение, не входит в submission
data/                     локальные данные, не входят в git/submission
```

## Следующий модельный этап

Для reranker стоит вынести построение пары текстов в отдельную функцию и
зафиксировать шаблон (`name`, затем нормализованные `attributes`, категория —
отдельным полем). Не превращайте JSON attributes в строку с зависимым от порядка
ключей представлением: сортируйте ключи и используйте явные разделители. Для
CatBoost полезнее сохранить отдельные численные признаки (совпадение бренда,
модели, размеров, чисел и единиц), а embedding similarity добавить как один из
признаков.

## Human data pipeline

Подготовка сериализует JSON attributes, ставит важные ключи раньше остальных,
ограничивает аномально длинные записи и создаёт item-disjoint split:

```bash
python scripts/prepare_human_data.py
```

Результат находится в `prepared/human/`:

- `items.parquet` — `id`, исходное имя, категория и `product_text`;
- `train_pairs.parquet` — обучающие пары с target;
- `val_pairs.parquet` — отдельная валидация с target;
- `report.json` — размеры, баланс категорий и время подготовки.

Товары из validation не встречаются в train. Подготовленные файлы локальные и
не входят в git или submission.

## Первый Qwen experiment: names only

GPU-зависимости:

```bash
pip install -r requirements-gpu.txt
```

Сначала измеряется pretrained-модель без обучения:

```bash
python scripts/benchmark_qwen_names.py --limit 10000
python scripts/benchmark_qwen_names.py
```

Скрипт считает macro Average Precision по категориям и отдельно печатает время
чтения данных, загрузки модели, инференса, полное время, пар/секунду и оценку
времени на миллион пар. Для лимита контейнера важен `total_seconds`. `--limit` предназначен только для
быстрого замера throughput. По умолчанию используется PyTorch SDPA. Если в
окружении установлен `flash-attn`, можно передать
`--attention-implementation flash_attention_2`.

LoRA-обучение только на названиях:

```bash
python scripts/train_qwen_names.py \
  --epochs 1 \
  --batch-size 16 \
  --gradient-accumulation 4
```

На машине с двумя GPU тот же эксперимент запускается одним процессом `torchrun`,
а не двумя независимыми ячейками:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_qwen_names.py \
  --epochs 1 --batch-size 16 --gradient-accumulation 4
```

Полное дообучение выделено в отдельный файл и сохраняется в
`model/qwen_names_full/`:

```bash
torchrun --standalone --nproc_per_node=2 scripts/train_qwen_names_full.py \
  --epochs 1 --batch-size 8 --gradient-accumulation 8
```

Проверка сохранённого адаптера:

```bash
python scripts/benchmark_qwen_names.py \
  --adapter model/qwen_names_lora
```

Размер batch следует увеличивать до заполнения GPU. `max-length=256` достаточен
для эксперимента только по названиям; для `product_text` лимит нужно выбирать
отдельно после анализа токенов Qwen.
