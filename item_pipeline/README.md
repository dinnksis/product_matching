# Standalone item generation pipeline

Пайплайн создаёт первоначальные самостоятельные товары. Он не создаёт пары,
match/non-match labels или скрытые entity IDs. Следующий независимый этап может
использовать сгенерированные items как anchors.

## Контракт

Основной результат `items.parquet` имеет ровно конкурсную схему:

```text
id: int64
name: string
attributes: JSON string object<string, string>
category: string
```

`generation_metadata.parquet` — служебный sidecar с provenance: schema donor,
retrieved examples, модель, prompt hash, попытки и validation diagnostics. В нём
нет match-связей.

## Логика

1. Из train-safe `llm/non_ood_items.parquet` потоково выбирается
   детерминированный exemplar bank по категориям.
2. `тип` используется как слабый подтип; при его отсутствии берётся короткий
   title fallback.
3. Для schema donor находятся похожие карточки: сначала та же категория и
   подтип, затем BM25 + multilingual MiniLM dense similarity.
4. Qwen получает точный набор ключей donor и несколько стилевых примеров. Он
   обязан создать новую товарную сущность, новые согласованные значения и новое
   название, не добавляя ключей.
5. Ответ проходит JSON-schema и локальные проверки: схема, непустые строковые
   значения, изменение identity-полей, отсутствие exact-name copy и чрезмерной
   близости к примерам.
6. Checkpoint можно безопасно продолжать той же командой.

Human и LLM items физически не объединяются: все human IDs уже присутствуют в
большом каталоге, а prepared non-OOD слой исключает human validation items.

## Установка dense retrieval

Основное окружение уже содержит pandas, PyArrow, requests, scikit-learn и
RapidFuzz. Для MiniLM дополнительно:

```bash
uv pip install --python .venv/bin/python -r item_pipeline/requirements.txt
```

## Запуск

Сначала создаётся ограниченный индекс. По умолчанию хранится до 10 000 карточек
на категорию, а не embeddings всех 12 млн items.

```bash
.venv/bin/python -m item_pipeline prepare
```

После первого скачивания модели повторную сборку без Hub-запросов можно запустить
с `--embedding-local-files-only`.

Диагностический индекс без скачивания MiniLM:

```bash
.venv/bin/python -m item_pipeline prepare \
  --limit-rows 100000 \
  --max-items-per-category 500 \
  --skip-embeddings \
  --output-dir /tmp/item-pipeline-index
```

Генерация через OpenAI-compatible Qwen endpoint:

```bash
.venv/bin/python -m item_pipeline generate \
  --count 1000 \
  --workers 15
```

Можно ограничить одну или несколько категорий повторяющимся `--category`.

Проверка итоговых артефактов:

```bash
.venv/bin/python -m item_pipeline validate
```

## Артефакты

```text
item_pipeline/artifacts/index/
  exemplar_bank.parquet
  embeddings.f16.npy
  embedding_ids.npy
  profile.json

item_pipeline/artifacts/generated/
  items.parquet
  generation_metadata.parquet
  errors.json
  summary.json
  validation_report.json
```

Synthetic IDs по умолчанию отрицательные, поэтому не пересекаются с исходным
каталогом. Настоящая полезность генерации проверяется только последующим
обучением matcher и метрикой на неизменённых human IID/hard splits.
