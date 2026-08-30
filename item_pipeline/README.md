# Standalone item generation pipeline

Пайплайн умеет создавать самостоятельные товары. Команда `generate-pairs`
использует их только как примеры стиля категории: содержательно каждая пара
строится с нуля от выбранного правила. Скрытые entity IDs не используются.

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

Если карточка не проходит все локальные проверки после внутренних вариантов,
задание автоматически возвращается в очередь с новым диапазоном seed. Число
таких повторных кругов задаётся `--task-retries` (по умолчанию 3); готовые
карточки при этом берутся из checkpoint и не генерируются повторно. Для ещё
одного прохода по полностью отклонённому хвосту можно передать новый
`--task-seed-offset`, не меняя совместимость checkpoint.

Можно ограничить одну или несколько категорий повторяющимся `--category`.

Проверка итоговых артефактов:

```bash
.venv/bin/python -m item_pipeline validate
```

## Генерация пар по каталогу правил

По умолчанию совместно читаются три каталога из
`configs/generation_rule_catalog_rare_v1/`: все 75 кандидатов, исполняемый JSON
с 24 готовыми правилами и CSV с 39 экспериментальными. Повторяющиеся записи
объединяются по `generation_rule_id`: JSON уточняет инструкции готовых правил,
но итоговый каталог всё равно содержит 75 записей. Три правила без
`allowed_categories` никогда не выбираются. В metadata сохраняется уровень
каждого правила: 24 `RARE_SAFE`, 39 экспериментальных и 12 aliases основного
каталога. Для автоматической генерации только безопасных правил передайте
`--tier RARE_SAFE`.

Для полного набора сначала создаются 10 000 новых исходных карточек. Флаг
`--plain-json` не отправляет `response_format/json_schema`: точный JSON-формат
задаётся только в prompt, после чего ответ проходит тот же строгий локальный
валидатор.

```bash
.venv/bin/python -m item_pipeline generate \
  --count 10000 \
  --workers 50 \
  --base-url http://127.0.0.1:8994/v1 \
  --model qwen3.5-397b-a17b-fp8 \
  --plain-json
```

Затем на их стиле создаются 10 000 размеченных пар. Переданный `--items` не
служит готовой левой стороной: для каждой пары сначала выбираются правила и
генерируется новая применимая исходная карточка, затем отдельным запросом — её
контролируемая мутация:

```bash
.venv/bin/python -m item_pipeline generate-pairs \
  --items item_pipeline/artifacts/generated/items.parquet \
  --count 10000 \
  --workers 50 \
  --base-url http://127.0.0.1:8994/v1 \
  --model qwen3.5-397b-a17b-fp8 \
  --two-rule-fraction 0.5 \
  --plain-json
```

Порядок этапов и локальные гарантии:

1. Выбирается одно правило или совместимая пара правил. Для двух правил общей
   категории недостаточно: concepts должны входить в явную консервативную группу
   характеристик одного типа товара (например, параметры контактных линз).
2. У каждого canonical concept есть фиксированный русский ключ целевого
   атрибута. Qwen создаёт исходный товар с `Тип товара`, этими точными ключами и
   подтверждением `rule → attribute_key → attribute_value`.
3. Валидатор требует, чтобы каждый ключ существовал, подтверждённое значение
   полностью совпадало со значением атрибута и дословно присутствовало в
   названии исходного товара.
4. Второй запрос получает уже проверенные ключи. Он может изменить только по
   одному целевому ключу на правило; все остальные attributes сохраняются.
5. Старое значение обязано исчезнуть из нового названия, а новое — дословно
   появиться. Итоговая проверка повторяет эти условия по сохранённым metadata.
6. Для конечных областей значений (цвет золота, память смартфона и размер
   смычка) разрешены только канонические значения каталога. Написания вроде
   `128`, `128 ГБ` и `128GB` дают одну семантическую подпись; физически равные
   `1 м` и `100 см` также не считаются изменением. Количество всегда содержит
   `шт`, а числовой физический размер — явную единицу.
7. До запросов строится детерминированное расписание. Оно соблюдает ёмкость
   конечных профилей, сохраняет один bundle на всех retry и балансирует общее
   число применений правил. Если запрошенная доля 1+1 перегружает только
   совместимые профили, она автоматически снижается примерно до 8%.

Команду можно повторять после обрыва: готовые пары загружаются из checkpoint.
Неуспешные задания автоматически переносятся в конец очереди и получают новый
model seed, но сохраняют тот же заранее назначенный набор правил; число
дополнительных кругов задаётся `--task-retries`. Отдельная
проверка артефактов:

```bash
.venv/bin/python -m item_pipeline validate-pairs
```

`RARE_EXPERIMENTAL_*` сохраняют целевую метку `0`, но их уровень обязательно
остаётся в metadata: статистическая проверка этих правил слабее, чем у
`RARE_SAFE`, поэтому результаты нельзя смешивать без учёта provenance.

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

item_pipeline/artifacts/rule_first_pairs/
  base_items.parquet
  items.parquet
  mutated_items.parquet
  pairs.parquet
  pair_generation_metadata.parquet
  errors.json
  summary.json
  validation_report.json
```

Synthetic IDs по умолчанию отрицательные, поэтому не пересекаются с исходным
каталогом. Настоящая полезность генерации проверяется только последующим
обучением matcher и метрикой на неизменённых human IID/hard splits.

Полный длительный прогон с автоматическим добором хвоста и переходом от items к
парам можно запустить одной командой:

```bash
.venv/bin/python scripts/run_full_item_rule_generation.py \
  --workers 50 \
  --base-url http://127.0.0.1:8994/v1 \
  --plain-json
```
