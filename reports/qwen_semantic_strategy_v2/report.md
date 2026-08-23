# Быстрая стратегия semantic extraction v2

## Почему прекращаем усложнять pair-level prompt

На одинаковых первых 50 парах:

| Метрика | v1.3, workers=2 | v1.4, workers=10 |
|---|---:|---:|
| Strict `ok` | 24/50 | 11/50 |
| Средняя latency запроса | 11,13 с | 18,43 с |
| Фактическая скорость | ~11,2 пар/мин | ~6,7 пар/мин |
| Completion tokens | 48 233 | 80 848 |
| Candidate facts | 261 | 338 |
| Обычные agreements, ошибочно выданные как anchors | 28 | 81 |

V1.4 стала длиннее и заставила Qwen генерировать больше лишних facts. Десять
одновременных запросов не ускорили 397B model, а создали contention. Дальнейшее
увеличение pair prompt или workers не решает задачу.

## Новая архитектура

```text
raw attribute names + несколько значений
              ↓ один раз, батчами
category-aware ontology mapping от Qwen
              ↓ кэш
детерминированное сравнение attributes всех пар
              ↓
Qwen fallback только для title-only/conflict/unresolved
              ↓
rule statistics после локального присоединения human label
```

Qwen больше не получает одну полную пару за один запрос. Она один раз определяет
смысл raw attribute name. Результат переиспользуется для сотен тысяч товаров и
пар.

## Реальный объём

Только внутри `RULE_DISCOVERY` находится 277 349 пар и 539 247 товаров. Для
95% покрытия attribute occurrences по каждой из 18 категорий достаточно 7 108
`(category, raw_attribute_name)` вместо всех редких имён длинного хвоста.

При 50 names на request подготовлен 151 batch. Это в 1 837 раз меньше запросов,
чем pair-by-pair обработка 277 349 пар. В Qwen input нет пар и human labels.

После получения mapping код будет:

- объединять aliases через canonical concept;
- нормализовывать placeholders и безопасные единицы;
- сравнивать значения;
- формировать same identity anchors, differences и relevant missing;
- сохранять raw names/evidence;
- присоединять human label только для подсчёта support/precision rules.

Ручная проверка всех пар не требуется. Проверяться будут только low-confidence
mappings, новые конфликтующие mappings и representative examples top rules.

## Подготовленные файлы

- `scripts/create_qwen_attribute_ontology_jobs.py` — воспроизводимая подготовка;
- `data/qwen_attribute_ontology_v1/ontology_entries.parquet` — 7 108 entries;
- `data/qwen_attribute_ontology_v1/ontology_batches.jsonl` — 151 batch;
- `prompts/qwen_attribute_ontology_v1.md` — компактный mapping prompt;
- `schemas/qwen_attribute_ontology_v1.schema.json` — строгий batch schema;
- `scripts/run_qwen_attribute_ontology.py` — resumable runner;
- `data/qwen_attribute_ontology_v1/manifest.json` — provenance и hashes.

## Порядок запуска

Сначала выполняются 10 batches (500 attribute names). После автоматического
анализа mapping та же команда без `--max-batches 10` продолжит checkpoint и
обработает остальные batches. Рекомендуются 4 workers: 10 workers уже показали
ухудшение throughput на 397B server.
