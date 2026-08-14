# MiniLM: ablation сериализации карточки товара

## Цель

Эксперимент изолирует только representation товара. Во всех запусках используется
один checkpoint `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, одинаковые данные,
split, train subset, normalization, optimizer и бюджет обучения. LLM-разметка не
читается и не подключается к notebook.

Проверяются четыре варианта:

| Код | Представление |
|---|---|
| `S0_TITLE` | Только нормализованное название |
| `S1_KEY_VALUE` | Название и все атрибуты в виде `key: value` |
| `S2_VALUES_ONLY` | Название и только значения атрибутов |
| `S3_HYBRID` | Частые train-derived ключи как `key: value`, для редких остаётся только value |

Новые special tokens (`[COL]`, `[VAL]` и подобные) не добавляются.

## Зафиксированный протокол

- human labels: `data/matches.parquet`, без `matches_llm.parquet`;
- split: `brand_model_family_holdout`, seed 42;
- компоненты графа пар и одинаковые консервативные brand/model/article family
  целиком находятся только в train или validation;
- screening subset: 120 000 пар из train pool, выбранных стабильным hash с seed
  `20260814`;
- validation: полная фиксированная grouped validation;
- одна эпоха, AdamW, LR `2e-5`, weight decay `0.01`, warmup 5%;
- single-GPU batch и effective batch: 64;
- `max_length=256`, longest-first pair truncation;
- случайная A/B ориентация в train и среднее A/B + B/A при validation;
- два независимых single-GPU запуска параллельно на двух T4, затем ещё два;
- primary metric: среднее из 20 category-wise
  `sklearn.metrics.average_precision_score`; дополнительно сохраняются overall AP,
  macro/overall ROC-AUC и AP каждой категории.

Все параметры находятся в `configs/serialization_ablation_minilm.json`.

## Единая normalization

Применяются Unicode NFKC, casefold и схлопывание whitespace. Простые единицы
приводятся к одной записи только рядом с числом: например, `ГБ/GB → gb`,
`мл/ML → ml`, `кг/KG → kg`. Числа, дефисы, слэши и пунктуация model/product
codes не удаляются: `RTX 4070`, `SM-S928B`, `WH-1000XM5` сохраняют состав
токенов и отличаются только регистром.

Атрибуты парсятся как JSON object либо список записей key/value. Пустые поля
пропускаются. Во всех трёх attribute-вариантах поля идут в одном порядке — по
train-derived глобальной частоте ключа, затем детерминированно по ключу и
значению.

## HYBRID threshold

Частота считается по уникальным товарам фиксированного train subset, а не по
validation и не по числу появлений товара в парах. Автоматическое правило
выбирает минимальный item support, при котором частые ключи покрывают 80%
всех train attribute occurrences; технический минимум support равен 100.
Threshold не подбирается по метрике.

Локальный полный prepare на текущих human-данных дал:

- train pool: 314 184 пары;
- train subset: 120 000 пар и 236 976 товаров;
- validation: 51 470 пар и 100 255 товаров;
- пересечение item ID: 0;
- пересечение непустых family signatures: 0;
- уникальных нормализованных attribute names в train subset: 24 916;
- выбранный item-support threshold: 694;
- frequent keys: 508;
- покрытие occurrences: 80.004%.

Notebook заново вычисляет эти значения из подключённой версии данных, сохраняет
полную таблицу `attribute_name_frequency.csv`, верхние строки показывает до
обучения, а итоговый отчёт содержит log-scale histogram.

## Запуск на Kaggle

Локальный dry-run:

```powershell
make serialization-ablation-dry-run
```

Отправка private kernel в фоне:

```powershell
make serialization-ablation-run
```

Позже дождаться завершения и скачать `/kaggle/working`:

```powershell
make serialization-ablation-monitor
```

Launcher создаёт или обновляет отдельный private code-only Dataset и подключает
к notebook human Dataset и credential Dataset для Google Sheets. Каждый из
четырёх вариантов записывается отдельной строкой в лист `experiments` и получает
20 строк category AP в `category_metrics`. Ошибка Sheets не скрывается: для
каждого варианта сохраняется отдельный pending JSON.

## Outputs и выбор победителя

В `/kaggle/working/serialization_ablation/` сохраняются:

- `serialization_comparison.csv` с колонками `serialization`, `PR_AUC`,
  `train_time`, `inference_speed`, `avg_tokens`, `p95_tokens` и дополнительными
  диагностическими метриками;
- `report.md`, `ablation_report.json` и `attribute_name_frequency.png`;
- training report, console log и validation predictions каждого варианта;
- checkpoint `S0_TITLE` и checkpoint варианта с максимальным macro AP.

Победитель выбирается только по primary macro AP. Если лучший вариант — сам
`S0_TITLE`, сохраняется один checkpoint, поскольку baseline и winner совпадают.

## Завершённый запуск

Private Kaggle kernel version 2 завершён 14 августа 2026 года со статусом
`COMPLETE`. Лучший вариант по primary macro AP — `S2_VALUES_ONLY` с результатом
`0.690153`. Сводка протокола, все метрики и решение по следующему запуску
зафиксированы в `docs/serialization-ablation-results.md`.

Автоматическая запись в Google Sheets во время notebook не прошла: credential
Dataset был подключён, но notebook не нашёл ожидаемый JSON-файл. Локальный
recovery `scripts/sync_serialization_ablation_sheets.py` затем записал все четыре
эксперимента и 80 category-metric строк без повторного обучения.
