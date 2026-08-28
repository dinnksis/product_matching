# Сравнение четырёх cross-encoder архитектур на 2×T4

Эксперимент сравнивает четыре модели на одном frozen-протоколе без ансамбля и
без поиска гиперпараметров:

| Профиль | Исходные веса | Human fine-tune |
| --- | --- | --- |
| `gte` | `Alibaba-NLP/gte-multilingual-reranker-base` | 1 эпоха |
| `rumodernbert` | `deepvk/RuModernBERT-base` | 1 эпоха |
| `bge-v2-m3` | `BAAI/bge-reranker-v2-m3` | 1 эпоха |
| `minilm-5ep` | frozen MiniLM checkpoint после 5 эпох synthetic pretraining | 1 эпоха |

Для всех профилей фиксированы human train, IID, hard и OOD splits,
`S2_VALUES_ONLY`, `max_length=384`, learning rate `2e-5`, одна эпоха и effective
batch `192`. Per-device batch и gradient accumulation различаются только для
того, чтобы модели помещались в память. T4 использует FP16: BF16 на этой
архитектуре GPU не поддерживается. MiniLM меняет сериализацию между этапами:
старый synthetic checkpoint был обучен с `S1_KEY_VALUE`, а общий human
fine-tune этого эксперимента выполняется с требуемой `S2_VALUES_ONLY`.

## Сборка и локальная проверка

```powershell
.venv\Scripts\python.exe scripts\create_architecture_baseline_notebooks.py
.venv\Scripts\python.exe scripts\run_architecture_baseline_kaggle.py all --dry-run
```

Готовые notebooks находятся в `notebooks/architecture_baselines/`. Генератор
встраивает trainer, сериализацию и frozen frequency table в notebook, поэтому
отдельный source Dataset создавать не требуется.

## Последовательный запуск на Kaggle

Каждая команда пересобирает notebooks, отправляет приватный kernel, ждёт
завершения, скачивает `/kaggle/working` и проверяет обязательные артефакты.
Запускать следующую команду нужно после успешного завершения предыдущей:

```powershell
.venv\Scripts\python.exe scripts\run_architecture_baseline_kaggle.py gte
.venv\Scripts\python.exe scripts\run_architecture_baseline_kaggle.py rumodernbert
.venv\Scripts\python.exe scripts\run_architecture_baseline_kaggle.py bge-v2-m3
.venv\Scripts\python.exe scripts\run_architecture_baseline_kaggle.py minilm-5ep
```

Для фоновой отправки одного профиля добавьте `--no-wait`. После завершения его
можно скачать без повторного запуска:

```powershell
.venv\Scripts\python.exe scripts\run_architecture_baseline_kaggle.py gte --download-existing
```

Чтобы сначала скачать только predictions, отчёты и logs без тяжёлого checkpoint:

```powershell
.venv\Scripts\python.exe scripts\run_architecture_baseline_kaggle.py gte --download-existing --essential-only
```

Флаг `--essential-only` также можно добавить к обычной команде запуска: после
завершения обучения runner пропустит полный output и скачает только лёгкие
экспериментальные артефакты. Полный checkpoint можно забрать позднее обычным
`--download-existing`.

## Куда складываются результаты

Runner автоматически раскладывает полные Kaggle outputs по отдельным каталогам:

- `artifacts/kaggle/product-matching-architecture-gte-v1/`;
- `artifacts/kaggle/product-matching-architecture-rumodernbert-v1/`;
- `artifacts/kaggle/product-matching-architecture-bge-v2-m3-v1/`;
- `artifacts/kaggle/product-matching-architecture-minilm-5ep-v1/`.

Если outputs скачиваются вручную с сайта, содержимое Kaggle output ZIP нужно
распаковать прямо в соответствующий каталог выше, сохранив вложенную структуру.
Для каждого запуска должны присутствовать:

- корневые `notebook_completed.json` и `google_sheets_sync.json`;
- каталог эксперимента с checkpoint, tokenizer, `training_config.json` и
  `training_report.json`;
- `iid_validation_predictions.parquet`;
- `hard_validation_predictions.parquet`;
- `ood_validation_predictions.parquet`;
- training log.

Prediction parquet содержит pair IDs, label/target, raw `logit`, probability
`score`/`probability`, а при симметричной оценке также значения для обоих
порядков пары.

После скачивания всех четырёх запусков итоговая локальная таблица строится так:

```powershell
.venv\Scripts\python.exe scripts\summarize_architecture_baselines.py
```

CSV, JSON и Markdown появятся в `artifacts/architecture_baselines/`.

## Google Sheets

Notebooks используют отдельный лист `architecture_exps` и не вызывают запись в
`experiments_v2`, `pretrain_exps`, `sft_exps` или `data_exps`. Если лист ещё не
существует, первый успешный notebook создаст его и заголовки автоматически.

Текущий локальный Kaggle-пользователь — `dinakepecheva`, а попытка прочитать
настроенный приватный Dataset с Google service-account вернула HTTP 403. Перед
запуском нужно сделать одно из двух:

1. дать `dinakepecheva` доступ к существующему private credential Dataset;
2. создать его копию под доступным аккаунтом и записать её ref в
   `KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET` файла `.env`.

Service-account email из этого Dataset также должен иметь права Editor на
целевую Google-таблицу. Сам JSON-ключ нельзя отправлять в чат или коммитить.
Frozen validation и MiniLM checkpoint подключаются из Dataset аккаунта
`alexproger23`; raw human data подключается из
`dinakepecheva/e-cup-human-data`.
