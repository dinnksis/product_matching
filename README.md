# E-CUP 2026 — Product Matching

Репозиторий решения задачи поиска дублей товарных карточек. Retrieval уже
выполнен организаторами: на вход подаются товары и подготовленные пары
кандидатов, для каждой пары нужно вернуть непрерывный score того, что товары
являются дублями.

Основное направление — cross-encoder на базе
[Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B),
дополненный быстрыми строковыми/атрибутными моделями, эвристиками и OOF-ансамблем.
Скорость считается частью качества решения: закрытые наборы нужно обработать за
жёстко ограниченное время в автономном Docker-контейнере.

## Задача и метрика

Данные охватывают 20 категорий. Обучающая разметка приходит из двух источников:

- около 365 тыс. пар с ручной разметкой;
- более 11 млн вероятностно размеченных LLM пар (ещё не добавлены в этот
  репозиторий).

Метрика — macro averaged PR-AUC: отдельно для каждой категории считается
`sklearn.metrics.average_precision_score`, затем 20 результатов усредняются.
Использовать `auc(precision, recall)` нельзя — это другая процедура подсчёта.

```python
from sklearn.metrics import average_precision_score

category_ap = pairs.groupby("category").apply(
    lambda part: average_precision_score(part["target"], part["predict"]),
    include_groups=False,
)
macro_ap = category_ap.mean()
```

В submit нужны все входные пары ровно один раз и строго три столбца:
`id1,id2,predict`. Для PR-AUC следует сохранять непрерывные scores, а не
пороговые классы.

Официальное описание структуры данных:
[страница соревнования ODS](https://ods.ai/competitions/e-cup-2026-matching/dataset).

## Данные в репозитории

Сейчас локально исследована только ручная разметка:

| Файл | Строк | Схема |
|---|---:|---|
| `data/items_human.parquet` | 711 304 | `id`, `name`, `attributes`, `category` |
| `data/matches.parquet` | 365 654 | `id1`, `id2`, `target` |

Parquet-файлы игнорируются Git. Подробнее — в [data/README.md](data/README.md).

Проверки текущей выгрузки:

- 93 890 положительных и 271 764 отрицательных пар;
- все 711 304 ID уникальны, все ссылки из пар существуют;
- нет пропусков, self-pairs, повторов неупорядоченных пар и межкатегорийных пар;
- все `attributes` разбираются как JSON-объекты;
- доля дублей сильно зависит от категории: от 7,3% до 56,2%;
- 97,7% карточек встречаются только в одной размеченной паре.

Последний пункт важен для валидации: пары делятся на фолды по компонентам графа,
чтобы одна карточка не попадала одновременно в train и validation.

## Воспроизводимый EDA

Готовые артефакты:

- [notebooks/01_human_data_eda.ipynb](notebooks/01_human_data_eda.ipynb) —
  выполненный notebook с кодом и выводами;
- [reports/human_data_eda.html](reports/human_data_eda.html) — статический отчёт;
- `reports/category_summary.csv`, `reports/univariate_feature_scores.csv`,
  `reports/light_baseline_by_category.csv` — таблицы для следующих экспериментов;
- `reports/eda_summary.json` — компактная сводка основных метрик.

Отчёт проверяет схему и связи, показывает распределения категорий/таргета,
анализирует JSON-атрибуты и граф пар, оценивает быстрые similarity-признаки и
строит component-disjoint 5-fold OOF baseline.

На текущих данных:

- лучший одиночный быстрый признак (`token_set_ratio` названий) даёт macro AP
  около **0,363**;
- лёгкий `HistGradientBoostingClassifier` на строковых, числовых, брендовых и
  JSON-признаках даёт **OOF macro AP 0,535** и overall AP 0,616;
- самые трудные для этого baseline категории — одежда, обувь и ювелирные
  изделия; сильнее всего он работает на детских товарах, хобби, бытовой технике
  и музыкальных инструментах;
- точное совпадение нормализованных названий не является достаточным правилом:
  только 39,5% таких размеченных пар положительны.

Цифры baseline нужны как нижняя граница, а не как оценка будущего leaderboard:
они получены на доступной ручной выборке.

## Быстрый старт

Нужны Python 3.11–3.12 и [uv](https://docs.astral.sh/uv/).

```bash
uv sync
make report
```

`make report` пересоздаёт notebook, исполняет его на локальных parquet-файлах и
экспортирует HTML. На текущем объёме расчёт занимает примерно 1–2 минуты на CPU.

## Удалённый запуск notebook на Kaggle (2×T4)

Один раз установите окружение и заполните локальный `.env`:

```bash
uv sync
# В .env обязательны KAGGLE_API_TOKEN и KAGGLE_USERNAME.
```

Токен создаётся в настройках Kaggle API. `.env` игнорируется Git и используется
только локальным CLI: его значение не попадает внутрь notebook. Проверить
подготовленную конфигурацию без обращения к Kaggle:

```bash
make kaggle-dry-run NOTEBOOK=notebooks/train.ipynb
```

Отправить notebook, дождаться полного выполнения и скачать файлы из
`/kaggle/working`:

```bash
make kaggle-run NOTEBOOK=notebooks/train.ipynb
```

Скрипт создаёт или обновляет приватный kernel, запрашивает ускоритель
`NvidiaTeslaT4`, проверяет в первой ячейке наличие ровно двух T4, опрашивает
статус и сохраняет результаты в `artifacts/kaggle/<kernel-slug>/`. Исходный
`.ipynb` не меняется; отправляемая копия лежит в `.kaggle/staging/`.

Локальные данные и файлы репозитория автоматически не загружаются. Нужные
Kaggle datasets/competition/model sources перечисляются через запятую в `.env`.
В notebook входы доступны в `/kaggle/input/<dataset-slug>/`, а сохраняемые
модели, checkpoints и метрики нужно писать в `/kaggle/working`, иначе Kaggle не
вернёт их как output. Выделение двух GPU само по себе не распараллеливает
обучение: notebook должен использовать DDP, `accelerate` или multi-GPU режим
конкретного trainer.

### Zero-shot Qwen3 reranker через vLLM

[notebooks/qwen3_vllm_inference_10k.ipynb](notebooks/qwen3_vllm_inference_10k.ipynb)
запускает `Qwen/Qwen3-Reranker-0.6B` без обучения на фиксированной
стратифицированной выборке из 10 000 ручных пар. Карточка сериализуется как
категория, название и полный упорядоченный JSON атрибутов; затем каждый товар
ограничивается 448 токенами. Обе T4 используются через
`tensor_parallel_size=2`.

Пересоздать notebook и локальный Parquet для приватного Kaggle Dataset:

```bash
uv run python scripts/create_qwen3_vllm_notebook.py
make kaggle-dry-run NOTEBOOK=notebooks/qwen3_vllm_inference_10k.ipynb
make kaggle-run NOTEBOOK=notebooks/qwen3_vllm_inference_10k.ipynb
```

Первый успешный zero-shot прогон выполнен приватной Kaggle-версией 3 на 2×T4:

- 10 000 пар за 412,6 секунды чистого inference — **24,23 пары/с**;
- загрузка модели — 95,3 секунды, токенизация — 13,5 секунды;
- truncation затронул 6,12% отдельных карточек;
- overall AP — **0,284**, macro AP — **0,336**;
- медиана score — 0,965, а 84,4% пар получили score не ниже 0,9.

Таким образом, технический inference работает, но исходный retrieval-reranker
не откалиброван под бинарное понятие дубля в этой задаче и в zero-shot режиме
уступает лёгкому OOF baseline. Предсказания, category AP, hard examples,
диагностический график и runtime JSON скачаны в
`artifacts/kaggle/product-matching-training/`.

### Контейнерный zero-shot submit

В `submits/qwen3-reranker-vllm-zero-shot/` лежит автономный speed-submit с
локальными весами Qwen3, vLLM 0.14.0 и vendored PyArrow. Он использует короткий
контекст 256 токенов и один H100 без tensor parallel. Готовый архив создаётся в
`submits/qwen3-reranker-vllm-zero-shot.zip`; модель не обучена и предназначена
в первую очередь для проверки end-to-end лимитов контейнера.

## План решения

### 1. Валидация

- Фиксированный 5-fold split по компонентам графа всех размеченных пар.
- Контроль отсутствия пересечения ID между train/validation.
- Основной критерий — macro AP; дополнительно сохраняются category AP, overall
  AP, wall-clock, pairs/sec, пиковая RAM/VRAM и размер артефактов.
- Все base-модели отдают OOF scores на одних и тех же фолдах. Stacker никогда не
  обучается на in-fold predictions.

### 2. Лёгкая модель и эвристики

- Нормализация Unicode, регистра, пробелов, единиц измерения и вариантов ключей.
- RapidFuzz, word/char TF-IDF, token/number overlap, признаки бренда,
  артикула/OEM/EAN, модели, размера, цвета и комплектации.
- CatBoost/LightGBM либо компактная линейная модель с category-specific
  взаимодействиями.
- High-precision правила только с проверкой variant-конфликтов: одинакового имени
  или бренда недостаточно.

Лёгкая ветка нужна сразу в трёх ролях: быстрый baseline, независимый сигнал для
ансамбля и gate для дорогой модели.

### 3. Cross-encoder

Первый кандидат — `Qwen/Qwen3-Reranker-0.6B`: это открытая Apache-2.0 модель для
text reranking на 0,6 млрд параметров; официальный model card показывает варианты
запуска через `sentence-transformers` и `transformers`.

Предлагаемый вход:

```text
Категория: <category>
Товар A: <name> | бренд: ... | артикул: ... | размер: ... | ...
Товар B: <name> | бренд: ... | артикул: ... | размер: ... | ...
```

Ключевые решения для абляций:

- длина 128/256/384/512 токенов и порядок атрибутов;
- случайная перестановка A/B для симметрии;
- category-balanced sampling и/или веса категорий;
- hard-negative mining из уверенных ошибок текущих моделей;
- human-only обучение как контрольный эксперимент;
- LLM pretraining/distillation с меньшим весом, затем human fine-tuning;
- BF16, quantization, dynamic padding и length bucketing для H100.

### 4. Ансамбль и каскад

На OOF-предсказаниях обучается небольшой stacker из:

- cross-encoder logits;
- TF-IDF/лёгкого model score;
- identifier и variant-conflict признаков;
- категории.

Сравниваются глобальный stacker и category-wise калибровка. Для ускорения можно
пропускать через cross-encoder только серую зону лёгкой модели, но такой каскад
должен сохранять глобальное ранжирование scores внутри каждой категории.

### 5. Работа с 11 млн LLM-меток

До смешивания с human labels:

1. канонизировать `(min(id1,id2), max(id1,id2))`;
2. удалить повторы и найти конфликты источников;
3. измерить согласие LLM с ручной разметкой по категориям и confidence-бинам;
4. оставить component-disjoint human holdout полностью чистым;
5. сравнить `human-only`, `LLM → human fine-tune` и weighted joint training.

Разрешены только модели с открытой лицензией; код обучения и переразметки должен
быть воспроизводимым.

## Ограничения контейнера

Решение запускается без интернета на 20 CPU, 200 GB RAM и NVIDIA H100 80 GB.
Ограничения времени: Check — 1 минута, Public — 6 минут, Private — 13 минут.
Архив решения — до 5 GB, Docker-образ в архивированном виде — до 15 GB.

Ожидаемый CLI:

```bash
python -u run.py \
  --items_path /data/items.parquet \
  --matches_path /data/matches.parquet \
  --output-path /output/submit.csv
```

Перед отправкой контейнер должен локально пройти контрактные проверки:

- работает без сети и содержит все веса/зависимости;
- сохраняет результат для каждой пары и не меняет порядок/ID;
- выдаёт только `id1,id2,predict`, без NaN/inf;
- укладывается в лимит времени на экстраполированном Private-объёме;
- `metadata.json` указывает существующий image и корректный `entry_point`.

## Структура проекта

```text
.
├── data/                       # локальные parquet-файлы (не в Git)
├── notebooks/
│   ├── 01_human_data_eda.ipynb
│   └── qwen3_vllm_inference_10k.ipynb
├── reports/                    # HTML и компактные таблицы EDA
├── scripts/
│   ├── create_eda_notebook.py
│   ├── create_qwen3_vllm_notebook.py
│   └── run_kaggle_notebook.py
├── src/product_matching/
│   └── eda.py                  # проверки, признаки, CV baseline
├── Makefile
├── pyproject.toml
└── uv.lock
```

Ближайшие практические эксперименты: char/word TF-IDF baseline и
human-only fine-tuning `Qwen3-Reranker-0.6B` на одном и том же component-disjoint
split, после чего — OOF blending и профилирование скорости.
