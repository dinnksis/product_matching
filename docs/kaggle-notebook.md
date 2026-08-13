# Как отправить notebook на Kaggle

Скрипт `scripts/run_kaggle_notebook.py` создаёт приватную копию notebook,
добавляет проверку двух T4, отправляет её на Kaggle, ждёт завершения и скачивает
содержимое `/kaggle/working`.

## 1. Подготовить окружение

Установите [uv](https://docs.astral.sh/uv/) и синхронизируйте зависимости:

```bash
uv sync
```

Kaggle CLI закреплён в `uv.lock`; отдельно устанавливать его не нужно.

## 2. Создать локальный `.env`

Скопируйте пример:

```bash
cp .env.example .env
```

Создайте API token на странице <https://www.kaggle.com/settings/api> и заполните
минимум два значения:

```dotenv
KAGGLE_API_TOKEN=<ваш token>
KAGGLE_USERNAME=<ваш Kaggle username>
```

`.env` исключён из Git. Token используется только локальным Kaggle CLI и не
добавляется в notebook, metadata или загружаемые sources.

## 3. Подключить входные данные и модели

При необходимости заполните в `.env` источники через запятую:

```dotenv
KAGGLE_DATASET_SOURCES=owner/dataset-slug
KAGGLE_COMPETITION_SOURCES=competition-slug
KAGGLE_KERNEL_SOURCES=owner/kernel-slug
KAGGLE_MODEL_SOURCES=owner/model/framework/variation/version
```

Используйте ссылки, которые показывает Kaggle для соответствующих resources.
Подключённые данные появятся под `/kaggle/input/`. Локальные файлы репозитория
автоматически не загружаются.

## 4. Сделать dry-run

```bash
make kaggle-dry-run NOTEBOOK=notebooks/train.ipynb
```

Dry-run не обращается к Kaggle. Он проверяет `.env` и notebook, удаляет outputs
из отправляемой копии, добавляет GPU preflight и создаёт:

```text
.kaggle/staging/<kernel-slug>/
├── kernel-metadata.json
└── notebook.ipynb
```

Исходный notebook не изменяется.

## 5. Запустить notebook

```bash
make kaggle-run NOTEBOOK=notebooks/train.ipynb
```

По умолчанию скрипт:

1. проверяет Kaggle credentials;
2. создаёт или обновляет приватный kernel;
3. запрашивает профиль `NvidiaTeslaT4`;
4. проверяет внутри notebook наличие ровно двух NVIDIA T4;
5. опрашивает статус до завершения;
6. при ошибке выводит remote log;
7. скачивает outputs в `artifacts/kaggle/<kernel-slug>/`.

Notebook должен сохранять нужные файлы именно в `/kaggle/working`, иначе Kaggle
не вернёт их как output.

## Обязательный контракт экспериментального notebook

Любой Kaggle notebook в этом проекте, который обучает, валидирует, сравнивает
или тестирует модель и производит метрики эксперимента, **обязан** отправлять
результат в общую Google-таблицу. Это относится как к новой версии существующего
notebook, так и к notebook с новым slug.

Экспериментальный notebook считается готовым к запуску только при выполнении
всех условий:

1. До начала эксперимента создаются стабильный UUID `run_id` и время старта.
2. Метрики, конфигурация и сведения об источниках сохраняются в
   `training_report.json` и `notebook_completed.json` в `/kaggle/working`.
3. После сохранения всех основных outputs расположена финальная ячейка
   синхронизации с Google Sheets. Она должна передавать как минимум `run_id`,
   время старта и завершения, статус, название эксперимента, модель,
   `dataset_ref`, `kaggle_kernel_ref`, SHA code bundle, конфигурацию, итоговые и
   покатегорийные метрики.
4. К notebook подключён private Dataset
   `alexproger23/ecom-matching-google-sheets-credentials`, а сам notebook также
   остаётся приватным.
5. Сбой Google Sheets не отменяет успешно завершённый эксперимент: финальная
   ячейка не делает `raise`, а сохраняет `google_sheets_sync.json` и при ошибке
   `sheets_sync_pending.json`.

Для notebook, генерируемых кодом, нельзя создавать собственную копию логики
синхронизации. Новый generator должен переиспользовать
`experiment_run_initialization_cell()` и `google_sheets_tracking_cells()` из
`scripts/create_qwen_training_notebook.py`. Так сохраняются единая схема,
идемпотентность по `run_id` и одинаковое поведение при ошибках.

При запуске через `scripts/run_kaggle_notebook.py` credential Dataset
подключается автоматически. Флаг `--no-google-sheets-credentials` допустим
только для неэкспериментальных или диагностических notebook; для настоящего
эксперимента его использовать нельзя. Если notebook создаётся вручную на сайте
Kaggle, перед запуском необходимо одновременно:

- добавить private Dataset через `Add Input`;
- проверить наличие финальной ячейки отправки метрик;
- оставить notebook приватным.

Запуск без финальной ячейки или без credential Dataset считается незавершённой
настройкой эксперимента, даже если обучение технически может выполниться.

## Автоматический журнал экспериментов в Google Sheets

Сгенерированные training notebooks после успешного сохранения модели записывают
`training_report.json` в таблицу
<https://docs.google.com/spreadsheets/d/1CtqT52XOrFyHfFt6rCiOMlnq6snJMlsMOJ0ubH79ikA/edit>:

- `experiments` содержит одну строку на запуск, основные метрики, конфиг и полный
  JSON-отчёт;
- `category_metrics` содержит AP каждой категории отдельными строками.

Для автоматической авторизации используется отдельный приватный Kaggle Dataset
`alexproger23/ecom-matching-google-sheets-credentials`. Создать или безопасно
обновить его из локального ключа можно командой:

```bash
make kaggle-google-credentials
```

`scripts/run_kaggle_notebook.py` автоматически добавляет этот Dataset в
`dataset_sources` каждого приватного notebook, включая новый slug. Для
конкретного запуска это можно отключить флагом
`--no-google-sheets-credentials`.

Новый notebook, а не новую версию существующего, можно создать той же командой,
передав ещё не использованный `--slug`:

```bash
uv run python scripts/run_kaggle_notebook.py notebooks/train.ipynb \
  --slug new-experiment-slug \
  --title "New Experiment"
```

При создании notebook вручную в web-интерфейсе runner не участвует, поэтому там
credential Dataset нужно добавить через `Add Input` вручную.

Kaggle Secret `GOOGLE_SERVICE_ACCOUNT_JSON` также поддерживается и имеет
приоритет, если он подключён вручную. Сам JSON-ключ не хранится в Git, не
встраивается в notebook и не копируется в Dataset с обучающими данными.

У каждого запуска есть UUID `run_id`. Повторное выполнение финальной ячейки
обновляет существующие строки и не создаёт дублей. Если Google временно
недоступен или credential Dataset не подключён, обучение всё равно остаётся
успешным, а в `/kaggle/working` сохраняются `google_sheets_sync.json` и
`sheets_sync_pending.json`. После исправления доступа достаточно повторно
выполнить последнюю ячейку.

Logger встраивается в notebook при генерации, а `google-auth` при необходимости
устанавливается только в финальной ячейке. Поэтому для изменений самого журнала
не требуется повторно загружать Dataset с training-данными.

## Дополнительные параметры

Скрипт можно запускать напрямую:

```bash
uv run python scripts/run_kaggle_notebook.py notebooks/train.ipynb --no-wait
uv run python scripts/run_kaggle_notebook.py notebooks/train.ipynb --no-download
uv run python scripts/run_kaggle_notebook.py notebooks/train.ipynb \
  --dataset owner/dataset-slug \
  --competition competition-slug
```

Если выбран другой GPU-профиль, отключите строгую проверку двух T4 флагом
`--no-gpu-check` и согласуйте `KAGGLE_ACCELERATOR` в `.env`.

Остановка локального ожидания через `Ctrl+C` не обязательно останавливает remote
kernel. Ссылку на него скрипт печатает перед отправкой.

## Полное обучение Qwen3-Reranker на двух T4

Для обучения используется сгенерированный notebook
`notebooks/qwen3_reranker_training_2xt4.ipynb` и отдельный приватный Dataset.
Dataset содержит `items_human.parquet`, `matches.parquet`, `requirements-gpu.txt`,
актуальные файлы `src/` и training scripts в code bundle, а также manifest с
SHA-256 каждого файла. Kaggle обычно автоматически разворачивает загруженный
ZIP в каталог `product_matching_training_code`; notebook поддерживает и каталог,
и исходный ZIP, проверяя хеш каждого файла до запуска. Поэтому код и данные не
могут незаметно разойтись.

Собрать notebook и локальный payload без обращения к Kaggle:

```bash
make kaggle-train-build
uv run python scripts/push_kaggle_training_dataset.py --dry-run
```

Создать приватный Dataset при первом запуске или добавить новую версию при
следующем:

```bash
make kaggle-train-data
```

Скрипт сам выбирает `datasets create` или `datasets version`, сохраняет Parquet
без преобразования в CSV и ждёт готовности новой версии. В конце он печатает
reference вида `owner/product-matching-qwen-training`.

Проверить будущую отправку notebook, явно подключив training Dataset:

```bash
uv run python scripts/run_kaggle_notebook.py \
  notebooks/qwen3_reranker_training_2xt4.ipynb \
  --dataset owner/product-matching-qwen-training \
  --no-env-sources \
  --dry-run
```

Отправить notebook и вернуть управление сразу после создания remote run:

```bash
uv run python scripts/run_kaggle_notebook.py \
  notebooks/qwen3_reranker_training_2xt4.ipynb \
  --dataset owner/product-matching-qwen-training \
  --no-env-sources \
  --no-wait
```

Без `--no-wait` runner будет отслеживать выполнение до конца и скачает model
adapter, `training_report.json`, полный training log и completion marker.

Notebook запускает DDP через `torch.distributed.run --nproc_per_node=2`.
`batch_size=32` относится к одной GPU (глобально 64), а `dataloader_workers=2`
создаёт по два worker-процесса на каждую T4. Сериализация и token cache строятся
до старта GPU-обучения, validation выполняется один раз после всех эпох.
Промежуточные prepared-файлы и token cache находятся в `/kaggle/temp`, чтобы не
попадать в скачиваемые notebook outputs.

## MiniLM cross-encoder без PEFT

Для более быстрого full fine-tuning добавлен отдельный notebook
`notebooks/cross_encoder_minilm_training_2xt4.ipynb`. Он использует
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, не импортирует PEFT и управляется
одним JSON-файлом `configs/cross_encoder_minilm.json`.

```bash
make kaggle-cross-build
make kaggle-train-data       # только при первом запуске или изменении src/scripts
make kaggle-cross-dry-run
make kaggle-cross-run
```

Подробное описание параметров, локального и Kaggle-запуска находится в
[`docs/cross-encoder-training.md`](cross-encoder-training.md).
