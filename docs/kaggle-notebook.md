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
## Embedding + boosting experiment

The repository has a dedicated autonomous launcher for the three CatBoost
ablations described in `docs/data-findings.md`:

```powershell
python scripts/run_embedding_boosting_kaggle.py --dry-run
python scripts/run_embedding_boosting_kaggle.py
```

The second command submits the notebook and returns while Kaggle continues in
the background. Closing the browser or the local terminal does not stop the
remote kernel. Later, wait for it and download all `/kaggle/working` artifacts:

```powershell
python scripts/run_embedding_boosting_kaggle.py --monitor-existing
```

Outputs are downloaded to
`artifacts/kaggle/product-matching-embedding-boosting/`. The remote output
contains three CatBoost models, validation predictions, macro/per-category AP,
feature importances, timings, selected train-only attribute keys, the Qwen
float16 item-embedding cache, the exact Qwen model snapshot, logs, the exact config, a manifest and a
`COMPLETED` marker. These are experiment artifacts; a separate offline submit
builder will package the winning preprocessing code, CatBoost model and Qwen
weights after the ablation result is known.

The launcher reuses the existing private Dataset
`<KAGGLE_USERNAME>/e-cup-human-data` for `items_human.parquet` and
`matches.parquet`. It creates a separate small private Dataset containing only
the versioned code/config bundle, so the parquet files are never uploaded a
second time.
