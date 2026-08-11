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
