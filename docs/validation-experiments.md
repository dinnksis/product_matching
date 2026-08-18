# Эксперименты на frozen IID, hard и OOD

Единый приватный Kaggle Dataset:
`alexproger23/product-matching-validation-splits-v1`.

Он фиксирует один train и три независимых протокола проверки:

- `human_train_pairs.parquet` — единственный human train, 306 669 пар;
- `human_iid_validation_pairs.parquet` — IID, 12 000 пар;
- `human_hard_validation_pairs.parquet` — hard, 5 814 пар;
- `human_ood_validation_pairs.parquet` — OOD, 41 171 пара только из категорий
  «Одежда» и «Бытовая техника»;
- `human_items.parquet` — тексты и категории для всех human item ID;
- `llm_non_ood_items.parquet` и `llm_non_ood_pairs.parquet` — допустимый источник
  дополнительного train;
- `llm_ood_items.parquet` и `llm_ood_pairs.parquet` — отложенная OOD-часть, её
  нельзя использовать для обучения или подбора гиперпараметров.

Human train, IID, hard и OOD не пересекаются по item ID. Все эксперименты должны
использовать ровно эти файлы: нельзя пересэмплировать validation или переносить
пары между split-ами. Для выбора модели основной показатель — macro AP на IID;
hard показывает устойчивость на специально сложных примерах, OOD — перенос на
две полностью исключённые из train категории.

Оба LLM item-каталога дополнительно очищены от всех item ID, встречающихся в
human IID, hard или OOD. Сборка Dataset завершается ошибкой, если хотя бы одна
LLM-пара касается frozen validation item. Таким образом, для `llm_non_ood_*`
нулевое пересечение гарантируется одновременно на уровне item-строк и пар.

## Готовый MiniLM baseline

Baseline обучает `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` одну эпоху только
на `human_train_pairs.parquet`, без LLM-разметки. После обучения один checkpoint
считается на всех трёх validation-наборах в обоих порядках пары.

Локально проверить будущую отправку, не обращаясь к Kaggle:

```bash
make kaggle-minilm-validation-dry-run
```

Создать notebook без запуска:

```bash
make kaggle-minilm-validation-build
```

Отправить приватный notebook, дождаться окончания и скачать outputs:

```bash
make kaggle-minilm-validation-run
```

Чтобы только отправить job и не ждать локально:

```bash
uv run python scripts/run_minilm_validation_baseline_kaggle.py --no-wait
```

Runner перед выделением GPU проверяет статус Dataset, полный набор файлов и
SHA-256 frozen manifest. Notebook сохраняет:

- `training_report.json` с секцией `validation_splits.{iid,hard,ood}`;
- `iid_validation_predictions.parquet`;
- `hard_validation_predictions.parquet`;
- `ood_validation_predictions.parquet`;
- checkpoint, training config, log и `notebook_completed.json`.

Для другого MiniLM-конфига скопируйте
`configs/cross_encoder_minilm_validation_baseline.json`, измените параметры и
передайте его runner-у:

```bash
uv run python scripts/run_minilm_validation_baseline_kaggle.py \
  --config configs/my_minilm_experiment.json
```

Для другой архитектуры контракт тот же: train берётся только из human train и,
если нужно, `llm_non_ood_*`; финальный report обязан содержать метрики `iid`,
`hard` и `ood`.

## Human fine-tune после LLM pretraining

Эксперимент `minilm_llm_pretrain_human_ft_v1` начинает обучение с локального
checkpoint `model/pretrain_minilm_1ep`, уже прошедшего одну эпоху на разрешённой
non-OOD LLM-разметке. Следующий stage использует только
`human_train_pairs.parquet`, создаёт optimizer и scheduler заново и валидирует
полученную модель на тех же frozen IID, hard и OOD.

Все параметры human fine-tune, кроме начального checkpoint, программно
сверяются с `configs/cross_encoder_minilm_validation_baseline.json`. Текущий
конфиг — `configs/cross_encoder_minilm_llm_pretrain_human_ft.json`; запуск
прервётся до отправки на Kaggle при любом случайном расхождении.

Для Kaggle в отдельный приватный Dataset загружаются только веса, tokenizer,
model config и небольшие provenance-файлы. `optimizer.pt`, `scheduler.pt`,
`rng_state.pt` и `elr_targets.pt` не загружаются, поскольку это новый fine-tune
stage, а не продолжение LLM-optimizer state:

```bash
make kaggle-minilm-pretrain-checkpoint-dry-run
make kaggle-minilm-pretrain-checkpoint
make kaggle-minilm-pretrain-human-ft-dry-run
make kaggle-minilm-pretrain-human-ft-run
```

Последняя команда ждёт завершения, скачивает outputs и добавляет одну компактную
строку с IID/hard/OOD метриками в `experiments_v2`.

Первый запуск завершён 2026-08-16:

| Validation | Human-only baseline | LLM pretrain 1ep → human FT | Δ к baseline |
| --- | ---: | ---: | ---: |
| IID | 0.705238 | 0.770041 | +0.064803 |
| hard | 0.268804 | 0.341307 | +0.072504 |
| OOD | 0.579545 | 0.630826 | +0.051280 |

Эксперимент использовал 306 669 human train-пар и checkpoint Dataset
`alexproger23/product-matching-minilm-llm-pretrain-1ep` v1 с manifest SHA-256
`5d8a6ac3b3e78b4cb61ddec4fd9453e6b6c49894f493e43bfd18a04fc40498ca`.
Kaggle kernel:
`alexproger23/product-matching-minilm-llm-pretrain-human-ft-v1`.

Тот же эксперимент после пяти эпох LLM pretraining запускается так:

```bash
uv run python scripts/run_minilm_llm_pretrain_human_ft_kaggle.py \
  --checkpoint-tag 5ep
```

Он завершён 2026-08-16 с неизменными human fine-tune параметрами и теми же
frozen validation splits:

| Validation | Human-only baseline | 1ep → human FT | 5ep → human FT | Δ 5ep к 1ep |
| --- | ---: | ---: | ---: | ---: |
| IID | 0.705238 | 0.770041 | 0.789388 | +0.019347 |
| hard | 0.268804 | 0.341307 | 0.365501 | +0.024194 |
| OOD | 0.579545 | 0.630826 | 0.642660 | +0.011834 |

Checkpoint Dataset —
`alexproger23/product-matching-minilm-llm-pretrain-5ep` v1 с manifest SHA-256
`354c7006898a9a44a3115c8384f12dbab520cfec7723a675f8ccedb108544533`.
Kaggle kernel:
`alexproger23/product-matching-minilm-5ep-human-ft-v1`.

Для командных экспериментов, в которых разрешено менять только train-данные и
loss при неизменных гиперпараметрах, используйте отдельный шаблон
[`notebooks/minilm_5ep_team_ablation/`](../notebooks/minilm_5ep_team_ablation/README.md).
В нём зафиксированы checkpoint, training recipe и три validation split, а две
редактируемые ячейки помечены тегом `team-editable`.

## Серверные baseline-запуски разных reranker-моделей

Для одной H100 80GB есть единый launcher
[`scripts/run_human_reranker_baseline.sh`](../scripts/run_human_reranker_baseline.sh).
Каждый запуск использует только `human/train_pairs.parquet`, затем считает один
checkpoint на frozen `iid`, `hard` и `ood` в обоих порядках пары. После успешной
валидации launcher автоматически делает idempotent upsert одной строки в
`experiments_v2`.

Установить общий набор GPU-зависимостей и показать доступные профили:

```bash
python -m pip install -r requirements-reranker-baselines.txt
scripts/run_human_reranker_baseline.sh --list
```

Примеры отдельных запусков:

```bash
scripts/run_human_reranker_baseline.sh gte
scripts/run_human_reranker_baseline.sh jina-v3.5
scripts/run_human_reranker_baseline.sh jina-v2
scripts/run_human_reranker_baseline.sh bge-v2-m3
scripts/run_human_reranker_baseline.sh qwen-0.6b
scripts/run_human_reranker_baseline.sh qwen-4b
scripts/run_human_reranker_baseline.sh rumodernbert
```

Чтобы последовательно выполнить всю очередь на одной GPU и останавливать её при
первой ошибке:

```bash
scripts/run_human_reranker_baseline.sh all
```

Это долгий запуск, особенно из-за Qwen-4B и его OOD validation; перед ним стоит
проверить все команды через `DRY_RUN=1`.

Launcher сам находит данные сначала в
`prepared/validation_splits_v1/human`, затем в
`data/validation_splits_v1/human`. Явный путь и GPU задаются через environment:

```bash
PREPARED_DIR=data/validation_splits_v1/human \
CUDA_VISIBLE_DEVICES=0 \
scripts/run_human_reranker_baseline.sh bge-v2-m3
```

Аргументы после alias добавляются в конец команды и позволяют сделать разовый
override без редактирования скрипта:

```bash
scripts/run_human_reranker_baseline.sh gte \
  --batch-size 160 \
  --eval-batch-size 384
```

Перед настоящим запуском профиль можно полностью просмотреть без загрузки модели:

```bash
DRY_RUN=1 scripts/run_human_reranker_baseline.sh qwen-4b
```

Профили различаются только там, где этого требует архитектура или память:

Во всех профилях зафиксированы одна эпоха, `max_length=384`, `sampling=none` и
effective batch `192`; для более крупных моделей он набирается через gradient
accumulation. Per-device batch, learning rate, attention backend и режим
checkpointing остаются model-specific. Полный config сохраняется в report, а
batch/LR — также в компактной строке таблицы.

| Alias | Checkpoint | Backend и режим |
| --- | --- | --- |
| `gte` | `Alibaba-NLP/gte-multilingual-reranker-base` | sequence classification, full FT, `eager` из-за воспроизводимого non-finite loss в SDPA на H100 |
| `jina-v3.5` | `jinaai/jina-reranker-v3.5` | custom LBNL prompt с одним document, full FT |
| `jina-v2` | `jinaai/jina-reranker-v2-base-multilingual` | sequence classification, full FT без optional Jina flash-attention extension |
| `bge-v2-m3` | `BAAI/bge-reranker-v2-m3` | sequence classification, full FT |
| `qwen-0.6b` | `Qwen/Qwen3-Reranker-0.6B` | causal yes/no reranker, full FT |
| `qwen-4b` | `Qwen/Qwen3-Reranker-4B` | causal yes/no reranker, LoRA rank 16, чтобы оставить запас памяти на одной H100 80GB |
| `rumodernbert` | `deepvk/RuModernBERT-base` | encoder full FT с новой случайно инициализированной binary classification head |

`Jina v3.5` — listwise LBNL-модель, поэтому её нельзя загружать через
`AutoModelForSequenceClassification`. Профиль сохраняет её штатные специальные
query/document tokens и оптимизирует выдаваемый cosine score на binary human
labels. Jina v2 и v3.5 имеют лицензию CC BY-NC 4.0; до использования в финальной
competition submission нужно отдельно подтвердить допустимость лицензии.

По умолчанию outputs складываются в
`model/baseline_<alias>_<UTC timestamp>/`, а переиспользуемый token cache — в
`artifacts/token_cache/<profile>/` (для GTE сохраняется уже использованный путь
`gte_multilingual_reranker_base`). В output остаются checkpoint,
`training_report.json`, три `*_validation_predictions.parquet`, полный
`training.log`, `server_run_completed.json` и `google_sheets_sync.json`.

### Автовыгрузка локальной валидации в experiments_v2

Для локального sync нужен тот же service-account, которому открыт spreadsheet.
Рекомендуемый вариант — путь в корневом `.env` (он читается без выполнения
shell-кода):

```dotenv
GOOGLE_SERVICE_ACCOUNT_JSON_PATH=/secure/path/google-service-account.json
```

После этого обычный запуск сам подхватит файл:

```bash
scripts/run_human_reranker_baseline.sh gte
```

Уже экспортированная environment variable имеет приоритет над `.env`. Также
поддерживается ignored-файл
`secrets/google-service-account.json` или raw JSON в
`GOOGLE_SERVICE_ACCOUNT_JSON`. Секрет не выводится в лог и не попадает в
completion artifact. Другой spreadsheet можно задать через
`EXPERIMENT_SPREADSHEET_ID`.

Если обучение завершилось, но Google API или credentials недоступны, модель и
все validation artifacts сохраняются, `google_sheets_sync.json` получает статус
`pending`, а launcher завершается с кодом `3`. После исправления доступа тот же
`run_id` безопасно синхронизируется повторной командой, напечатанной перед
обучением. Для уже готового стороннего отчёта:

```bash
.venv/bin/python scripts/sync_local_experiment_to_google_sheet.py \
  --report model/my_run/training_report.json \
  --experiment my_human_baseline
```

Sync до обращения к Google проверяет наличие всех трёх секций
`validation_splits.{iid,hard,ood}`, macro AP каждого split, Hard R@P99,
Hard ROC-AUC и OOD LogLoss. Неполный single-split report в таблицу не попадёт.
Если таблица временно не нужна, запуск можно сделать с `SYNC_GOOGLE_SHEETS=0` и
синхронизировать отчёт позднее.

## Google Sheets

Новые запуски записываются в компактный лист `experiments_v2`: одна строка на
`run_id`. В нём остаются `dataset_ref`, macro AP для IID/hard/OOD, три
диагностические метрики, число train-пар и основные гиперпараметры.
`kaggle_kernel_ref` и `code_bundle_sha256` сохраняются в completion/report JSON,
но из компактной таблицы удалены.

Диагностические колонки считаются по общим probabilities на соответствующем
frozen split:

- `hard_recall_at_p99` — максимальный recall среди score-threshold, для которых
  precision на hard не ниже `0.99`;
- `hard_roc_auc` — стандартный `roc_auc_score(y_true, y_score)` на всём hard;
- `ood_log_loss` — binary cross-entropy на OOD после clipping probabilities в
  `[1e-15, 1 - 1e-15]`; здесь меньше — лучше.

Для сравнения родственных запусков заведены ещё три листа:

- `pretrain_exps` — изменения pretraining/checkpoint;
- `sft_exps` — изменения supervised fine-tuning recipe;
- `data_exps` — изменения состава и подготовки train-данных.

При синхронизации группа задаётся явно полем `experiment_group` со значением
`pretrain`, `sft` или `data`; автоматической классификации по названию нет.
Baseline фиксируется отдельно в `B2` каждого листа. Пока `B2` пустая, строки не
подсвечиваются. Колонки `A:S` повторяют `experiments_v2` в том же порядке; все
поля baseline/statistical comparison добавлены только справа от них.

Статистику нельзя восстанавливать из одной пары scalar AP. Для каждой пары
candidate/baseline используются сохранённые
`iid_validation_predictions.parquet`, `hard_validation_predictions.parquet` и
`ood_validation_predictions.parquet`: парный component-level permutation test,
95% paired confidence interval и Holm correction для трёх split. Цвет строки
определяется по основной IID macro AP: приглушённый зелёный для значимого роста,
приглушённый красный для значимого падения и белый при отсутствии значимости.

В `minilm_5ep_team_ablation` это сравнение выполняется автоматически после
создания candidate predictions и до Sheets sync. Пользователь задаёт
`EXPERIMENT_SHEET` равным `pretrain_exps`, `sft_exps` или `data_exps`; notebook
проверяет manifest отдельного private Dataset
`alexproger23/product-matching-minilm-5ep-significance-v1` и сравнивает
с run `67f4fe76886b43d6b52ed5cb49068e1e`. Dataset содержит только пять нужных
колонок frozen predictions и собирается командами
`make kaggle-significance-baseline-dry-run` и
`make kaggle-significance-baseline`.

Время validation, throughput, VRAM, padding efficiency, per-category строки и
полный JSON больше не отправляются в таблицу. Эти подробности остаются в
`training_report.json` среди Kaggle outputs. Старые листы `experiments` и
`category_metrics` не удаляются и сохраняют историю прежних запусков.
