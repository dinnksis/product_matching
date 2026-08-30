# MiniLM 5ep: командный шаблон data/loss-абляций

Этот notebook начинает каждый эксперимент с одного и того же приватного
checkpoint `alexproger23/product-matching-minilm-llm-pretrain-5ep` и оценивает
итоговую модель на неизменных IID, hard и OOD split-ах human-разметки.

Цель шаблона — сравнивать изменения train-данных и loss, не смешивая их со
сменой training recipe. После обучения notebook автоматически сравнивает
candidate с общим baseline `67f4fe76886b43d6b52ed5cb49068e1e`, считает
p-value/95% CI на трёх split и отправляет результат в `experiments_v2` и один
выбранный тематический лист. Зафиксированы:

- начальный MiniLM checkpoint после пяти эпох LLM pretraining;
- одна эпоха downstream fine-tune;
- learning rate `2e-5`, batch size `96` на GPU и gradient accumulation `1`;
- AdamW, cosine scheduler, warmup `0.05`, weight decay `0.01`;
- `max_length=384`, сериализация карточек и tokenizer;
- seed, sampling, clipping и остальные параметры;
- IID, hard и OOD validation-файлы и их SHA-256.
- baseline predictions для статистического сравнения и их SHA-256.

Notebook: [`minilm_5ep_team_ablation_2xt4.ipynb`](minilm_5ep_team_ablation_2xt4.ipynb).

Для подбора epochs, learning rate, batch size, weight decay, warmup и других
параметров supervised recipe не ослабляйте этот locked notebook. Используйте
отдельный SFT protocol из
[`docs/minilm-5ep-sft-hparam-search.md`](../../docs/minilm-5ep-sft-hparam-search.md):
он сохраняет тот же checkpoint, human train и frozen validation, но создаёт
отдельный immutable notebook для каждой разрешённой точки и направляет её в
`sft_exps`.

## Что нужно задать перед запуском

В ячейке `RUN SETUP — label and comparison sheet` задаются только три поля:

```python
EXPERIMENT_LABEL = "minilm_5ep_my_ablation"
EXPERIMENT_SHEET = "data_exps"  # pretrain_exps | sft_exps | data_exps
EXPERIMENT_NOTES = "Короткое описание проверяемой гипотезы"
```

`EXPERIMENT_LABEL` должен быть уникальным и состоять из строчных латинских букв,
цифр, `_` и `-`. В `experiments_v2` запуск записывается всегда. Значение
`EXPERIMENT_SHEET` выбирает дополнительный лист со статистическим сравнением:

- `pretrain_exps` — изменение pretraining или начального checkpoint;
- `sft_exps` — изменение loss/оптимизации supervised fine-tune;
- `data_exps` — изменение состава или подготовки train-данных.

Эта routing-ячейка имеет тег `run-configurable`. Она не считается изменением
frozen training recipe.

## Что разрешено менять в эксперименте

Кроме routing-ячейки в notebook есть ровно две алгоритмические группы ячеек с
тегом `team-editable`.

### `EDIT 1/2 — DATA HOOK`

Меняйте только тело функции:

```python
def build_train_data(human_train_pairs, human_items, input_root):
    ...
    return train_pairs, items
```

`train_pairs` должен содержать как минимум `id1`, `id2`, `target`. Можно также
вернуть `sample_weight`, `label_source` и другие колонки: они сохранятся в
`train_frame`, который получает loss hook. `items` должен содержать `id`,
`product_text`, `category` для всех train- и validation-item ID.

Защитная ячейка автоматически запрещает:

- попадание любого frozen validation item ID в train;
- изменение текста или категории validation items;
- пропущенные item ID, self-pairs и повторяющиеся unordered pairs;
- target вне `[0, 1]` и некорректные `sample_weight`.

Дополнительный Kaggle Dataset подключается при запуске через ещё один
`--dataset owner/slug`. В hook он доступен ниже `input_root` (`/kaggle/input`).

### `EDIT 2/2 — LOSS HOOK`

Меняйте только `initialize_loss` и `compute_loss` в помеченной ячейке.

`initialize_loss` вызывается один раз на каждом DDP rank и получает:

- `train_frame` — итоговый train после data hook и присоединения item fields;
- `device`, `rank`, `world_size`.

`compute_loss` вызывается на каждом batch и получает:

- `logits`, `targets`, `sample_weights`;
- глобальные `pair_indices` строк train и `orientations`;
- zero-based `epoch` и `step`.

Он должен вернуть scalar `torch.Tensor` либо словарь с обязательным scalar
тензором `loss`. Дополнительные scalar-поля, например `bce` или
`regularizer`, усредняются и записываются в log:

```python
return {
    "loss": bce + regularizer,
    "bce": bce.detach(),
    "regularizer": regularizer.detach(),
}
```

Модульные состояния допустимы. Например, `initialize_loss` может создать
per-pair history для ELR, используя число строк `train_frame` и
`pair_indices`. При DDP состояние существует отдельно на каждом rank, поэтому
обновлять его нужно только для индексов, увиденных этим rank.

## Что менять нельзя

Не редактируйте ячейки с тегом `frozen`, JSON config, команды `torchrun`,
validation paths, baseline manifest и код сохранения отчёта. Перед обучением
notebook повторно проверяет frozen recipe и все baseline prediction-файлы. Если
требуется другой LR, число эпох, backbone, `max_length` или validation protocol,
это уже отдельный экспериментальный протокол, а не data/loss-абляция этого
baseline.

Технически защиту notebook можно намеренно обойти, изменив одновременно guard и
ожидаемый hash. Она предназначена для предотвращения случайных изменений; при
ревью сравнивайте diff и убеждайтесь, что изменены только routing-ячейка и две
разрешённые алгоритмические ячейки.

## Сопоставимость data-экспериментов

При фиксированной одной эпохе больший train означает больше optimizer steps.
Notebook записывает число пар и `same_size_as_human_baseline` в completion
report. Для чистой абляции состава данных сохраняйте те же `306 669` пар. Если
размер отличается, результат валиден, но его нужно описывать как изменение
данных **и** compute budget.

## Подготовка и запуск

```bash
git fetch origin
git switch pretrain_exps
git pull --ff-only
uv sync
```

Нужен доступ к четырём приватным Dataset:

- `alexproger23/product-matching-validation-splits-v1`;
- `alexproger23/product-matching-minilm-llm-pretrain-5ep`;
- `alexproger23/product-matching-minilm-5ep-significance-v1`;
- `alexproger23/ecom-matching-google-sheets-credentials`.

Третий Dataset содержит только `id1,id2,target,category_1,score` для frozen
IID/hard/OOD baseline — около 1 MiB, без текстов товаров, модели и credential.
Последний Dataset содержит только Google service-account key; смешивать эти два
Dataset нельзя. Владелец даёт коллеге доступ к каждому private Dataset по его
Kaggle username через настройки доступа Dataset.

Перед первым общим запуском владелец создаёт или обновляет slim baseline Dataset:

```bash
make kaggle-significance-baseline-dry-run
make kaggle-significance-baseline
```

Dry-run собирает payload локально и не обращается к Kaggle. Обычная команда —
внешнее изменение Kaggle и выполняется только владельцем после проверки.

Для общего credential Dataset добавьте в локальный `.env`:

```dotenv
KAGGLE_GOOGLE_SHEETS_CREDENTIALS_DATASET=alexproger23/ecom-matching-google-sheets-credentials
```

У каждого эксперимента должны быть собственные slug и title:

```bash
uv run python scripts/run_kaggle_notebook.py \
  notebooks/minilm_5ep_team_ablation/minilm_5ep_team_ablation_2xt4.ipynb \
  --slug <username>-minilm-5ep-<experiment-name> \
  --title "MiniLM 5ep: <experiment-name>" \
  --dataset alexproger23/product-matching-validation-splits-v1 \
  --dataset alexproger23/product-matching-minilm-llm-pretrain-5ep \
  --dataset alexproger23/product-matching-minilm-5ep-significance-v1 \
  --no-env-sources \
  --no-wait
```

Сначала рекомендуется убрать `--no-wait` и добавить `--dry-run`. Для
дополнительных данных добавьте ещё один `--dataset owner/slug`.

Notebook рассчитан на две T4 и запускает DDP через `torchrun` с двумя
процессами. Он сохраняет checkpoint, predictions для трёх validation split,
training log, `training_report.json`, `baseline_comparison.json` и
`notebook_completed.json` в `/kaggle/working`. Затем один idempotent upsert
отправляет итог в `experiments_v2` и выбранный тематический лист.

Статистика считается по совпадающим frozen парам, а не по трём итоговым AP:

- двухсторонний paired permutation test меняет baseline/candidate scores внутри
  целых связных компонент товаров;
- 95% CI строится paired component bootstrap;
- raw p-value трёх split корректируются методом Holm;
- строка таблицы окрашивается по IID delta только когда IID Holm p-value не выше
  alpha листа. Baseline хранится в `B2`, alpha — в `E2`.

По умолчанию используются 2 000 permutations и 2 000 bootstrap resamples.
Расчёт выполняется после validation и может добавить несколько минут CPU-времени.
Повторный запуск ячеек significance + Google Sheets не переобучает модель и
обновляет строку по тому же `run_id`, не создавая дубль.

## Воспроизводимость

Каждый завершённый запуск сохраняет:

- SHA-256 frozen training recipe;
- SHA-256 loss hook;
- SHA-256 итоговых train pairs и item catalogue;
- количество train-пар, positive rate и `label_source` counts;
- SHA embedded code bundle и начального checkpoint manifest;
- IID/hard/OOD macro AP, overall AP и predictions.

Сгенерировать committed notebook заново после изменения инфраструктуры. Лист,
label и notes можно сразу передать в generator либо изменить в routing-ячейке:

```bash
uv run python scripts/create_minilm_5ep_team_ablation_notebook.py \
  --experiment-label minilm_5ep_my_ablation \
  --experiment-sheet data_exps \
  --notes "Проверка новых train-пар"
```
