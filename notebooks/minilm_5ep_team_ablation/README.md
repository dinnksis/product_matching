# MiniLM 5ep: командный шаблон data/loss-абляций

Этот notebook начинает каждый эксперимент с одного и того же приватного
checkpoint `alexproger23/product-matching-minilm-llm-pretrain-5ep` и оценивает
итоговую модель на неизменных IID, hard и OOD split-ах human-разметки.

Цель шаблона — сравнивать изменения train-данных и loss, не смешивая их со
сменой training recipe. Зафиксированы:

- начальный MiniLM checkpoint после пяти эпох LLM pretraining;
- одна эпоха downstream fine-tune;
- learning rate `2e-5`, batch size `96` на GPU и gradient accumulation `1`;
- AdamW, cosine scheduler, warmup `0.05`, weight decay `0.01`;
- `max_length=384`, сериализация карточек и tokenizer;
- seed, sampling, clipping и остальные параметры;
- IID, hard и OOD validation-файлы и их SHA-256.

Notebook: [`minilm_5ep_team_ablation_2xt4.ipynb`](minilm_5ep_team_ablation_2xt4.ipynb).

## Что разрешено менять

В notebook есть ровно две группы ячеек с тегом `team-editable`.

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
validation paths и код сохранения отчёта. Перед обучением notebook повторно
проверяет frozen recipe. Если требуется другой LR, число эпох, backbone,
`max_length` или validation protocol, это уже отдельный экспериментальный
протокол, а не data/loss-абляция этого baseline.

Технически защиту notebook можно намеренно обойти, изменив одновременно guard и
ожидаемый hash. Она предназначена для предотвращения случайных изменений; при
ревью сравнивайте diff и убеждайтесь, что изменены только две разрешённые ячейки.

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

Нужен доступ к трём приватным Dataset:

- `alexproger23/product-matching-validation-splits-v1`;
- `alexproger23/product-matching-minilm-llm-pretrain-5ep`;
- `alexproger23/ecom-matching-google-sheets-credentials`.

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
  --no-env-sources \
  --no-wait
```

Сначала рекомендуется убрать `--no-wait` и добавить `--dry-run`. Для
дополнительных данных добавьте ещё один `--dataset owner/slug`.

Notebook рассчитан на две T4 и запускает DDP через `torchrun` с двумя
процессами. Он сохраняет checkpoint, predictions для трёх validation split,
training log, `training_report.json` и `notebook_completed.json` в
`/kaggle/working` и отправляет итог в `experiments_v2`.

## Воспроизводимость

Каждый завершённый запуск сохраняет:

- SHA-256 frozen training recipe;
- SHA-256 loss hook;
- SHA-256 итоговых train pairs и item catalogue;
- количество train-пар, positive rate и `label_source` counts;
- SHA embedded code bundle и начального checkpoint manifest;
- IID/hard/OOD macro AP, overall AP и predictions.

Сгенерировать committed notebook заново после изменения инфраструктуры:

```bash
uv run python scripts/create_minilm_5ep_team_ablation_notebook.py
```
