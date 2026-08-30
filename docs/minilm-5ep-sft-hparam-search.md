# Подбор SFT-гиперпараметров MiniLM после 5ep pretraining

## Цель и исходная точка

Исследование подбирает supervised human fine-tuning recipe для checkpoint
`alexproger23/product-matching-minilm-llm-pretrain-5ep`. Пять эпох относятся к
предварительному обучению на LLM-разметке. Текущий downstream baseline проходит
human train только один раз.

Frozen baseline run `67f4fe76886b43d6b52ed5cb49068e1e`:

| Параметр | Значение |
| --- | ---: |
| Human train | 306 669 пар |
| Epochs | 1 |
| LR | `2e-5` |
| Batch на T4 | 96 |
| Effective batch | 192 |
| Weight decay | 0.01 |
| Warmup | 0.05 |
| Scheduler | cosine до нуля |
| Label smoothing | 0 |
| Max grad norm | 1.0 |
| Max length | 384 |
| Seed | 42 |
| IID macro AP | 0.789388 |
| Hard macro AP | 0.365501 |
| OOD macro AP | 0.642660 |

Чистое обучение baseline заняло около 18.2 минуты, весь training pipeline —
около 25.8 минуты. Paired significance добавляет примерно 7 минут. Peak VRAM
составил около 9.2 GiB на каждой T4.

## Почему используется отдельный protocol

[`minilm_5ep_team_ablation`](../notebooks/minilm_5ep_team_ablation/README.md)
фиксирует одну эпоху, LR, batch, optimizer и scheduler и разрешает менять только
train data и loss hook. Это необходимо для чистых data/loss сравнений. Ослаблять
его guard ради sweep нельзя.

SFT sweep использует отдельный generator
[`create_minilm_5ep_sft_hparam_notebooks.py`](../scripts/create_minilm_5ep_sft_hparam_notebooks.py).
Он наследует проверки checkpoint, data leakage, frozen validation и baseline
predictions, но принимает только явный allowlist SFT-параметров. Data hook всегда
заморожен. Loss hook также immutable внутри запуска и выбирается только из
hard-coded registry; произвольный Python из campaign JSON не исполняется.

## Что заморожено

Во всех запусках одинаковы:

- initial checkpoint и его manifest SHA-256;
- 306 669 human train-пар и item catalogue;
- IID, hard и OOD пары и их SHA-256;
- сериализация карточек, tokenizer и `max_length=384`;
- для optimizer-гиперпараметрических линий — обычный BCE без class/category
  weights; отдельная loss-стадия меняет только allowlisted hook;
- `sampling=none`, одна случайная ориентация пары в train;
- symmetric validation по A/B и B/A;
- full fine-tuning, DDP на двух T4, FP16 и SDPA;
- baseline predictions для парной статистики.

Каждая точка начинает обучение заново с одного initial checkpoint. Модель из
предыдущей точки не используется.

## Метрики и правила выбора

Единственная primary selection metric — macro AP на frozen IID split. Hard
служит стресс-диагностикой. OOD содержит две категории, полностью исключённые из
train, и не используется для выбора гиперпараметров. Это предотвращает подбор
по отложенным категориям.

Каждый notebook сохраняет predictions для всех трёх split и автоматически
считает против baseline:

- paired permutation test по connected components;
- paired component-bootstrap 95% CI;
- Holm correction для трёх split одного запуска.

Поиск создаёт много гипотез, поэтому одного внутри-run Holm недостаточно.
[`summarize_minilm_5ep_sft_hparams.py`](../scripts/summarize_minilm_5ep_sft_hparams.py)
дополнительно корректирует raw IID p-values по полной заранее зарезервированной
семье стадии; ещё не выполненные conditional slots считаются как `p=1` и не
показываются как результаты. Победитель последовательного coordinate-search затем
подтверждается отдельными seed,
а не объявляется финальным по единственному максимуму.

При разнице меньше 0.002 варианты считаются практически связанными до seed
confirmation. Если максимум находится на границе одномерной линии, диапазон
расширяется не более чем на одну внешнюю точку и только при преимуществе над
соседом больше `0.002`.

## План стадий

Полный machine-readable план находится в
[`minilm_5ep_sft_hparam_search_v1.json`](../configs/minilm_5ep_sft_hparam_search_v1.json).

Это большой, но не факториальный sweep. Типичный бюджет — 28 уникальных
Kaggle kernels вместе с уже завершённым control; жёсткий максимум — 37, если
сработают все boundary/combination/LR-refinement и second-finalist условия.
Parent-точки и seed 42 переиспользуются, поэтому повторно в бюджет не входят.

### 0. Current-protocol control

Первым отдельно запускается `minilm5_sft_e1_lr2e5_control_v1`. Он использует
точный baseline recipe, но актуальный embedded source. Остальная серия не
продолжается, если control отличается более чем на 0.001 IID macro AP, меняет
число train-пар/config hashes или показывает необъяснимый runtime drift.

Этот control был отправлен за несколько минут до замены первоначального grid
на LR-линию, поэтому его remote `notes.stage` равен `lr_epochs_grid`. Plan
содержит узкий alias на точный SHA этой единственной notes-строки; все recipe,
source, data и BCE hashes совпадают. Для любого нового run требуется уже точный
`lr_log_line`, общий fallback для старых notes запрещён.

### 1. Логарифмическая LR-линия

При строго одной human-эпохе сравниваются `5e-6`, `1e-5`, `2e-5`, `4e-5`.
Точка `2e-5` переиспользует current-protocol control. Если край выигрывает у
ближайшей внутренней точки более чем на `0.002`, добавляется ровно одна точка:
`2.5e-6` или `8e-5`.

Это не Cartesian grid: сначала определяется масштаб LR, остальные координаты
зафиксированы.

### 2. Линия по числу эпох

При выбранном LR выполняются свежие рецепты на `1`, `2`, `3` эпохи.
Одноэпоховая точка переиспользуется; реально нужны два новых запуска. Если 3
эпохи выигрывают у 2 более чем на `0.002`, один раз проверяются 4 эпохи. Cosine
horizon зависит от полного числа updates, поэтому метрики эпох 1/2 нельзя
достоверно взять из одного трёхэпохового запуска.

### 3. Coordinate-search регуляризации и optimizer geometry

Вокруг выбранной LR/epoch точки последовательно проверяются одномерные линии:

- effective batch: `96`, `192`, `384`;
- warmup: `0`, `0.05`, `0.1`;
- weight decay: `0`, `0.01`, `0.05` с условной границей `0.1`;
- label smoothing: `0`, `0.02`, `0.05` с условной границей `0.1`;
- classifier dropout: `0`, `0.1`, `0.2`;
- max grad norm: `0.5`, `1`, `2`.

На каждой координате текущий anchor переиспользуется и запускаются только две
альтернативы. Anchor обновляется лишь при IID-приросте больше `0.002`; внутри
практической ничьей сохраняется более простой или дешёвый рецепт. Каждый вариант
всё равно стартует с исходного 5ep checkpoint, а не продолжает веса предыдущего.

`warmup_ratio=0` в текущем trainer означает один минимальный warmup update из-за
`max(1, ...)`. В отчёте это трактуется как `one_step_warmup`, а не математически
нулевой warmup.

### 4. Специальные loss-функции

При выбранном optimizer recipe сравниваются plain BCE anchor и четыре новых
loss hook:

- `balanced_binary_bce`: одинаковая суммарная масса positive/negative;
- `balanced_category_class_sqrt_bce`: мягкие веса
  `1/sqrt(n_category,class)`;
- `balanced_category_class_bce`: одинаковая масса всех 36 train
  category×class strata;
- `focal_bce_gamma2_scale4`: focal `gamma=2` с фиксированным множителем `4`;
  `p_t` берётся по исходному hard class даже при label smoothing, а при
  нейтральной вероятности `p_t=0.5` итоговый вес равен 1.

Отдельный category-only balance не входит в primary screen: размеры 18 train
категорий уже лежат в узком диапазоне 16 325–17 987 пар (max/min `1.102`), так
что его веса были бы всего около `0.95–1.05` и ожидаемый эффект меньше seed
noise и practical tie `0.002`. Category×class variants проверяют существенно
более сильный дисбаланс prevalence `7.3–56%`.

Все 306 669 human-пар остаются на месте, веса дополнительно умножаются на
`sample_weights`, которые frozen protocol фиксирует равными 1, а loss делится
на размер microbatch. Поэтому глобально нормированные balance-веса дают
unbiased minibatch estimate заявленного weighted risk, не самонормировку
текущего batch. Если balanced и focal по
отдельности полезны, допускается один заранее allowlisted combination check с
тем же вариантом balance и фиксированно масштабированным focal-множителем.
Формально combination запускается, только когда raw IID delta относительно
tuned-BCE строго положительна и у лучшего balance-варианта, и у focal; лучший
balance выбирается по IID AP с tie-break в порядке объявления loss в plan.
В combination loss focal дополнительно перераспределяет массу по сложности,
поэтому точное равенство итоговой массы category×class strata уже не заявляется.
Победивший non-BCE loss получает короткое LR-уточнение `0.5×/1×/2×`, поскольку
форма loss может сдвинуть оптимальный LR. Две новые LR-точки запускаются только
при seed-42 улучшении non-BCE над tuned-BCE строго больше `0.002`; центральная
точка переиспользуется и внешнего LR boundary для этой ветки нет.

### 5. Matched-seed confirmation

Исходный baseline, tuned-BCE и один-два loss-финалиста проверяются на одинаковых
seed `17`, `42`, `2026`; сравнения строятся seed-к-seed, а seed 42 переиспользует
screening run. В финальный submission входит один checkpoint, а не seed
ensemble: inference budget соревнования не допускает умножение cross-encoder
inference.

То есть `top_k_loss_finalists=2` относится именно к loss-рецептам и не исключает
из matched-seed набора ни исходный baseline, ни tuned-BCE. При одном loss-финалисте
нужно шесть новых запусков (по seed 17 и 2026 для трёх рецептов), при втором —
восемь; уже имеющиеся seed-42 точки повторно не обучаются.
Первый loss-финалист — лучший non-BCE после возможного LR-refine и проверяется
даже при неудачном одиночном seed 42. Второй включается только если это другой
loss-вариант, его raw delta к tuned-BCE положительна, а отставание от первого не
превышает `0.002`.

Гиперпараметры не меняют inference graph, поэтому Docker runtime повторно
измеряется только для выбранного checkpoint. Контрольные soft limits с 20%
запасом: Check меньше 48 секунд, Public меньше 288 секунд, Private меньше 624
секунд.

## Генерация и dry-run

Сгенерировать все ready-варианты:

```bash
uv run python scripts/create_minilm_5ep_sft_hparam_notebooks.py \
  --stage lr_log_line
```

Проверить один exact control без обращения к Kaggle:

```bash
uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --dry-run \
  --only minilm5_sft_e1_lr2e5_control_v1
```

Dry-run проверяет notebook syntax, два T4 preflight metadata и четыре private
Dataset inputs: validation, initial checkpoint, baseline predictions и Google
Sheets credentials.

## Отправка и мониторинг

Отправить control в фоне:

```bash
uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --submit \
  --only minilm5_sft_e1_lr2e5_control_v1
```

Проверить существующий run и скачать его после завершения:

```bash
uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --status \
  --download-complete \
  --only minilm5_sft_e1_lr2e5_control_v1
```

После прохождения control вся ready LR-линия запускается последовательно:

```bash
uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --submit \
  --wait \
  --stage lr_log_line
```

Так launcher держит только одну GPU-сессию, после каждого run скачивает slim
набор reports/logs/predictions и переходит к следующей точке. Model weights по
умолчанию не скачиваются; для выбранного финалиста используется
`--full-download`. Уникальные kernel slug позволяют безопасно возобновлять
мониторинг без создания новой kernel version.

## `sft_exps` и локальная сводка

Каждый завершённый notebook делает idempotent upsert в `experiments_v2` и
`sft_exps`. В стандартных колонках видны epochs, per-GPU batch, gradient
accumulation, LR, max length и seed. Остальные параметры записываются
machine-readable JSON в `notes` и полностью остаются в
`training_report.json.args`.

После скачивания outputs:

```bash
uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage lr_log_line
```

Сводка создаёт:

```text
reports/minilm_5ep_sft_hparam_search_v1/
├── runs.csv
├── summary.json
└── report.md
```

Она валидирует experiment label, `experiment_group=sft`, frozen baseline ID,
три validation split и фактические config overrides, затем считает stage-wise
Holm correction по всей заранее заданной семье non-control вариантов стадии.

Если summary выставил `needs_boundary_extension=true`, сначала материализуется
ровно один заранее объявленный внешний уровень той же стадии:

```bash
uv run python scripts/materialize_minilm_5ep_sft_hparam_stage.py \
  --from-stage lr_log_line \
  --boundary-extension

uv run python scripts/create_minilm_5ep_sft_hparam_notebooks.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/lr_log_line_boundary.lock.json

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/lr_log_line_boundary.lock.json \
  --submit --wait

uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/lr_log_line_boundary.lock.json
```

Boundary lock содержит точные recipe/source/loss/IID/completion/notes hashes всех
уже завершённых строк и запускает только новую внешнюю точку. Итоговая сводка
объединяет их в исходную Holm family, переводит единственный reserved slot в
planned hypothesis и помечает расширение использованным; второй внешний уровень
той же оси запросить нельзя. Для coordinate-stage в `--from-stage` передаётся
эффективное имя, например
`regularization_coordinate_search__weight_decay`.

Только когда итоговый summary имеет `needs_boundary_extension=false`, следующий
anchor материализуется один раз в immutable lock:

```bash
uv run python scripts/materialize_minilm_5ep_sft_hparam_stage.py \
  --from-stage lr_log_line \
  --to-stage epoch_line
```

Lock фиксирует parent experiment/run ID, recipe SHA, SHA IID predictions и
полный resolved config. Повторный вызов переиспользует byte-identical lock;
изменившийся plan, summary или parent приводит к отказу, а не к молчаливому
перевыбору anchor.

Сгенерировать, проверить и затем последовательно выполнить только варианты из
lock:

```bash
uv run python scripts/create_minilm_5ep_sft_hparam_notebooks.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/epoch_line.lock.json

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/epoch_line.lock.json \
  --dry-run

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/epoch_line.lock.json \
  --submit --wait

uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/epoch_line.lock.json
```

Locked-stage summary повторно читает IID parquet parent/candidate, считает
paired component permutation/bootstrap именно относительно stage anchor и
применяет Holm к полной family с reserved conditional slot. Сравнение в
`sft_exps` при этом остаётся против общего frozen baseline для совместимости
между всеми экспериментами проекта.

Coordinate substages разрешены только в порядке из plan. Полная цепочка
переходов:

```text
lr_log_line
→ epoch_line
→ regularization_coordinate_search__effective_batch
→ regularization_coordinate_search__warmup_ratio
→ regularization_coordinate_search__weight_decay
→ regularization_coordinate_search__label_smoothing
→ regularization_coordinate_search__classifier_dropout
→ regularization_coordinate_search__max_grad_norm
```

Например, первый coordinate lock создаётся так:

```bash
uv run python scripts/materialize_minilm_5ep_sft_hparam_stage.py \
  --from-stage epoch_line \
  --to-stage regularization_coordinate_search \
  --coordinate effective_batch
```

Для следующего шага `--from-stage` должен быть ровно предыдущим эффективным
именем, а `--coordinate` — следующим элементом цепочки. Materializer и notebook
generator независимо отклоняют пропуск, повтор или перестановку координат даже
для корректно перехешированного lock.

## Исполнение adaptive loss и confirmation lock

После финальной координаты следующие стадии материализуются и исполняются
строго последовательно. Для primary screen prerequisite — последний
schema-v1 lock:

```bash
uv run python scripts/materialize_minilm_5ep_sft_loss_confirmation.py loss_primary \
  --prerequisite-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/regularization_coordinate_search_max_grad_norm.lock.json

uv run python scripts/create_minilm_5ep_sft_hparam_notebooks.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__primary.lock.json

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__primary.lock.json \
  --dry-run

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__primary.lock.json \
  --submit --wait

uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__primary.lock.json
```

Conditional balance×focal overlay использует primary summary и lock:

```bash
uv run python scripts/materialize_minilm_5ep_sft_loss_confirmation.py loss_overlay \
  --prerequisite-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__primary.lock.json

uv run python scripts/create_minilm_5ep_sft_hparam_notebooks.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__overlay.lock.json

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__overlay.lock.json \
  --submit --wait

uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__overlay.lock.json
```

Если trigger не сработал, materializer создаёт immutable skipped receipt вместо
runnable lock. Generator возвращает пустой список, launcher валидирует receipt и
завершается с `kaggle_actions=0`, а summarizer всё равно пишет complete closure
summary. Поэтому приведённая последовательность одинакова для обеих веток и не
нуждается в ручном редактировании JSON.

LR-refine получает оба prerequisite, включая skipped overlay receipt:

```bash
uv run python scripts/materialize_minilm_5ep_sft_loss_confirmation.py loss_lr_refine \
  --prerequisite-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__primary.lock.json \
  --prerequisite-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__overlay.lock.json

uv run python scripts/create_minilm_5ep_sft_hparam_notebooks.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__lr_refine.lock.json

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__lr_refine.lock.json \
  --submit --wait

uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__lr_refine.lock.json
```

Confirmation материализуется из всей adaptive closure и отдельной frozen
baseline summary:

```bash
uv run python scripts/materialize_minilm_5ep_sft_loss_confirmation.py confirmation \
  --baseline-summary reports/minilm_5ep_sft_hparam_search_v1/stages/lr_log_line/summary.json \
  --prerequisite-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__primary.lock.json \
  --prerequisite-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__overlay.lock.json \
  --prerequisite-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/special_loss_screen__lr_refine.lock.json

uv run python scripts/create_minilm_5ep_sft_hparam_notebooks.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/confirmation__matched_seeds.lock.json

uv run python scripts/run_minilm_5ep_sft_hparam_kaggle.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/confirmation__matched_seeds.lock.json \
  --submit --wait

uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/confirmation__matched_seeds.lock.json
```

Последняя команда без runtime attestation сохраняет summary со статусом
`runtime_gate_pending`. После Docker-проверки именно выбранного
`selection_before_runtime_gate` её повторяют с аргументом:

```bash
uv run python scripts/summarize_minilm_5ep_sft_hparams.py \
  --stage-lock reports/minilm_5ep_sft_hparam_search_v1/stage_locks/confirmation__matched_seeds.lock.json \
  --inference-runtime-check reports/minilm_5ep_sft_hparam_search_v1/confirmation_runtime_check.json
```

Runtime JSON — canonical schema-v1 object с точными полями `campaign`,
`confirmation_lock_payload_sha256`, `status=passed`, выбранными
`selected_recipe_group_id` и `checked_recipe_family_sha256`, а также
`check_seconds`, `public_seconds`, `private_seconds` и
`runtime_check_payload_sha256`. Последний SHA считается по объекту без самого
поля SHA. Файл записывается как `canonical_json_dumps(payload) + "\n"`: ключи
отсортированы, лишнего whitespace и duplicate keys нет. Attestation принимается
только для выбранной recipe family и при строгих soft limits `<48`, `<288`,
`<624` секунд соответственно.

Adaptive summary хранит runnable SHA только в `execution_lock_sha256s`, skipped
SHA отдельно в `execution_receipt_sha256s`, а их полный closure union — в
`execution_campaign_lock_sha256s`. `runs` содержит каждый переиспользованный
run ровно один раз; Holm-проекции вынесены в `hypothesis_families`. Budget
ledger фиксирует `history_complete_through`, полный union kernel slug и hard cap
37, поэтому следующий materializer может использовать summary напрямую без
повторного выбора parent.

### Локальная граница доверия schema-v2

Рядом с каждым adaptive lock materializer write-once создаёт два обязательных
объекта:

```text
<stage>.lock.json
<stage>.lock.json.trusted-provenance.json
<stage>.lock.json.trusted-provenance/
```

Последний каталог содержит immutable архив summary/lock authorities, которыми
было принято решение. Generator, launcher и summarizer выводят sidecar path
только из локального lock path через фиксированное правило и валидируют его до
генерации notebook, staging или обращения к Kaggle. Manifest нельзя передать
через CLI, lock payload, remote artifact или summary. Lock нельзя переносить,
переименовывать или копировать отдельно от его sidecar/archive; path mismatch
fail-closed. `--artifacts-dir` summarizer и вычисленный launcher-ом
`KAGGLE_OUTPUT_DIR` обязаны точно совпадать с artifacts root, зафиксированным
sidecar; submit/status отклоняются до первого Kaggle-вызова при несовпадении.

Для schema-v2 при каждом локальном чтении заново вычисляется SHA текущего
embedded training-source bundle и сравнивается одновременно с top-level lock и
каждым runnable variant. Generator ещё раз сверяет фактическую notebook metadata
до записи файла. Launcher не перезаписывает нормализованный lock contract
ответом generator: он сравнивает exact source/recipe/config/notes/loss,
role/family, slug/path и origin lineage и при любом drift останавливается до
чтения `.env`, subprocess или Kaggle API.

Root `summary.json` остаётся указателем на последнюю стадию и закономерно
перезаписывается. Уже созданный lock от него не зависит: при materialization
summary копируется в content-bound write-once archive. Повреждение sidecar или
архивной копии делает replay невалидным, тогда как нормальная смена root summary
после следующей стадии старый lock не ломает.

## Интерпретационные ограничения

- Frozen IID остаётся одной validation realization. Большой перебор без seed
  confirmation переобучается на неё.
- Component permutation измеряет uncertainty на validation examples, но не
  training variance.
- Hard split намеренно обогащён ошибками MiniLM и не является естественной test
  distribution.
- OOD состоит только из «Одежды» и «Бытовой техники». Его нельзя считать
  независимым после ручного выбора модели по OOD.
- Requirements используют диапазоны версий. Exact control перед sweep нужен в
  том числе для выявления дрейфа Kaggle environment.
