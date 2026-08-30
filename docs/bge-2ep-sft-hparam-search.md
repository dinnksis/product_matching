# BGE 2ep: cost-aware SFT search с former OOD в train

## Статус и границы протокола

Кампания `bge_2ep_sft_oodtrain_v1` подбирает supervised human fine-tuning для
локального checkpoint `model/pretrain_bge_2ep`. Это отдельный baseline и
отдельная статистическая семья: результаты нельзя вычитать из MiniLM-метрик,
поскольку одновременно меняются backbone и состав train.

На текущем этапе зафиксированы только machine-readable план, этот документ и
тест контракта. BGE-специфические generator, validator, Kaggle launcher,
summarizer и controller ещё не реализованы. Нельзя передавать этот JSON
MiniLM-скриптам или отправлять notebook до появления и проверки отдельного
исполняющего пути.

## Исходный checkpoint

`pretrain_bge_2ep` — `XLMRobertaForSequenceClassification` с одним ranking
logit. В checkpoint 567 755 777 FP32-параметров: 24 слоя, hidden size 1024,
16 attention heads и intermediate size 4096. Для сравнения, исследованный
MiniLM содержит 117 641 089 параметров; BGE больше в 4.83 раза по числу
параметров, а плотная transformer-часть требует примерно в 14 раз больше
операций на токен. Поэтому BGE-точки запускаются только последовательно и
кампания имеет существенно меньший бюджет.

SHA-256 всех пяти checkpoint-файлов записаны в
[`configs/bge_2ep_sft_hparam_search_v1.json`](../configs/bge_2ep_sft_hparam_search_v1.json).
Каждая точка должна начинаться заново с этого checkpoint. Продолжать обучение
от checkpoint предыдущей точки запрещено; optimizer и cosine scheduler также
создаются заново.

## Train с former OOD и доступная validation

Train строится детерминированной конкатенацией в таком порядке:

1. `human/train_pairs.parquet`: 306 669 пар и 80 136 positive;
2. бывший `human/ood_validation_pairs.parquet`: 41 171 пара и 9 155 positive.

Итог — 347 840 пар, 89 291 positive (`25.6701%`) и все 20 категорий. Все строки
сохраняются, sampling и loss weights для baseline отсутствуют. Перед запуском
обязательны проверки unordered duplicates, self-pairs, cross-category pairs,
target, разрешения item ID и нулевого item-ID overlap с IID и hard.

Бывший OOD состоит из категорий «Бытовая техника» и «Одежда». Он полностью
переходит в train, поэтому больше не является validation. Доступны только:

| Split | Пар | Positive | Категорий | Роль |
| --- | ---: | ---: | ---: | --- |
| IID | 12 000 | 3 118 | 18 | единственная selection metric |
| hard | 5 814 | 1 481 | 18 | диагностика |
| OOD | — | — | — | отключён, поскольку перенесён в train |

В `sft_exps` поля `ood_macro_ap` и `ood_delta` всегда равны `-1`; OOD p-value,
CI и per-category metrics должны быть `null`. Значение `-1` — sentinel, а не
измеренная метрика, и оно не участвует в Holm correction. Следствие протокола:
для двух перенесённых категорий нет независимой оценки качества, поэтому IID
campaign score нельзя интерпретировать как надёжную оценку macro AP по всем
20 категориям.

## Новый BGE baseline

Baseline создаётся первым и получает собственный `run_id`. Нельзя использовать
MiniLM baseline predictions или его deltas.

| Параметр | Значение |
| --- | ---: |
| Checkpoint | `model/pretrain_bge_2ep` |
| Epochs | 1 |
| Learning rate | `2e-5` |
| Batch на T4 | 8 |
| Gradient accumulation | 12 |
| GPU | 2 × T4 |
| Effective batch | 192 |
| Loss | plain BCE |
| Scheduler | cosine до нуля |
| Warmup | 0.05 |
| Weight decay | 0.01 |
| Label smoothing | 0 |
| Classifier dropout | 0.1 |
| Max grad norm | 0.5 |
| Max length | 384 |
| Sampling/weights | none |
| Gradient checkpointing | true |
| Seed | 42 |

Перед первым optimizer update baseline делает memory preflight. Безопасная
точка — `8 × 2 GPU × accumulation 12`. Если `16 × 2 × accumulation 6`
проходит с минимум 1 GiB свободной VRAM, разрешено один раз выбрать её. При OOM
fallback — `4 × 2 × accumulation 24`. Во всех случаях effective batch остаётся
192. Выбранная физическая геометрия записывается в identity baseline и затем
замораживается. Изменение геометрии после baseline требует нового baseline.

## Стадии поиска

### 1. Log2-линия learning rate

При одной эпохе и plain BCE выполняются `1e-5`, baseline `2e-5` и `4e-5`.
Baseline переиспользуется как центральная точка. Это одномерная линия, а не
Cartesian grid.

Если край выигрывает соседнюю центральную точку строго больше чем на `0.002`
IID macro AP, допускается ровно одна boundary-точка: `5e-6` для нижнего края
или `8e-5` для верхнего. Если условие одновременно выполнили оба края,
расширяется край с большим IID; при точном равенстве выбирается меньший LR.
В пределах practical tie сохраняется `2e-5`, затем предпочтение отдаётся
меньшему LR.

### 2. Число эпох

Точка epoch 1 на выбранном LR переиспользуется, epoch 2 запускается всегда.
Epoch 3 выполняется только когда epoch 2 выигрывает epoch 1 строго больше чем
на `0.002` IID и ожидаемый полный pipeline укладывается во внутренний soft limit
9 часов. Epoch 4 запрещена: MiniLM уже показал регрессию на четвёртом проходе,
а цена BGE-run слишком велика.

Каждое число эпох — отдельный свежий рецепт, поскольку меняется полный cosine
horizon. Побеждает минимальное число эпох в пределах `0.002` от численного
максимума.

### 3. Один sanity-check регуляризации

MiniLM показал широкое плато по effective batch, warmup, weight decay, label
smoothing, dropout и clipping. Поэтому эти линии не повторяются. Только если
победили как минимум две эпохи, разрешён один BGE-specific check
`weight_decay=0.05` против anchor `0.01`. Он принимается лишь при IID-приросте
строго больше `0.002`; иначе остаётся `0.01`.

### 4. Один transfer-check loss

Plain BCE сравнивается только с `balanced_category_class_sqrt_bce`. Для каждой
из 40 category×hard-class strata используется `1/sqrt(n)`, после чего веса
глобально нормируются до среднего 1. На augmented train ожидаемый диапазон
весов — `0.6814–2.7535`.

Binary balance исключён, поскольку ухудшил MiniLM. Full category×class balance
исключён как слишком агрессивный: его веса составили бы примерно
`0.422–6.891`. Focal и loss-LR refinement не входят в cost-capped кампанию.
Sqrt loss принимается только при IID-приросте строго больше `0.002`.

### 5. Подтверждение на seed 17

Seed 42 переиспользуется из поиска. Финальный tuned-BCE всегда один раз
запускается на seed 17. Если `weight_decay=0.05` или sqrt loss выиграл tuned BCE
на seed 42 строго больше чем на `0.002`, лучший такой challenger также
запускается на seed 17 вместе с matched tuned-BCE.

Challenger принимается только если его delta положительна на обоих seed 42 и
17, а средняя IID delta не меньше `0.002`. Иначе финалом остаётся plain BCE.
Seed 2026 и seed ensemble не используются.

## Статистика и правила остановки

Primary metric — category-macro AP на IID. Hard не участвует в выборе. Для
совпадающих IID-пар используются component-level paired permutation test и
paired bootstrap CI. Внутри каждой стадии IID p-values корректируются Holm по
полной запланированной семье. Practical tie margin на всех координатах —
`0.002`.

Запуск считается валидным только после проверки checkpoint SHA, source parquet
SHA, точных counts, augmented-train identity, отсутствия IID/hard leakage,
resolved config, run identity, completion marker и успешного idempotent upsert
в `sft_exps`. При non-finite loss или исчерпании memory fallback кампания
останавливается, а точка не считается результатом.

## Бюджет и runtime

| Стадия | Максимум уникальных kernels |
| --- | ---: |
| LR-линия с boundary | 4 |
| Новые epoch-точки | 2 |
| Regularization sanity | 1 |
| Sqrt loss | 1 |
| Seed-17 confirmation | 2 |
| **Жёсткий максимум** | **10** |

Типичный бюджет — 6–7 kernels. До первого baseline плановая оценка одной эпохи
составляет 2–4 часа; типичная кампания — 20–35 часов, максимальная — 45–60
часов с очередями и условными многоэпоховыми точками. Все kernels запускаются
последовательно.

Если baseline занимает строго больше четырёх часов, regularization и loss
стадии пропускаются. Остаются LR-линия, epoch 2 и seed 17. Для каждого kernel
действует внутренний soft limit 9 часов. ODS, Docker submission и повторный
runtime benchmark не входят в эту кампанию.

## Kaggle и отчётность

Пользователь разрешил private Kaggle mutations для этой задачи, однако будущий
launcher всё равно обязан выполнять локальный dry-run перед каждым submit,
запрещать `force`, retry-fanout и параллельные kernels. Каждый experiment должен
иметь уникальные slug, run ID и директорию артефактов.

Каждый завершённый notebook должен сохранить checkpoint, resolved config,
training report, IID/hard predictions, логи, baseline comparison, Sheets sync
receipt и completion marker в `/kaggle/working`. В `sft_exps` BGE baseline и
все кандидаты образуют новую внутреннюю семью; MiniLM baseline используется
только как исторический контекст без paired delta.
