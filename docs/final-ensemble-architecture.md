# Итоговая архитектура ансамбля и selective routing

Состояние зафиксировано на 28 августа 2026 года. Документ разделяет два
несовместимых протокола: исследовательский `384 + symmetric AB/BA` и текущий
production inference. Числа между ними нельзя сравнивать как результаты одного
и того же pipeline.

## Краткий вывод

- Обязательный backbone — `BAAI/bge-reranker-v2-m3`, BGE считается на 100% пар.
- Единственная архитектура, для которой уже подтверждено прохождение актуального
  competition runtime, — полный BGE + полный MiniLM в реализации
  SentenceTransformers, `max_length=192`, один проход `A→B`, FP16, SDPA,
  `batch_size=1024`, mean normalized rank raw logits. Точное wall time в
  репозитории не сохранено; факт прохождения подтверждён фактическим запуском.
- Если нужен запас compute, наиболее обоснованный вариант без RuModern — BGE 100%
  + MiniLM top 40% по classification benefit-router, на routed парах
  `0.5 * p_BGE + 0.5 * p_MiniLM`. В production-aligned one-way проверке он дал
  mean macro AP `0.601513` против `0.600791` у полного one-way BGE+MiniLM, но на
  честном train OOF отставал от полного ансамбля на `0.001639`. Поэтому это
  runtime-кандидат, а не доказанно более качественный ансамбль.
- Лучший проверенный трёхмодельный selective pipeline — BGE 100% + MiniLM 40% +
  RuModernBERT 5%, иерархически. В one-way проверке mean macro AP `0.602048`.
  Он пока **не подтверждён по runtime**: последние H100-запуски доходили до
  загрузки RuModernBERT, но падали до первого законченного RuModern forward.
- Полный one-way ансамбль трёх моделей имеет mean macro AP `0.602145`, лишь на
  `0.000097` выше selective 40%/5%, при `3.0` model-forward на пару вместо
  среднего `1.45`. Кроме того, его Hard AP заметно хуже. Его runtime также не
  подтверждён.
- RuModern 10% не выбран: в старом OOF-safe compute allocation переход 5%→10%
  не улучшал frontier и ухудшал Hard. Пока нет нового evidence, 5% — разумный
  предел.

## Модели и исходный quality anchor

Все cross-encoder predictions в первом цикле получены с сериализацией
`S2_VALUES_ONLY`, `max_length=384` и symmetric probability inference
`mean(sigmoid(logit_AB), sigmoid(logit_BA))`. IID, Hard и OOD использовались
только для итоговой оценки; параметры последующих routers выбирались по human
train OOF.

| Модель | IID macro AP | Hard | OOD | Mean |
| --- | ---: | ---: | ---: | ---: |
| BGE | 0.782222 | **0.375975** | 0.641270 | 0.599822 |
| MiniLM | **0.782508** | 0.368075 | 0.616523 | 0.589035 |
| RuModernBERT | 0.773296 | 0.350399 | 0.633022 | 0.585572 |
| GTE | 0.761850 | 0.348445 | 0.615033 | 0.575109 |

GTE оказался самым слабым одиночным экспертом и ухудшил all-4 относительно
лучшей тройки, поэтому исключён из дальнейшей архитектуры.

| Старый symmetric ансамбль | IID | Hard | OOD | Mean |
| --- | ---: | ---: | ---: | ---: |
| BGE + MiniLM, mean rank | 0.792348 | 0.374699 | 0.640976 | 0.602674 |
| BGE + MiniLM + RuModern, mean probability | **0.796635** | 0.366133 | **0.648387** | **0.603718** |
| All-4 | — | — | — | 0.601127 |

Главный вывод одиночных моделей: BGE — strongest backbone; MiniLM слабее сам по
себе, но даёт дешёвое дополнительное ранжирование; RuModern в среднем ещё слабее,
но добавляет OOD-diversity. Поэтому specialists полезны выборочно, а не потому,
что их одиночный AP выше BGE.

## Что проверили в трёх routing-экспериментах

### 1. Pairwise corrections, slices и oracle headroom

Для BGE против MiniLM и RuModern проверены binary correctness при фиксированном
threshold, per-example logloss и absolute error, затем macro AP. Анализ включал
category, label, BGE score/uncertainty, title, brand/model/code, typed numeric
conflicts, attributes, missingness и lengths.

Единственный slice, прошедший строгую устойчивость одновременно на IID и Hard:
MiniLM при `numeric_volume_state=match` (IID ΔAP `+0.002695`, Hard
`+0.023283`). Этого недостаточно для набора жёстких production-правил.

Oracle показал большой теоретический headroom. Например, oracle best expert при
5% coverage давал `+0.053836 / +0.029806 / +0.090521` AP на IID/Hard/OOD. Это
не production-результат: oracle видит label. Вывод только один — selective
ensemble потенциально полезен, если selector научится находить правильные пары.

### 2. Простые deterministic routers

Проверены BGE uncertainty (`abs(p-0.5)` и entropy), score-region и conflict
routing. `abs(p-0.5)` и entropy дали одинаковый порядок. До появления совместимых
train OOF эти правила не принимались как production-tuned; они были baselines.
После появления OOF uncertainty оставался сильным простым baseline, но learned
classification router выиграл у лучшего simple baseline до `0.001023` OOF macro
AP. Десятки ручных slice-правил не создавались.

### 3. Learned benefit routers

Router предсказывает не product label, а ожидаемую пользу специалиста. Для пары
с label `y`:

`benefit = logloss(y, p_current) - logloss(y, p_specialist)`.

Проверены:

- regression: напрямую предсказывать `benefit`;
- classification: `help=1`, если `benefit > 0.02`.

Classification стабильно оказался лучше regression и выбран. Обучение использует
только neural OOF на human train. Пять router-fold являются component-disjoint:
один product id не попадает в train и validation разных folds. IID/Hard/OOD
загружаются лишь после фиксации route method, coverage и blend weight.

Первый полный router использовал CatBoost с 134 признаками, из них 7
категориальных. Параметры: 400 деревьев, depth 7, learning rate 0.05,
`l2_leaf_reg=7`, `random_strength=0.5`, Bayesian bootstrap,
`bagging_temperature=0.5`, `border_count=128`, seed 2026, 20 CPU threads. Были
обучены classifier и regressor отдельно для MiniLM и RuModern. Полный цикл занял
3260 секунд (около 54 минут) локально; это training time, не submission runtime.

Полные 134 признака:

- title exact/fuzzy/token/number overlap и длины;
- counts, overlap, exact/conflict, missingness и value similarity attributes;
- category;
- brand, model/code, color, material и title-code match/conflict/missingness;
- typed numeric families: size, RAM/storage, volume, weight, pack count, power,
  dimensions, optical, model number, voltage, frequency, capacity;
- conflict signatures и matching regime;
- BGE probability/logit/uncertainty/entropy, AB/BA disagreement и token lengths.

Специалистские scores и embeddings запрещены до решения соответствующего router.
Именно full feature extraction оказался production-проблемой: 118-признаковый
C1A-проход занял около 90 секунд на 59k пар, то есть примерно 7 минут на private.
Сам `CatBoost.predict` быстрый; дорогой частью были parsing и Python feature
engineering. Поэтому полный router оставлен для анализа, а в submit перенесён
компактный CatBoost.

## Точные правила compact CatBoost

Это не набор `if/else`. CatBoost выдаёт priority score; затем берутся ровно top
`floor(N * coverage)` пар. При равенстве priority порядок стабилизируется по
`id1,id2`.

### MiniLM router

Production-compact CatBoostClassifier: 300 деревьев, depth 7, learning rate
0.06, остальные параметры как выше. Target — benefit MiniLM относительно BGE,
`help = benefit > 0.02`.

Проверены два feature set:

- `score_category`, 7 признаков: `category`, `bge_probability`, `bge_logit`,
  `bge_abs_from_half`, `bge_uncertainty`, `bge_entropy`, `bge_raw_logit`;
- `score_title`, 15 признаков: те же 7 плюс `title_exact`, RapidFuzz ratio и
  token-sort ratio, token Jaccard, length ratio/delta, number overlap/Jaccard.

На 40% OOF разница `score_title` против `score_category` всего `+0.0000715` AP.
Поэтому для трёхмодельного pipeline выбран `score_category`: 7 признаков, без
RapidFuzz и без attribute parsing. Для отдельного MiniLM40 submit собран
`score_title`, потому что он дал лучший one-way validation результат.

Финальный шаг MiniLM:

1. посчитать BGE на всех парах;
2. CatBoost priority → top 40% (или 60% в quality-probe);
3. запустить MiniLM только на выбранных парах;
4. на них использовать `0.5 * p_BGE + 0.5 * p_MiniLM`, на остальных оставить
   `p_BGE`.

### Sequential RuModern router

RuModern router обучен относительно уже доступного current score, а не только
относительно BGE:

`current = 0.5 * p_BGE + 0.5 * p_MiniLM`;

`benefit_Ru = logloss(y, current) - logloss(y, p_RuModern)`.

Target classifier тот же: `benefit_Ru > 0.02`. Он применяется только внутри
MiniLM-routed subset и использует 14 дешёвых признаков:

- 7 признаков `score_category`;
- `minilm_probability`, `minilm_logit`;
- absolute и signed BGE/MiniLM disagreement;
- mean, min и max двух probabilities.

RuModern score до решения router не используется. Из routed MiniLM subset
выбирается top 5% от всех пар. На них:

`final = 0.5 * current + 0.5 * p_RuModern`.

Вес RuModern 0.5 выбран по OOF (`0.788532`), против `0.788501` для веса 0.25 и
`0.786606` для replace. Полный replace отвергнут.

## Coverage и macro AP

### Старый 384 symmetric MiniLM sweep

Таблица отвечает на вопрос о quality headroom, но не на вопрос о runtime текущего
one-way submit.

| MiniLM coverage | IID | Hard | OOD | Mean | Δ mean vs MiniLM100 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 0.785554 | 0.378029 | 0.639541 | 0.601041 | -0.001510 |
| 10% | 0.787380 | 0.377306 | 0.640971 | 0.601886 | -0.000665 |
| 15% | 0.787547 | 0.377739 | 0.642038 | 0.602441 | -0.000110 |
| 20% | 0.788477 | 0.377364 | 0.641849 | 0.602563 | +0.000012 |
| 30% | 0.790447 | 0.376574 | 0.641362 | 0.602795 | +0.000244 |
| 40% | 0.790349 | 0.377149 | 0.641795 | **0.603098** | **+0.000547** |
| 50% | 0.791406 | 0.376180 | 0.641388 | 0.602991 | +0.000441 |
| 70% | 0.792591 | 0.374380 | 0.641720 | 0.602897 | +0.000346 |
| 100% | 0.792857 | 0.373480 | 0.641315 | 0.602551 | 0 |

40% — хороший reserved-compute point: лучший mean в sweep и хороший Hard, хотя
по честному OOF он всё же уступает full MiniLM. 60% — более консервативный point:
one-way OOF сохраняет почти весь full-ensemble gain, но экономит только 40%
MiniLM inference.

### Production-aligned one-way quality, max length 384

Здесь один AB forward, raw logits переводятся sigmoid в probabilities, а веса
фиксируются по OOF.

| Pipeline | Mini | Ru | IID | Hard | OOD | Mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE | 0% | 0% | 0.780072 | 0.376409 | 0.636524 | 0.597668 |
| BGE + full MiniLM, 50/50 | 100% | 0% | 0.791406 | 0.372259 | 0.638708 | 0.600791 |
| Uncertainty MiniLM | 20% | 0% | 0.785524 | **0.377501** | 0.638172 | 0.600399 |
| Learned MiniLM, `score_title` | 40% | 0% | 0.788195 | 0.377088 | 0.639255 | 0.601513 |
| Learned MiniLM, `score_title` | 60% | 0% | 0.790052 | 0.375782 | 0.638387 | 0.601407 |
| Learned MiniLM, `score_category` | 40% | 0% | 0.788322 | 0.377071 | 0.638521 | 0.601305 |
| Hierarchical compact | 40% | 5% | 0.789238 | 0.376792 | 0.640114 | 0.602048 |
| Full equal triple | 100% | 100% | **0.795607** | 0.365598 | **0.645230** | **0.602145** |

Почему 40%/5% выглядит разумно:

- против BGE mean gain `+0.004380`;
- против полного BGE+MiniLM mean gain `+0.001257`;
- RuModern5 добавляет к compact Mini40 около `+0.000743` mean, главным образом
  за счёт OOD (`+0.001593`), почти не меняя IID и Hard;
- полный triple добавляет к 40/5 только `+0.000097` mean, но ухудшает Hard на
  `0.011194` и требует более чем вдвое больше model forwards.

### Почему не RuModern 10%

В старом OOF-safe hierarchical grid:

| Mini / Ru | IID | Hard | OOD | Mean |
| --- | ---: | ---: | ---: | ---: |
| 40% / 5% | 0.792534 | **0.376139** | 0.643866 | **0.604180** |
| 40% / 10% | 0.792380 | 0.375216 | **0.643988** | 0.603861 |
| 60% / 5% | 0.794034 | **0.375256** | 0.643439 | **0.604243** |
| 60% / 10% | **0.794140** | 0.374533 | **0.643997** | 0.604223 |

Дополнительные 5% RuModern покупают небольшой OOD/IID прирост, но теряют Hard и
не улучшают mean. Поэтому 5% остаётся Pareto point; 10% можно возвращать только
после изменения checkpoint/router или нового one-way OOF evidence.

## Inference engineering и runtime

### Что проверялось

На раннем T4 benchmark при старом `384 + symmetric` лучшим был native
Transformers/PyTorch FP16 + SDPA + pretokenization + exact-length bucketing +
dynamic padding:

| Модель | Safe T4 batch | 5k pairs | Throughput |
| --- | ---: | ---: | ---: |
| BGE | 32 | 84.94 s | 58.87 pairs/s |
| MiniLM | 192 | 14.46 s | 345.71 pairs/s |
| RuModernBERT | 192 | 47.29 s | 105.74 pairs/s |

На том T4 и старом runner SentenceTransformers был медленнее native, а vLLM —
существенно медленнее. Этот вывод не перенесён автоматически на H100: поздняя
SentenceTransformers реализация использовала другой batch, одну direction и
иной pipeline. В отдельном H100 сравнении пользователя BGE показал примерно
787.62 pairs/s через Transformers и 817.28 pairs/s через SentenceTransformers
при batch 128; финальный batch был 1024.

### Что дало финальное ускорение

- `sentence_transformers.CrossEncoder` 5.1.2;
- `torch.float16`;
- `attn_implementation="sdpa"` — не FlashAttention2;
- H100 `batch_size=1024`;
- `torch.inference_mode()`;
- TF32 разрешён для CUDA matmul/cuDNN, matmul precision `high`;
- stable sort по суммарной character length для уменьшения padding;
- Rust/Rayon tokenizer parallelism, 20 CPU threads;
- `activation_fn=torch.nn.Identity()` для возврата raw logits;
- последовательная загрузка моделей с освобождением CUDA между ними;
- offline weights и writable HF/Triton/TorchInductor caches рядом с output;
- главное ускорение — один AB forward вместо symmetric AB+BA.

Symmetric inference означал не «модель загружается дважды», а два directed
forward на каждую пару. Для BGE100+MiniLM40 это было `2.0 + 0.8 = 2.8` forward
на пару; one-way — `1.0 + 0.4 = 1.4`. Для 40/5 с RuModern это `1.45`, для 60/5
`1.65`, для full triple `3.0`. Поэтому symmetric проход пришлось убрать.

### Зафиксированные submit-протоколы

| Submit | Length | Direction | Score | Runtime status |
| --- | ---: | --- | --- | --- |
| Full BGE+MiniLM ST | 192 | AB | 50/50 normalized rank raw logits | **Прошёл актуальный лимит** |
| MiniLM40 compact router | 384 | AB | probabilities, routed 50/50 | Собран; отдельный полный pass не зафиксирован |
| MiniLM40 + Ru5 | 384 | AB | hierarchical probability blends | Не подтверждён: RuModern startup failure |
| MiniLM60 + Ru5 | 384 | AB | hierarchical probability blends | Не подтверждён: RuModern startup failure |
| Full triple | 384 | AB | equal probability blend | Не подтверждён: RuModern startup failure |

У первого полного ансамбля length 192 и rank-logit aggregation отличаются от
384-token OOF quality-протокола. Его runtime доказан, но точное IID/Hard/OOD
качество именно этого submit нельзя подставлять из старой таблицы.

В последних 384-token RuModern archives сначала был read-only error для
`/root/.triton`, затем ModernBERT автоматически включал `torch.compile` и падал
из-за отсутствия C compiler. Builder теперь выставляет одновременно
`compile_model=false`, `reference_compile=false`, runner задаёт
`TORCHDYNAMO_DISABLE=1`, а compiler caches направлены в writable directory.
Docker-конфиг проверен, но успешный полный competition run после фикса ещё не
получен. Поэтому нельзя утверждать, что 40/5, 60/5 или full triple укладываются.

## Рекомендуемые архитектуры

### 1. Подтверждённая production anchor

Полный BGE + полный MiniLM, 192 tokens, one-way SentenceTransformers/SDPA,
batch 1024, mean normalized rank raw logits. Это единственный безопасный выбор,
если прямо сейчас важнее гарантированно пройти runtime.

### 2. Reserved-compute candidate

BGE100 + compact CatBoost → MiniLM40. Если нужен RuModern и его runtime наконец
проходит, добавить sequential RuModern top 5% внутри MiniLM subset. Использовать
`score_category` для минимального CPU overhead и 50/50 probability blends.

### 3. Quality probe, пока не production

BGE100 + MiniLM60 + RuModern5. В старом symmetric grid это лучший mean point,
но для текущего one-way compact pipeline нет отдельной полной quality таблицы и
нет успешного runtime. Он нужен как следующий probe, а не как уже выбранный
финал.

Full triple можно использовать как quality diagnostic. При текущих результатах
он не оправдывает `3.0` forwards/pair: выигрыш против 40/5 слишком мал, а Hard
хуже. Learned router сохраняет смысл именно как способ высвободить compute, а не
как обязательный компонент прошедшего full BGE+MiniLM anchor.

## Артефакты и воспроизводимость

- одиночные модели и ensembles: `docs/ensemble_cross.md`,
  `reports/architecture_ensemble_v1/`;
- oracle/slices: `reports/selective_specialist_oracle_v1/`;
- deterministic routing: `reports/deterministic_specialist_routing_v1/`;
- full benefit-router: `scripts/train_benefit_routers.py`,
  `configs/benefit_router_all_experts.json`;
- MiniLM coverage: `reports/minilm_coverage_sweep_v1/`;
- старый three-model allocation:
  `reports/compute_allocation_bge_minilm_rumodern_v1/`;
- one-way compact MiniLM: `reports/fast_oneway_benefit_router_v1/`;
- one-way compact RuModern: `reports/fast_oneway_rumodern_router_v1/`;
- builders текущих submit: `scripts/build_bge_minilm_sentence_transformers_submit.py`,
  `scripts/build_bge_minilm_fast_oneway_submit.py`,
  `scripts/build_bge_minilm_rumodern_fast_oneway_submit.py`,
  `scripts/build_bge_minilm_rumodern_fast_oneway_60_5_submit.py`,
  `scripts/build_bge_minilm_rumodern_full_oneway_submit.py`.

В Git должны храниться код, configs, manifests и небольшие CSV/JSON/Markdown
reports. OOF predictions, Parquet caches, fitted model weights, `.cbm` внутри
submission payload, Docker ZIP и credentials являются локальными артефактами и
в коммит не входят.
