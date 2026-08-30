# Эксперименты с reranker и cross-encoder моделями

Документ фиксирует эксперименты в порядке их проведения. Результаты локальной
валидации и результаты на сайте соревнования нельзя напрямую сравнивать: в
первом случае используются открытые human-labelled данные, во втором — закрытый
тест организаторов.

## 1. Qwen3-Reranker-0.6B, zero-shot по названиям

Первым проверялся `Qwen/Qwen3-Reranker-0.6B` без дообучения. В модель подавалась
пара названий товаров, атрибуты не использовались.

На component-disjoint validation получено:

- 10 000 пар: macro AP `0.422389`, скорость около `45.1 пары/с` на Tesla T4;
- вся validation: macro AP `0.428163`, inference `1297.5 с`, около
  `42.3 пары/с` на Tesla T4.

Такой throughput заведомо не соответствовал ограничениям соревнования. Обучение
LoRA было начато, но остановлено: улучшение качества не решало проблему скорости
инференса. Возвращаться к этому варианту имеет смысл только при появлении
принципиально более быстрого способа исполнения.

Где лежит:

- benchmark: `scripts/benchmark_qwen_names.py`;
- общая реализация: `src/qwen_reranker.py`;
- сохранённый отчёт 10k: `predicted/qwen_names_zero_shot_10k.report.json`;
- скрипты обучения: `scripts/train_qwen_names.py` и
  `scripts/train_qwen_names_full.py`.

## 2. Jina reranker v2, zero-shot по названиям

В соревнование была отправлена модель
`jinaai/jina-reranker-v2-base-multilingual` (около 278M параметров). Это
multilingual cross-encoder/reranker: каждую пару названий он читает совместно и
возвращает один logit. Дообучение и атрибуты не использовались.

Подтверждённый результат на сайте соревнования:

- решение прошло Docker-стадию и ограничения времени;
- полное выполнение заняло примерно 6–7 минут;
- leaderboard score: около `0.25`.

Это важный runtime-baseline: архитектура достаточно быстрая для H100, но
zero-shot relevance score плохо переносится на задачу идентичности товаров.
Вероятная точка роста — supervised fine-tuning на human-labelled парах и
короткая, стабильная сериализация наиболее полезных атрибутов.

Где лежит:

- submission runner и metadata:
  `submits/jina-reranker-v2-zero-shot/`;
- готовый архив: `submits/jina-reranker-v2-zero-shot.zip`;
- сборщик архива: `scripts/build_jina_submit.py`;
- Docker runtime: `docker/jina-reranker-runtime/Dockerfile`.

Примечание по лицензии: model card указывает `CC-BY-NC-4.0`. Перед дальнейшим
использованием и особенно дообучением нужно отдельно подтвердить соответствие
лицензии правилам соревнования.

## 3. Multilingual MiniLM cross-encoder

Подготовлен эксперимент с
`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` — компактной multilingual
cross-encoder моделью примерно на 0.1B параметров. Она архитектурно близка по
идее к Jina v2: пара текстов обрабатывается совместно классификатором, но сама
базовая модель меньше и потенциально быстрее.

Текущая конфигурация предполагает full fine-tuning на human labels:

- 1 эпоха;
- training batch 96, validation batch 192;
- max length 256;
- component-disjoint split;
- симметричная validation для двух порядков пары.

Модель пока не отправлялась в соревнование, а подтверждённый validation score в
этом репозитории не сохранён. Это направление улучшает другой участник, поэтому
перед изменением его файлов нужно сверять актуальное состояние Git.

Где лежит:

- конфигурация: `configs/cross_encoder_minilm.json`;
- обучение: `scripts/train_cross_encoder.py` и
  `src/cross_encoder_training.py`;
- генератор notebook: `scripts/create_cross_encoder_training_notebook.py`;
- инструкция: `docs/cross-encoder-training.md`.

## Какие модели рассмотреть дальше

Приоритет — не искать просто более крупный reranker, а проверить компактные
multilingual cross-encoders единым протоколом: zero-shot на небольшой validation,
замер H100 throughput, затем одно supervised обучение только для вариантов,
которые проходят лимит.

Кандидаты:

1. `Alibaba-NLP/gte-multilingual-reranker-base` — около 304M параметров,
   multilingual, Apache-2.0. По размеру близок к прошедшему по времени Jina v2,
   поэтому это наиболее логичный следующий zero-shot benchmark.
2. `BAAI/bge-reranker-v2-m3` — multilingual reranker, Apache-2.0, примерно
   568M параметров. Потенциально сильнее, но почти вдвое тяжелее Jina v2; до
   обучения сначала нужен жёсткий speed benchmark.
3. `jinaai/jina-reranker-v3` — 0.6B multilingual Qwen3-based reranker с другой
   late-interaction схемой. Он интересен архитектурно, но крупнее Jina v2 и имеет
   `CC-BY-NC-4.0`; сначала проверить лицензию и скорость, не начинать с обучения.

Официальные model cards: [GTE multilingual reranker](https://huggingface.co/Alibaba-NLP/gte-multilingual-reranker-base),
[BGE reranker v2 m3](https://huggingface.co/BAAI/bge-reranker-v2-m3),
[Jina reranker v3](https://huggingface.co/jinaai/jina-reranker-v3),
[multilingual MiniLM](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).

## Единый протокол следующих сравнений

Чтобы не повторять дорогие эксперименты вслепую, для каждого нового reranker
фиксировать:

1. точную ревизию модели и лицензию;
2. zero-shot macro AP на одном и том же component-disjoint split;
3. pairs/s, model-loading time, max GPU memory и оценку полного test runtime;
4. только после прохождения speed gate — fine-tuning на human labels;
5. отдельно names-only и name + ограниченные структурированные атрибуты;
6. перед submission — запуск именно submission runner на validation, а не только
   training pipeline.

## 4. BGE reranker v2 m3, zero-shot runtime на 2×T4

`BAAI/bge-reranker-v2-m3` (revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, 568M, Apache-2.0) проверен без
обучения на полном component-disjoint human train seed 42: `310 767` пар
названий. Оба текста заранее ограничены 94 model-токенами, общий
`max_length=192`; truncation затронул только `0.0597%` отдельных названий.
Каждый backend использовал две независимые FP16-реплики на Tesla T4 и
length-sorted shard на каждой GPU.

| backend | batch/concurrency на GPU | чистый inference | throughput 2×T4 | backend wall time | active GPU util |
|---|---:|---:|---:|---:|---:|
| sentence-transformers 5.4.1 | batch 128 | 380.24 с | 817.28 пар/с | 521.16 с | 98.52% |
| transformers 4.57.6 | batch 128 | 394.57 с | 787.62 пар/с | 534.99 с | 93.87% |
| vLLM 0.19.1 | 1024 seq / 65 536 tokens | 911.37 с | 340.99 пар/с | 1005.96 с | 99.73% |

Крупнейший batch оказался не самым быстрым: автоматический перебор
`128, 256, 384, 512, 768` выбрал `128` на обеих T4. На этих коротких парах
`sentence-transformers` быстрее прямого Transformers примерно на `3.8%`, а
vLLM при почти полной утилизации медленнее примерно в `2.40 раза`. Две
vLLM-реплики выгоднее tensor parallel для этой модели, но continuous batching
всё равно не окупает overhead encoder-only pooling runner.

Zero-shot качество на train (не frozen validation и не leaderboard): macro AP
`0.4712`, overall AP `0.5020`. Scores совпадают между реализациями: Pearson для
Sentence Transformers logits против Transformers `0.99999993`; для vLLM
probability против sigmoid Transformers logit `0.99999923`.

Артефакты и воспроизводимость:

- generator: `scripts/create_bge_reranker_runtime_notebook.py`;
- launcher: `scripts/run_bge_reranker_runtime_kaggle.py`;
- notebook: `notebooks/bge_reranker_v2_m3_runtime_2xt4.ipynb`;
- результаты: `artifacts/kaggle/product-matching-bge-reranker-v2-m3-runtime/`;
- полный notebook pipeline: `2236.74 с`, включая installs, download, подготовку
  данных, три backend-прогона, метрики и сохранение outputs.

## 5. MiniLM supervised search

Полная MiniLM-линия проверила learning rate, число эпох, effective batch,
warmup, weight decay, label smoothing, classifier dropout и loss-функции.
Итоговый anchor — `minilm5_sft_e3_lr8e5_v1`: plain BCE, 3 эпохи, LR `8e-5`,
IID macro AP `0.808502`, hard `0.423286`, OOD `0.647977`. Dropout `0` и `0.2`
не улучшили anchor; дополнительные regularization coordinates также не прошли
practical IID gate.

Полный протокол, lock/receipt chain и paired comparisons находятся в
`reports/minilm_5ep_sft_hparam_search_v1/` и
[`docs/minilm-5ep-sft-hparam-search.md`](minilm-5ep-sft-hparam-search.md).

## 6. BGE supervised controlled campaign

Для BGE human train был объединён с former OOD: 347 840 пар, 89 291 positive,
20 категорий. Former OOD после этого не является метрикой; IID и hard содержат
18 категорий и остаются component-disjoint относительно train.

| вариант | IID macro AP | hard macro AP |
|---|---:|---:|
| 1 ep, LR `1e-5`, BCE | 0.813638 | 0.411162 |
| 1 ep, LR `2e-5`, BCE | 0.818291 | 0.414717 |
| 1 ep, LR `4e-5`, BCE | 0.815148 | 0.409547 |
| 2 ep, LR `2e-5`, BCE | **0.823461** | **0.437775** |
| 2 ep, sqrt category×class BCE | 0.822150 | 0.431759 |

Таким образом, e2 plain BCE выбран как controlled winner. Loss reweighting
ухудшил обе доступные метрики, а OOD везде записан как `-1` и не вычисляется.
Exact reports лежат в `reports/bge_2ep_sft_candidate_v1/` и
`reports/bge_2ep_sft_loss_confirmation_v1/`.

## 7. Финальный трёхэпоховый BGE на H100

Отдельный server run использовал другой, предоставленный checkpoint той же
архитектуры XLM-R/BGE: plain BCE, LR `2e-5`, `max_length=384`, effective batch
192, BF16/SDPA, 3 эпохи на одной H100. Symmetric validation дала IID macro AP
`0.824975` и hard `0.461148`.

Финальный submission использует один порядок пары, ставя более длинную карточку
первой: IID `0.823629`, hard `0.462631`. Это снижает стоимость inference вдвое
относительно symmetric evaluation. Веса имеют SHA-256
`d7e899ea3cd305db970aa6f3466eb71a138ad418c74b8b6ac730d1828c4a4ab8` и
опубликованы как private Kaggle Dataset
[`product-matching-bge-3ep-h100-oodtrain`](https://www.kaggle.com/datasets/alexproger23/product-matching-bge-3ep-h100-oodtrain).

Submission bundle: `submits/bge-reranker-v2-m3-3ep-h100/`. Поскольку стартовые
checkpoint bytes отличаются от controlled BGE campaign, этот запуск нельзя
использовать как чистую абляцию эффекта третьей эпохи.

Общая интерпретация, data-generation попытки и ссылки на полный журнал:
[`docs/final-results-and-experiment-history.md`](final-results-and-experiment-history.md).
