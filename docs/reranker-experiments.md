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
