# Qwen3-Reranker-0.6B zero-shot vLLM submit

Экспериментальный автономный submit для проверки end-to-end времени на одной
H100 80 GB. Модель не обучалась на данных соревнования. Все пары оцениваются
`Qwen/Qwen3-Reranker-0.6B` через pooling/score runner vLLM 0.14.0.

## Runtime

- Docker image: `vllm/vllm-openai:v0.14.0`.
- Offline checkpoint: `models/Qwen3-Reranker-0.6B/`.
- PyArrow поставляется уже распакованным в `vendor/pyarrow_runtime`, поскольку
  он не является гарантированной зависимостью официального vLLM image. Это
  убирает установку пакета из измеряемого времени запуска.
- Один GPU: для модели 0,6B tensor parallel только добавил бы коммуникационные
  расходы.
- BF16, FlashAttention, eager execution, prefix caching.
- Контекст — 256 токенов; каждая карточка предварительно ограничена 180
  символами, а vLLM выполняет дополнительное безопасное token truncation.
- До 1024 последовательностей и 131 072 токенов в continuous batch.

Параметры можно менять переменными `PM_MAX_ITEM_CHARS`, `PM_MAX_MODEL_LEN`,
`PM_MAX_NUM_SEQS`, `PM_MAX_NUM_BATCHED_TOKENS` и `PM_SCORE_CHUNK_SIZE`.

Минимальная средняя скорость всего запуска для Public и Private — примерно 319
и 353 пары/с соответственно. Поскольку часть лимита занимает startup,
inference-only скорость должна быть выше. Это намеренно короткая
speed-конфигурация; фактическую скорость определит только запуск на H100
проверяющей системы.

## Контракт

```bash
bash run.sh \
  --items_path /data/items.parquet \
  --matches_path /data/matches.parquet \
  --output_path /output/submit.csv
```

Для локальной проверки чтения данных и формата без CUDA/vLLM:

```bash
bash run.sh \
  --items_path ../../data/items_human.parquet \
  --matches_path ../../data/matches.parquet \
  --output_path /tmp/qwen3-submit-smoke.csv \
  --skip-model
```

В output всегда записываются все пары в исходном порядке и ровно три столбца:
`id1,id2,predict`.

## Воспроизведение архива

Веса, распакованный Linux wheel и готовый ZIP исключены из Git из-за размера.
Скрипт сборки скачивает закреплённую ревизию модели и PyArrow, проверяет SHA-256
и создаёт автономный архив:

```bash
make submit-build
```

Если локальные assets уже существуют и нужно только пересобрать ZIP без сети:

```bash
make submit-archive
```

Результат: `submits/qwen3-reranker-vllm-zero-shot.zip`. Корнем архива являются
сразу `metadata.json`, `run.py` и остальные файлы submit, без внешней папки.

## Важное ограничение

Это speed smoke test, а не качественный финальный submit. На локальной
10-тысячной выборке исходная zero-shot модель дала macro AP около 0,336 и
сильно завышенные scores. Следующая версия должна использовать обученный
checkpoint и, вероятно, ансамбль с лёгкими признаками.
