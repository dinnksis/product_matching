# Финальное полное обучение BGE и экспорт весов

`bge_2ep_final_fulltrain_export_v1` — отдельный one-shot workflow для
deployment-модели. Он не изменяет и не переиспользует slug-и замороженной
абляционной кампании.

## Зафиксированный рецепт

- начальный checkpoint: `model/pretrain_bge_2ep`, exact model SHA-256
  `c21ccfcd5de310ca0328620bf8ba09e838dbe3f6394be656bd7fec16ad8377d1`;
- две эпохи, LR `2e-5`, cosine schedule, warmup `0.05`, weight decay `0.01`;
- plain BCE, seed 42, max length 384;
- 2×T4, batch 8/GPU, gradient accumulation 12, effective batch 192,
  FP16, SDPA и gradient checkpointing;
- canonical recipe SHA-256
  `d46be9217a43396ecc8c594fc1864ee93761d288c30e5a40041adbb28bd7adfe`.

Полный train восстанавливается из frozen `human/split_assignments.parquet` в
порядке `human_row_id=0..365653`. Все прежние train/IID/hard/OOD части входят в
обучение: 365 654 пары, 93 890 positive, withheld=0. Каждая часть независимо
сверяется с соответствующим frozen parquet и manifest. `label_source` сохраняет
происхождение строки: `human_train`, `human_iid`, `human_hard` или
`human_former_ood`.

Из-за полного train честного holdout нет. Notebook не создаёт validation
loaders или prediction-файлы и не заявляет качество. Для обязательного журнала
он записывает IID/hard/OOD как `evaluated=false`, `status=unavailable...` и
метрики `-1`, затем использует общий Google Sheets logger с
`experiment_group=sft`.

## Защита запуска

Launcher допускает ровно один push на identity и не имеет resubmit/force пути.
Он никогда не регенерирует notebook: launch читает уже проверенный файл через
no-follow stable-read и требует его exact byte SHA-256. Генератор запускается
отдельно только до review/freeze.
До push он проверяет:

- локальную identity notebook и staged private 2×T4 metadata;
- включённый Kaggle internet для pinned dependency bootstrap;
- отсутствие такого kernel двумя независимыми чтениями;
- remote validation Dataset: exact ready status и version 3; checkpoint Dataset:
  exact ready status и version 1;
  privacy, file set и скачанный manifest SHA-256;
- credential Dataset: exact ref, ready status, privacy и единственный файл
  `google-service-account.json`, не скачивая его содержимое;
- повторную неизменность staging и финальный status/version всех трёх Dataset.

Dry-run не обращается к Kaggle:

```bash
env PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/run_bge_2ep_final_fulltrain.py --dry-run
```

После независимого GO one-shot запуск выполняется явно:

```bash
env PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/run_bge_2ep_final_fulltrain.py --execute --no-wait
```

Продолжение мониторинга никогда не делает новый push:

```bash
env PYTHONPATH="$PWD:$PWD/scripts" .venv/bin/python \
  scripts/run_bge_2ep_final_fulltrain.py --monitor
```

Успешный output содержит deployable Hugging Face model/tokenizer, offline-load
и finite-forward smoke, metric-free training summary, audit reports и строгий
SHA-256 tree manifest. Локальный downloader повторно проверяет exact topology
(включая автоматически скачиваемый Kaggle kernel log),
обычные изолированные файлы без symlink/hardlink, стабильность inode/размера/
mtime/ctime при чтении, model SHA и completion identity.
