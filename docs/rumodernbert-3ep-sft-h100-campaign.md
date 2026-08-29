# RuModernBERT: пять последовательных SFT-запусков на одной H100

Этот workflow обучает task-pretrained checkpoint
`model/pretrain_rumodernbert_3ep` на human-разметке. Он не использует Kaggle,
Google Sheets или ODS и рассчитан ровно на одну H100 80GB.

## Что именно запускается

Checkpoint содержит `ModernBertForSequenceClassification` с одним уже
обученным ranking-logit: 22 слоя, hidden size 768 и 149 605 633 параметра.
Каждая точка начинается заново с этого checkpoint; веса предыдущей точки не
используются, потому что cosine schedule должен быть рассчитан на полный
горизонт конкретного запуска.

Очередь всегда содержит ровно пять обучений:

1. 1 эпоха, LR `8e-5` — anchor, перенесённый из MiniLM;
2. 1 эпоха, LR `4e-5`;
3. 1 эпоха, LR `1.6e-4`;
4. 2 эпохи на LR, выбранном по IID после первых трёх запусков;
5. 3 эпохи на том же LR — обязательный трёхэпоховый finalist.

Это последовательная log-линия, а не grid. MiniLM уже показал, что
`warmup=0.05`, `weight_decay=0.01`, отсутствие label smoothing и inherited
dropout образуют широкое плато, поэтому эти координаты повторно не перебираются.
Все пять точек используют plain BCE с finite guard, `max_length=384`,
effective batch 192, symmetric IID/hard validation и seed 42.

LR выбирается только по IID macro AP. Если `8e-5` отстаёт от численного лучшего
не больше чем на `0.002`, остаётся anchor; иначе среди практического tie
выбирается меньший LR. После e1/e2/e3 рекомендуется минимальное число эпох в
пределах `0.002` от численного максимума. Hard остаётся диагностическим.

## Данные

Контроллер детерминированно объединяет:

- 306 669 строк `train_pairs.parquet`;
- 41 171 строк бывшего `ood_validation_pairs.parquet`.

Итоговый train содержит 347 840 пар, 89 291 positive и все 20 категорий.
Бывший OOD после этого не является validation: OOD parquet не передаётся
trainer, в итоговом summary его metric равен sentinel `-1`. Validation содержит
только IID (12 000 пар, 18 категорий) и hard (5 814, 18 категорий). До первого
запуска проверяются SHA-256, unordered duplicates, target, category и нулевой
item overlap train↔IID/hard.

## Перенос на сервер через SCP

Ниже `USER@HOST` и `/srv/product_matching` нужно заменить на свои значения.
Команды выполняются на локальной машине из корня репозитория после того, как на
сервере уже сделан `git clone` или `git pull` нужного коммита.

```bash
SERVER=USER@HOST
REMOTE=/srv/product_matching

ssh "$SERVER" "mkdir -p '$REMOTE/model/pretrain_rumodernbert_3ep' '$REMOTE/prepared/validation_splits_v1/human'"

scp \
  model/pretrain_rumodernbert_3ep/added_tokens.json \
  model/pretrain_rumodernbert_3ep/config.json \
  model/pretrain_rumodernbert_3ep/merges.txt \
  model/pretrain_rumodernbert_3ep/model.safetensors \
  model/pretrain_rumodernbert_3ep/special_tokens_map.json \
  model/pretrain_rumodernbert_3ep/tokenizer.json \
  model/pretrain_rumodernbert_3ep/tokenizer_config.json \
  model/pretrain_rumodernbert_3ep/vocab.json \
  "$SERVER:$REMOTE/model/pretrain_rumodernbert_3ep/"

scp \
  prepared/validation_splits_v1/human/items.parquet \
  prepared/validation_splits_v1/human/train_pairs.parquet \
  prepared/validation_splits_v1/human/ood_validation_pairs.parquet \
  prepared/validation_splits_v1/human/iid_validation_pairs.parquet \
  prepared/validation_splits_v1/human/hard_validation_pairs.parquet \
  "$SERVER:$REMOTE/prepared/validation_splits_v1/human/"
```

`optimizer.pt` из pretrain-каталога не нужен: SFT намеренно создаёт новый
AdamW и новый cosine scheduler для каждой точки. Для нестабильного соединения
те же файлы удобнее передавать `rsync -avP`, но контракт байтов остаётся тем же.

## Установка и проверки на сервере

```bash
cd /srv/product_matching
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-rumodernbert-h100.txt

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)"
```

Посмотреть план без записи файлов и без GPU:

```bash
.venv/bin/python scripts/run_rumodernbert_sft_campaign.py
```

Проверить SHA checkpoint/data без обучения:

```bash
.venv/bin/python scripts/run_rumodernbert_sft_campaign.py --dry-run
```

Подготовить combined train отдельно, также без GPU:

```bash
.venv/bin/python scripts/run_rumodernbert_sft_campaign.py --prepare-only
```

## Запуск всей очереди

Рекомендуется `tmux`, чтобы SSH disconnect не остановил процесс:

```bash
cd /srv/product_matching
tmux new -s rumodernbert-sft
CUDA_VISIBLE_DEVICES=0 scripts/run_rumodernbert_h100_queue.sh
```

Перед первым обучением controller выполняет один реальный H100 preflight:
BF16 forward/backward, gradient clipping, AdamW step со всеми 138 state tensors
и eval batch 512 при резидентных moments. Затем пять запусков идут строго
последовательно. Общий token cache переиспользуется, но каждый model/optimizer/
scheduler создаётся заново из исходного checkpoint.

Повтор той же команды безопасно пропускает только полностью проверенные
завершённые точки. Partial output не перезаписывается и не считается resume:
controller остановится с указанием директории, поскольку trainer не сохраняет
полный optimizer/scheduler state.

Артефакты находятся в:

```text
artifacts/rumodernbert_3ep_sft_oodtrain_h100_v1/
├── prepared/
├── h100_preflight.json
├── resolved_configs/
├── token_cache/
├── runs/
│   ├── e1_lr8e5/
│   ├── e1_lr4e5/
│   ├── e1_lr1p6e4/
│   ├── e2_selected_lr/
│   └── e3_selected_lr/
├── campaign_state.json
└── campaign_summary.json
```

Каждый run сохраняет model/tokenizer, `training_report.json`, IID/hard
predictions, полный `training.log` и hash-bound `run_completed.json`. Финальный
summary содержит две paired component-permutation/bootstrap семьи с Holm
correction, выбранный LR/epoch и точный путь рекомендуемого checkpoint.

Если обучение завершилось, но summary нужно пересчитать отдельно:

```bash
.venv/bin/python scripts/run_rumodernbert_sft_campaign.py --summarize
```
