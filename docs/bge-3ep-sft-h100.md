# BGE: отдельный трёхэпоховый SFT на одной H100

Ветка `experiment/bge-3ep-h100` содержит один изолированный серверный запуск
`bge_3ep_sft_oodtrain_h100_v1`. Он не использует Kaggle и не продолжает
сохранённый e2 checkpoint: модель начинает заново с
`model/pretrain_bge_2ep`, а cosine scheduler строится сразу на полный горизонт
из трёх эпох.

## Замороженный рецепт

- `XLMRobertaForSequenceClassification`, 567 755 777 параметров;
- 3 эпохи, LR `2e-5`, cosine-to-zero, warmup `0.05`, weight decay `0.01`;
- plain finite BCE, без sampling, external weights и label smoothing;
- `max_length=384`, SDPA, BF16, clip `0.5`, seed 42;
- одна H100 80GB: batch 64, accumulation 3, effective batch 192;
- former OOD входит в train, поэтому OOD не вычисляется и считается `-1`;
- IID 12 000 пар — primary, hard 5 814 — diagnostic.

Train равен точной конкатенации `train_pairs.parquet` и
`ood_validation_pairs.parquet`: 347 840 пар, 89 291 positive, 20 категорий.
До GPU-запуска runner проверяет SHA-256 всех входов, отсутствие unordered
duplicates/cross-category пар и нулевой item overlap с IID/hard.

## Клонирование ветки

```bash
git clone --branch experiment/bge-3ep-h100 --single-branch \
  git@github.com:dinnksis/product_matching.git
cd product_matching
```

Checkpoint и prepared Parquet не лежат в Git. Перед запуском на сервере должны
существовать следующие каталоги:

```text
model/pretrain_bge_2ep/
├── config.json
├── model.safetensors
├── special_tokens_map.json
├── tokenizer.json
└── tokenizer_config.json

prepared/validation_splits_v1/human/
├── items.parquet
├── train_pairs.parquet
├── ood_validation_pairs.parquet
├── iid_validation_pairs.parquet
└── hard_validation_pairs.parquet
```

Например, с машины, где эти файлы уже находятся в корне репозитория:

```bash
SERVER=USER@HOST
REMOTE=/srv/product_matching

rsync -avP model/pretrain_bge_2ep/ \
  "$SERVER:$REMOTE/model/pretrain_bge_2ep/"
rsync -avP \
  prepared/validation_splits_v1/human/items.parquet \
  prepared/validation_splits_v1/human/train_pairs.parquet \
  prepared/validation_splits_v1/human/ood_validation_pairs.parquet \
  prepared/validation_splits_v1/human/iid_validation_pairs.parquet \
  prepared/validation_splits_v1/human/hard_validation_pairs.parquet \
  "$SERVER:$REMOTE/prepared/validation_splits_v1/human/"
```

Runner сверяет известные SHA-256, поэтому неполная или другая копия будет
отвергнута до выделения GPU.

## Установка

Проверено для Python 3.11 и Transformers 4.57.6:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-bge-h100.txt

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -c \
  "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 2**30)"
```

Workflow требует ровно одну видимую H100 с памятью не меньше 75 GiB. Он
намеренно не переключается на другую GPU или precision молча.

## Проверки без GPU

Показать план, не писать файлы:

```bash
.venv/bin/python scripts/run_bge_3ep_h100.py
```

Полностью проверить checkpoint, данные и execution hashes без записи:

```bash
.venv/bin/python scripts/run_bge_3ep_h100.py --dry-run
```

Материализовать combined train без GPU:

```bash
.venv/bin/python scripts/run_bge_3ep_h100.py --prepare-only
```

## Запуск

Рекомендуется `tmux`:

```bash
tmux new -s bge-3ep
CUDA_VISIBLE_DEVICES=0 scripts/run_bge_3ep_h100.sh 2>&1 | tee bge-3ep-console.log
```

Сначала выполняется реальный worst-case H100 preflight: BF16
forward/backward, finite gradient clipping, создание всех 393 AdamW states и
eval batch 192 при резидентных optimizer moments. Только после успешного
preflight начинается обучение.

Ориентир по времени — примерно 2.5–4.5 часа вместе с токенизацией и двумя
validation split, но фактическая скорость зависит от H100/SXM/PCIe, CPU и
диска. Во время обучения следить можно так:

```bash
tail -f artifacts/bge_3ep_sft_oodtrain_h100_v1/training.log
nvidia-smi
```

## Результаты

```text
artifacts/bge_3ep_sft_oodtrain_h100_v1/
├── prepared/
├── token_cache/
├── h100_preflight.json
├── preflight.log
├── resolved_config.json
├── training.log
├── deployment_smoke.json
├── run/
│   ├── model.safetensors
│   ├── config.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── special_tokens_map.json
│   ├── iid_validation_predictions.parquet
│   ├── hard_validation_predictions.parquet
│   ├── training_config.json
│   └── training_report.json
└── run_completed.json
```

`run_completed.json` создаётся последним и содержит SHA-256 каждого итогового
артефакта. До него runner заново загружает сохранённую модель и tokenizer на
H100 и требует finite logit. Повтор той же команды безопасно принимает только
полностью hash-валидный completion. Partial run не перезаписывается: точного
optimizer/scheduler resume в этом one-shot workflow нет.
