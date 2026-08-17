# Full fine-tuning на LLM-корпусе на одной или нескольких GPU

[`scripts/train_llm_full.py`](../scripts/train_llm_full.py) обучает все параметры
MiniLM cross-encoder на всех `10 043 007` парах из `llm/non_ood_pairs.parquet`.
Ни одна строка не фильтруется: значения `0`, `1/9`, ..., `8/9`, `1` передаются
в supervised BCE как исходные soft targets. Поверх него включена
[Early-Learning Regularization](https://proceedings.neurips.cc/paper_files/paper/2020/hash/ea89621bee7c88b2c5be6681c8ef4906-Abstract.html),
которая не даёт модели за десять эпох просто запомнить шум слабой разметки.

По умолчанию OOD-файлы не используются. Это сохраняет категории «Одежда» и
«Бытовая техника» невиданными при обучении. Флаг `--include-ood` расширяет train
до всех `11 187 780` LLM-пар, но после этого frozen OOD-валидация больше не
является честной.

## Установка на сервере

Нужны Python 3.11–3.12, одна или несколько CUDA GPU на одном сервере и
достаточно локального SSD для исходных parquet, mmap-кэша и checkpoints. Для
H100 используется BF16. `deepspeed` и `kernels` этому entry point не нужны и
не входят в `requirements-cross-encoder.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cross-encoder.txt
```

Проверка окружения:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
```

## Запуск

Из корня репозитория:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_llm_full.py \
  --data-dir prepared/validation_splits_v1/llm \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --human-validation-dir prepared/validation_splits_v1/human \
  --human-items data/items_human.parquet \
  --output-dir models/minilm_llm_full \
  --cache-dir artifacts/llm_full_cache \
  --epochs 10 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --learning-rate 5e-6 \
  --elr-beta 0.7 \
  --elr-lambda 3.0 \
  --max-length 512 \
  --serialization-variant S1_KEY_VALUE \
  --num-workers 8
```

`batch-size=256` задаётся **на одну GPU**. Effective batch равен
`batch_size × world_size × gradient_accumulation`. После проверки peak VRAM
per-device batch можно увеличить; при OOM достаточно уменьшить его. Скрипт не
масштабирует LR автоматически.

## Несколько GPU через torchrun

Один процесс закрепляется за одной GPU, а градиенты синхронизируются DDP. Все
пары глобально проходят ровно один раз за эпоху: sampler не дополняет последний
batch дублями и не выбрасывает остаток. Validation также делится между ranks,
но predictions и metrics записывает только rank 0.

Например, запуск на четырёх H100:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  scripts/train_llm_full.py \
  --data-dir prepared/validation_splits_v1/llm \
  --model cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 \
  --human-validation-dir prepared/validation_splits_v1/human \
  --human-items data/items_human.parquet \
  --output-dir models/minilm_llm_full_4gpu \
  --cache-dir artifacts/llm_full_cache \
  --epochs 10 \
  --batch-size 256 \
  --eval-batch-size 512 \
  --learning-rate 5e-6 \
  --elr-beta 0.7 \
  --elr-lambda 3.0 \
  --max-length 512 \
  --serialization-variant S1_KEY_VALUE \
  --num-workers 8
```

Эквивалент через Makefile:

```bash
make train-llm-full LLM_NPROC=4 TRAIN_ARGS="--output-dir models/minilm_llm_full_4gpu"
```

При `batch-size=256` effective batch для четырёх GPU равен `1024`. Для строгого
сравнения с single-GPU effective batch `256` передайте `--batch-size 64`; для
максимального throughput можно оставить `256`, но это уже другой optimizer
schedule по числу примеров на update.

`max_length=512` выбран намеренно. У XLM-R checkpoint из команды это штатный
предел tokenizer (в конфигурации encoder — 514 позиций с учётом служебных
токенов). В имеющихся validation predictions при длине 384 около 21,6% примеров
уже упирались в cap, поэтому 384 для полного корпуса слишком агрессивен. Скрипт
проверяет предел tokenizer и не позволит случайно передать неподдерживаемую
длину.

`5e-6` — более консервативный peak LR для десяти проходов по 10,04 млн пар.
Scheduler делает warmup на первых 3% всех optimizer updates, затем cosine decay
до нуля. Это означает 100 430 070 train-примеров за полный запуск, поэтому перед
ним имеет смысл обязательно выполнить smoke-run.

## Early-Learning Regularization

Для каждого train pair хранится двухкомпонентная EMA прошлых предсказаний
`tᵢ`. Один logit cross-encoder преобразуется в
`pᵢ = [1-sigmoid(logit), sigmoid(logit)]`, после чего используется loss:

```text
BCE(logit, soft_target) + λ · mean(log(1 - <pᵢ, tᵢ>))
tᵢ ← β · tᵢ + (1-β) · stop_gradient(pᵢ)
```

Это binary-эквивалент формул (6) и (9) статьи. Defaults `β=0.7`, `λ=3.0` и
clamp `1e-4` совпадают с
[официальной реализацией](https://github.com/shengliu66/ELR). ELR-history имеет
форму `[10 043 007, 2]`, занимает около 77 MiB FP32 на GPU и сохраняется в
`elr_targets.pt` каждого checkpoint. Без этого файла точный resume невозможен.
В DDP каждая GPU держит реплику history. Между checkpoint синхронизируются
накопленные rank-local изменения; перед сохранением все ranks получают
одинаковую полную history.

## Альтернатива: soft BCE + pairwise margin distillation

Для отдельного контролируемого эксперимента есть launcher
[`scripts/run_llm_full_margin_distillation.sh`](../scripts/run_llm_full_margin_distillation.sh).
Он сохраняет ELR как регуляризацию, но уменьшает его коэффициент с `3.0` до
`1.0`, и оптимизирует

```text
tᵢ = log(clamp(qᵢ) / (1 - clamp(qᵢ)))
L = BCEWithLogits(zᵢ, qᵢ)
    + 1.0 · ELR(zᵢ, historyᵢ)
    + λ · Huber((zᵢ - zⱼ) - (tᵢ - tⱼ) / T)
```

Здесь `q` — soft probability LLM teacher, `t` — её logit, `z` — logit
student. Сравнения строятся **только между парами одной товарной категории**.
Внутри каждого mini-batch примеры категории сортируются по teacher logit, после
чего нижняя половина попарно сравнивается с верхней. Равные teacher scores не
дают comparison. Такое deterministic extreme pairing не тратит основную часть
вычислений на многочисленные пары `0` против `0` и имеет линейную стоимость по
batch size.

Первый запуск с этим loss достроит рядом с существующим токен-кэшем
`pair_category_ids.npy` и `pair_categories.json`. Для 10 млн строк sidecar
занимает примерно 20 MiB; item-тексты не токенизируются повторно. Последующие
запуски переиспользуют его по fingerprint исходного item parquet.

Рекомендуемый первый прогон на одной H100:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_llm_full_margin_distillation.sh
```

Defaults этого launcher зафиксированы как отдельная абляция:

- 5 эпох, `learning_rate=5e-6`, `batch_size=256`;
- `λ=0.1`, `T=1`, Huber `delta=1`;
- ELR `β=0.7`, ELR coefficient `1.0` вместо прежнего `3.0`;
- teacher probabilities `0/1` clamp до `1e-4 / (1-1e-4)` перед logit;
- `max_length=512`, `S1_KEY_VALUE`, тот же human validation после эпохи;
- output — `models/minilm_llm_full_elr1_margin_l01_5ep`.

Для второго эксперимента с `λ=0.2` обязательно используйте другой output:

```bash
CUDA_VISIBLE_DEVICES=0 \
PAIRWISE_MARGIN_LAMBDA=0.2 \
LLM_OUTPUT_DIR=models/minilm_llm_full_elr1_margin_l02_5ep \
scripts/run_llm_full_margin_distillation.sh
```

Resume сохраняет обычную семантику полного trainer:

```bash
CUDA_VISIBLE_DEVICES=0 scripts/run_llm_full_margin_distillation.sh --resume
```

В train JSON дополнительно выводятся `pairwise_margin_loss`, число сравнений,
comparisons per example и средние абсолютные teacher/student margins. Полный
loss в этой конфигурации равен
`supervised_loss + elr_regularizer + λ * pairwise_margin_loss`, где уже
выведенный `elr_regularizer` умножен на коэффициент `1.0`. Поскольку ELR-терм
логарифмический и отрицательный, суммарный loss всё ещё может стать
отрицательным; для сравнения качества нужно отдельно следить за
`supervised_loss`, `pairwise_margin_loss` и human AP.

На нескольких GPU comparisons строятся внутри local mini-batch каждого rank,
но никогда между ranks. Поэтому для строгой абляции сначала лучше сравнить
обычный loss и margin loss на одной H100 с одинаковым batch size. Multi-GPU
запуск поддержан через `LLM_NPROC`, например:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
LLM_NPROC=8 \
LLM_BATCH_SIZE=32 \
scripts/run_llm_full_margin_distillation.sh
```

Так global batch останется равен single-GPU batch `256`. Если оставить
`LLM_BATCH_SIZE=256`, global batch станет `2048`, а набор внутриранговых
comparisons и optimizer schedule уже будут другим экспериментом.

## Human validation после каждой эпохи

После каждой эпохи checkpoint оценивается на frozen `iid`, `hard` и `ood` в
обоих порядках пары; итоговый score — среднее вероятностей A/B и B/A. Для всех
трёх наборов сохраняются overall AP, macro AP, AP по категориям, order gap и
predictions parquet.

Human-тексты сериализуются тем же `S1_KEY_VALUE` и с тем же частотным порядком
атрибутов, что и LLM train. Поэтому нужен сырой
`data/items_human.parquet` с колонками `id/name/attributes/category`; старый
`prepared/.../human/items.parquet` содержит только прежний `product_text` и для
этого запуска не используется.

`checkpoint-best` выбирается только по `IID macro AP`. Hard и OOD остаются
диагностическими метриками и не участвуют в выборе модели. Все эпохи всё равно
сохраняются, поэтому кривые трёх протоколов можно сравнить вручную.

## Сериализация товаров

По умолчанию используется точный вариант `S1_KEY_VALUE` из serialization
ablation. Текст имеет вид:

```text
нормализованный title. ключ: значение. ключ: значение
```

Применяются NFKC, casefold, замена `ё` на `е`, схлопывание пробелов и
нормализация единиц (`512 ГБ` → `512 gb`, `1 ТБ` → `1 tb` и т. п.). Вложенные
и списочные значения разворачиваются тем же способом, что в ablation. Категория
в текст не добавляется. Ключи сортируются по глобальной частоте на всех товарах,
на которые ссылаются выбранные train-пары: сначала `occurrences`, затем
`item_support`, затем имя ключа.

Полученный порядок сохраняется в cache как
`attribute_name_frequency.csv`. Если нужно буквально переиспользовать таблицу
из отдельного ablation-run, передайте:

```bash
python scripts/train_llm_full.py \
  --attribute-frequency-csv path/to/attribute_name_frequency.csv
```

Доступны также `S0_TITLE`, `S2_VALUES_ONLY` и `S3_HYBRID`. Для `S3_HYBRID`
нужно дополнительно передать исходный список:

```bash
python scripts/train_llm_full.py \
  --serialization-variant S3_HYBRID \
  --frequent-keys-json path/to/frequent_attribute_names.json
```

Пайплайн сначала:

1. находит все товары, реально используемые парами;
2. строит частотный ранг нормализованных атрибутов;
3. потоково сериализует и токенизирует каждый уникальный товар ровно один раз;
4. записывает токены и pair-index в mmap-кэш;
5. запускает 10 эпох full fine-tuning всех параметров модели в BF16 с ELR;
6. после каждой эпохи считает IID/hard/OOD validation;
7. сохраняет `checkpoint-last`, checkpoint каждой эпохи и лучший по IID.

Ожидаемые результаты:

```text
models/minilm_llm_full/
├── checkpoint-last/
│   ├── model.safetensors
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── elr_targets.pt
│   ├── rng_state.pt
│   └── training_state.json
├── checkpoint-best/
├── checkpoint-epoch-01/
├── ...
├── checkpoint-epoch-10/
├── validation/
│   ├── epoch-01/
│   │   ├── metrics.json
│   │   ├── iid_predictions.parquet
│   │   ├── hard_predictions.parquet
│   │   └── ood_predictions.parquet
│   └── ...
├── model/
│   ├── model.safetensors
│   └── tokenizer.json
├── training_args.json
├── validation_history.json
└── training_report.json
```

Epoch и best directories являются hard-link snapshots `checkpoint-last`: внутри
одной файловой системы неизменившиеся большие файлы физически не дублируются.
Если файловая система не поддерживает hard links, скрипт автоматически
переходит на обычное копирование. При копировании каждого каталога на другой
диск по отдельности экономия тоже пропадёт.

Кэш переиспользуется при повторном запуске с теми же файлами, tokenizer,
`max_length` и параметрами сериализации. Для отдельного построения кэша без GPU
training:

```bash
python scripts/train_llm_full.py --cache-only
```

Для быстрой сквозной проверки перед полным запуском:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_llm_full.py \
  --max-pairs 100000 \
  --output-dir models/minilm_llm_smoke \
  --cache-dir artifacts/llm_full_smoke_cache \
  --batch-size 256 \
  --save-every-updates 0
```

## Возобновление

Скрипт перезаписывает `checkpoint-last` каждые 5000 optimizer updates и после
каждой эпохи. Для продолжения нужны те же data/cache/training параметры:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_llm_full.py \
  --output-dir models/minilm_llm_full \
  --cache-dir artifacts/llm_full_cache \
  --batch-size 256 \
  --resume
```

Resume восстанавливает модель, optimizer, scheduler, ELR temporal targets, номер
эпохи, следующий batch и отдельные RNG states каждого rank. Порядок пар и
ориентация `A/B` также детерминированы. Для точного resume число процессов
должно совпадать с checkpoint: single-GPU checkpoint нельзя продолжить как
4-GPU DDP и наоборот.

## Совместимость с другими backbone

Текущий backend не привязан к XLM-R классами слоёв. `--model` может указывать на
другой checkpoint, если одновременно выполняются условия:

- он загружается через `AutoModelForSequenceClassification`;
- выдаёт ровно один ranking logit на пару;
- tokenizer поддерживает стандартный pair template
  `build_inputs_with_special_tokens(first, second)`;
- tokenizer/model поддерживает выбранный `max_length`.

Это покрывает многие BERT/RoBERTa/XLM-R/DeBERTa-style cross-encoders. Для
моделей с проверенным пользовательским Hub-кодом есть явный opt-in
`--trust-remote-code`; без флага произвольный remote code не выполняется.

`Qwen/Qwen3-Reranker-0.6B` напрямую этим контрактом **не покрывается**. Его
исходный checkpoint — `AutoModelForCausalLM`: нужно собрать специальный chat
prompt, использовать left padding и взять разность last-token logits токенов
`yes`/`no`. Простой `--model Qwen/Qwen3-Reranker-0.6B` потерял бы оригинальный
reranker prompt и был бы некорректен. Официальный `sentence_transformers.CrossEncoder`
умеет оборачивать этот checkpoint для inference, но текущий custom training loop
через эту обёртку не проходит. Правильный causal path уже реализован для
human-обучения в `scripts/train_qwen_names.py`; перенос его на полный LLM-кэш с
ELR должен быть отдельным backend, потому что потребует другого token cache и
существенно меньшего per-device batch.

## Явное включение OOD

Только для финальной модели, когда OOD уже не используется для выбора рецепта:

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 \
  scripts/train_llm_full.py \
  --include-ood \
  --output-dir models/minilm_llm_all_categories \
  --cache-dir artifacts/llm_full_all_categories_cache
```

## Последующий human fine-tune

Итоговый checkpoint совместим с human fine-tune, но human train тоже нужно
подавать через тот же `S1_KEY_VALUE` serializer и частотный ранг из LLM cache.
Validation path в этом скрипте уже делает это корректно; старый generic trainer
с готовым `product_text` всё ещё использует прежний формат.
