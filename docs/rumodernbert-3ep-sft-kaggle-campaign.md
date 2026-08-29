# RuModernBERT: по одному SFT-запуску на Kaggle

Кампания `rumodernbert_3ep_sft_oodtrain_kaggle_v1` запускает отдельные private
Kaggle kernels с `2×Tesla T4`, строго по одному за invocation controller.
Каждая точка начинает обучение заново с точного task-pretrained checkpoint;
former OOD входит в train, а validation содержит только IID и hard. OOD metric
явно равен `-1`, OOD predictions не создаются.

Короткий план: исправленный `e1/LR=8e-5`, затем не более одного LR challenger,
выбранного после чтения baseline, затем `e2` и обязательный свежий `e3` на
выбранном LR. Никакой автоматической очереди нет. Геометрия каждого rank:
microbatch 24, accumulation 4, effective batch `24×2×4=192`, eval batch 96,
max length 384, FP16, без activation checkpointing. Перед full train выполняется
настоящий DDP forward/backward/AdamW/eval preflight с bounded GradScaler
backoff и полной проверкой 138 optimizer states и `2×149,605,633` moment
elements. Model-specific adapter явно передаёт 138 в общий validator: его
исходный default 393 относится к BGE и захватывается Python при импорте.
Checkpointing отключён после отдельного diagnostic run: ModernBERT под
autocast нарушал metadata-invariant PyTorch non-reentrant recompute. Пройденный
preflight с moments занимал только 3.42 GiB, а новый preflight до обучения
обязан измерить уже честный no-checkpoint peak.

## Подготовка private checkpoint Dataset

```bash
.venv/bin/python scripts/push_rumodernbert_pretrain_checkpoint_dataset.py --dry-run
.venv/bin/python scripts/push_rumodernbert_pretrain_checkpoint_dataset.py
```

`optimizer.pt` намеренно исключён: каждая SFT-точка создаёт свежие AdamW и
cosine scheduler. Validation использует уже опубликованный private Dataset
`alexproger23/product-matching-validation-splits-v1`.

## Локальный no-network stage

```bash
.venv/bin/python scripts/run_rumodernbert_sft_kaggle.py \
  --key e1_lr8e5 --stage-only
```

Команда создаёт и компилирует ровно один notebook, metadata с exact тремя
private Dataset refs и не обращается к Kaggle.

## Запуск одного эксперимента

```bash
.venv/bin/python scripts/run_rumodernbert_sft_kaggle.py \
  --key e1_lr8e5 --execute
```

Для `e2/e3` LR задаётся явно, например
`--key e2_selected_lr --selected-lr 8e-5 --execute`. Controller никогда не
заменяет terminal-failed slug. Первый неудачный baseline
`pm-rmb-e1-lr8e5-06abd6796702-s42-v1` tombstoned и не переиспользуется.
Его диагностический successor `pm-rmb-e1-lr8e5-c37bbe4011cc-s42-v1` также
tombstoned: он подтвердил, что причина была в захваченном BGE default `393`.
Третий diagnostic `pm-rmb-e1-lr8e5-092f927d6256-s42-v1` tombstoned после
успешного preflight: он выявил несовместимость non-reentrant checkpointing с
ModernBERT/autocast на первом настоящем backward.
Существующий running kernel только мониторится, complete kernel скачивается и
строго валидируется. После каждого отдельного запуска проверяются pair identity,
метрики, OOD=-1, Sheets receipt, memory-preflight contract и SHA сохранённой
модели. Следующий эксперимент выбирается только после ручного анализа текущего.
