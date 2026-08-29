# RuModernBERT: пять последовательных SFT-запусков на Kaggle

Кампания `rumodernbert_3ep_sft_oodtrain_kaggle_v1` переносит исходный
одногпу-H100 план на пять отдельных private Kaggle kernels с `2×Tesla T4`.
Каждая точка начинает обучение заново с точного task-pretrained checkpoint;
former OOD входит в train, а validation содержит только IID и hard. OOD metric
явно равен `-1`, OOD predictions не создаются.

Порядок фиксирован: `e1/LR=8e-5`, `e1/4e-5`, `e1/1.6e-4`, затем `e2` и
обязательный `e3` на LR, выбранном по IID. Геометрия каждого rank:
microbatch 24, accumulation 4, effective batch `24×2×4=192`, eval batch 96,
max length 384, FP16, gradient checkpointing. Перед full train выполняется
настоящий DDP forward/backward/AdamW/eval preflight с bounded GradScaler
backoff и полной проверкой 138 optimizer states.

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
.venv/bin/python scripts/run_rumodernbert_sft_kaggle.py --stage-only
```

Команда создаёт и компилирует пять notebooks, metadata с exact тремя private
Dataset refs и не обращается к Kaggle.

## Последовательный запуск

```bash
.venv/bin/python scripts/run_rumodernbert_sft_kaggle.py --execute
```

Controller никогда не заменяет terminal-failed slug. Существующий running
kernel только мониторится, complete kernel скачивается и строго валидируется.
После каждого запуска проверяются pair identity, метрики, OOD=-1, Sheets receipt
и SHA сохранённой модели. После пяти runs создаётся
`reports/rumodernbert_3ep_sft_oodtrain_kaggle_v1/campaign_summary.json` с путём
и SHA выбранного checkpoint. Все пять model folders сохраняются под
`artifacts/kaggle/<kernel-slug>/`.
