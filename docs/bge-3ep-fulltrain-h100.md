# BGE 3ep: финальный H100 export на всех human-парах

Этот workflow использует тот же проверенный checkpoint и тот же трёхэпоховый
H100-рецепт, что `bge_3ep_sft_oodtrain_h100_v1`, но переводит в train все
human split:

| Источник | Строк | Positive |
| --- | ---: | ---: |
| train | 306 669 | 80 136 |
| IID | 12 000 | 3 118 |
| hard | 5 814 | 1 481 |
| former OOD | 41 171 | 9 155 |
| **Итого** | **365 654** | **93 890** |

После такого запуска честного validation больше нет. Runner не создаёт
predictions и не заявляет AP: IID, hard и OOD имеют sentinel `-1`. Назначение
этого run — получить deployable checkpoint после окончательного выбора рецепта.

Рецепт: fresh start от checkpoint `cdaf66bb…`, 3 эпохи, LR `2e-5`, BF16,
batch 64 × accumulation 3, cosine schedule, warmup `0.05`, WD `0.01`, plain
BCE, max length 384, одна H100 80GB.

## Проверка и подготовка

Запускать после завершения обычного трёхэпохового эксперимента: два процесса не
должны одновременно занимать одну H100.

```bash
cd ~/aekuleshevskii/product_matching-bge3
source .venv/bin/activate

CKPT="/home/jovyan/aekuleshevskii/product_matching/models/bge_reranker_v2_m3_llm_full_20cat_elr1_5ep_1gpu_eager_bs32/checkpoint-epoch-03"

python scripts/run_bge_3ep_h100_fulltrain.py \
  --model-dir "$CKPT" \
  --dry-run

python scripts/run_bge_3ep_h100_fulltrain.py \
  --model-dir "$CKPT" \
  --prepare-only
```

`--prepare-only` материализует точный train в отдельном workdir и проверяет:

- 365 654 строк и 93 890 positive;
- все 20 категорий и все 711 304 товара;
- отсутствие self/cross-category/unordered duplicate пар;
- точные SHA каждого исходного split;
- порядок train → IID → hard → former OOD.

## Запуск

```bash
tmux new -s bge-3ep-full

cd ~/aekuleshevskii/product_matching-bge3
source .venv/bin/activate
CKPT="/home/jovyan/aekuleshevskii/product_matching/models/bge_reranker_v2_m3_llm_full_20cat_elr1_5ep_1gpu_eager_bs32/checkpoint-epoch-03"

CUDA_VISIBLE_DEVICES=0 scripts/run_bge_3ep_h100_fulltrain.sh \
  --model-dir "$CKPT"
```

У full corpus последний batch содержит 22 строки. Trainer нормализует каждый
microbatch по точному числу примеров во всём accumulation group, поэтому эти
строки не получают повышенный вес. План содержит 5 714 microsteps и 1 905
optimizer updates на эпоху, 5 715 updates за три эпохи и 285 warmup updates.

## Артефакты

```text
artifacts/bge_3ep_fulltrain_h100_v1/
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
│   ├── training_config.json
│   └── training_report.json
└── run_completed.json
```

`run_completed.json` создаётся последним, после exact file-set проверки,
SHA-256 ledger и reload-smoke сохранённой модели на H100. Excel/Google Sheets
этот серверный export не изменяет; quality metrics намеренно отсутствуют.
