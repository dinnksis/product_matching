# MiniLM 5ep: 18 category-specific heads

SFT-эксперимент использует baseline human train и тот же MiniLM checkpoint после
пяти эпох pretraining, что и `notebooks/minilm_5ep_team_ablation`.

Отличие одно: sequence-classification слой имеет 18 выходов — по одному для
каждой категории, присутствующей в human train. Для каждой train-пары BCE
считается только по выходу её категории, поэтому градиент конкретного примера
обновляет соответствующую строку классификационной головы и общий encoder.

IID и hard validation используют такую же маршрутизацию. Frozen OOD содержит
две категории, которых нет в train (`Бытовая техника`, `Одежда`), поэтому для
них заранее зафиксирован fallback: среднее значение 18 логитов, после чего
применяется sigmoid. Этот выбор и полный порядок категорий сохраняются в model
config, `training_report.json` и `notebook_completed.json`.

Notebook автоматически пишет результат в `experiments_v2` и `sft_exps`, а
также считает paired significance против общего MiniLM 5ep baseline.

Сгенерировать notebook:

```bash
uv run python scripts/create_minilm_5ep_category_heads_notebook.py
```

Dry-run:

```bash
uv run python scripts/run_kaggle_notebook.py \
  notebooks/minilm_5ep_category_heads/minilm_5ep_category_heads_18_2xt4.ipynb \
  --slug minilm-5ep-category-heads-18-v1 \
  --title "MiniLM 5ep: 18 category heads" \
  --dataset alexproger23/product-matching-validation-splits-v1 \
  --dataset alexproger23/product-matching-minilm-llm-pretrain-5ep \
  --dataset alexproger23/product-matching-minilm-5ep-significance-v1 \
  --no-env-sources \
  --dry-run
```

Для remote-запуска убрать `--dry-run`; credential Dataset для Google Sheets
runner подключает автоматически.
