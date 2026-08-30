# MiniLM: human-only error curriculum v1

Эксперимент продолжает fine-tuning из того же frozen MiniLM 5ep checkpoint,
что и командный data/loss-ablation шаблон. Основой остаются 306 669 human-пар.
Для 9 311 audit-eligible пар из `RULE_DISCOVERY` вес увеличивается с 1.0 до 2.0:
это эквивалентно добавлению одной копии с весом 1.0, но не нарушает frozen guard,
запрещающий duplicate unordered pairs.

Синтетика и Qwen-labels не используются. Структурированная семантика Qwen была
нужна только для анализа ошибок и в MiniLM input не передаётся.

## Подготовка Kaggle Dataset

```powershell
python scripts/prepare_minilm_human_error_curriculum_kaggle_payload.py

kaggle datasets create `
  -p artifacts/kaggle_datasets/minilm_human_error_curriculum_v1 `
  --keep-tabular
```

Если Dataset с таким slug уже существует:

```powershell
kaggle datasets version `
  -p artifacts/kaggle_datasets/minilm_human_error_curriculum_v1 `
  -m "Human-only OOF error curriculum v1"
```

## Запуск notebook

```powershell
python scripts/run_kaggle_notebook.py `
  notebooks/minilm_5ep_team_ablation/minilm_5ep_human_error_curriculum_v1_2xt4.ipynb `
  --dataset alexproger23/product-matching-validation-splits-v1 `
  --dataset alexproger23/product-matching-minilm-llm-pretrain-5ep `
  --dataset alexproger23/product-matching-minilm-5ep-significance-v1 `
  --dataset alexproger23/ecom-matching-google-sheets-credentials `
  --dataset dinakepecheva/product-matching-minilm-human-error-curriculum-v1 `
  --slug minilm-5ep-human-error-curriculum-v1 `
  --title "MiniLM 5ep human error curriculum v1" `
  --no-env-sources
```

Для проверки без отправки добавить `--dry-run`. Для отправки без ожидания и
скачивания результатов добавить `--no-wait`.

## Статистики

Notebook автоматически считает IID/hard/OOD macro AP и overall AP, сохраняет
predictions и training report, а также сравнивает каждый split с frozen baseline
через permutation p-value, Holm correction и bootstrap confidence interval.
Результат записывается только в сравнительный лист `data_exps`; запись строки в
`experiments_v2` для этого notebook явно отключена. Маршрут зафиксирован как
`EXPERIMENT_SHEET = "data_exps"`.
