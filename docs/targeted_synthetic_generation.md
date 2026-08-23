# Targeted synthetic data через Qwen3.5-397B

Генератор: `scripts/generate_targeted_synthetic.py`.
Он использует `prepared/human/items.parquet`, human train pairs и OOF S2
`artifacts/kaggle/product-matching-minilm-s2-targeted-hard-results2/.../oof_predictions_and_hardness.parquet`.
Для hard negatives используется только одобренный Qwen-регистр
`reports/variant_attributes/qwen_validation.jsonl` и значения, встречавшиеся в
human items. Для каждого примера выполняются два независимых вызова Qwen:
generation и validation.

## Проверка и smoke-run

В SSH-сессии, где поднят vLLM:

```powershell
Invoke-RestMethod http://localhost:8193/v1/models
python scripts/generate_targeted_synthetic.py --limit 2 --workers 2 `
  --hard-negative-target 2 --hard-positive-target 2 --ood-target 2
```

Если туннель доступен только через `127.0.0.1` или `0.0.0.0`, добавьте
соответствующий `--api-base` (например, `--api-base http://127.0.0.1:8193/v1`). В модели по умолчанию уже указано
`Qwen3.5-397B-A17B-FP8`.

## Полный запуск

```powershell
python scripts/generate_targeted_synthetic.py --workers 20 `
  --hard-negative-target 10000 --hard-positive-target 7000 --ood-target 7000
```

Dataset 3 запускается только после того, как в output появились parquet для
первых двух datasets. Запуск можно повторять с теми же параметрами: уже
записанные generation/validation jobs не отправляются повторно, поэтому после
обрыва продолжается с checkpoint.

Для запуска без привязки к SSH-окну PowerShell:

```powershell
$out = "artifacts/targeted_synthetic_v3"
New-Item -ItemType Directory -Force $out | Out-Null
Start-Process python -WindowStyle Hidden -RedirectStandardOutput "$out/run.stdout.log" `
  -RedirectStandardError "$out/run.stderr.log" -ArgumentList @(
  "scripts/generate_targeted_synthetic.py", "--workers", "20",
  "--hard-negative-target", "10000", "--hard-positive-target", "7000",
  "--ood-target", "7000")
```

## Мониторинг

```powershell
Get-Content artifacts/targeted_synthetic_v3/*.status.json
Get-Content artifacts/targeted_synthetic_v3/run.stdout.log -Tail 20
Get-Content artifacts/targeted_synthetic_v3/run.stderr.log -Tail 20
```

`<dataset>.status.json` содержит dataset, phase (`generation`, `validation`,
`completed` или `completed_shortfall`), generated/validated/pending/accepted/
rejected и время обновления. Отдельные status-файлы позволяют безопасно
запускать D1 и D3 параллельно.
Подробные append-only checkpoints находятся в
`<dataset>.generation.jsonl` и `<dataset>.validation.jsonl`; их размер и число
строк можно смотреть так:

```powershell
Get-ChildItem artifacts/targeted_synthetic_v3/*.jsonl |
  Select-Object Name,Length
```

Финальные файлы:

- `hard_negatives_v3.parquet` и `hard_negatives_v3_items.parquet`;
- `hard_positives_v1.parquet` и `hard_positives_v1_items.parquet`;
- `ood_style_positives_v1.parquet` и `ood_style_positives_v1_items.parquet`.

Для каждого dataset также сохраняются `<dataset>_stats.json` и 30 случайных
примеров в `<dataset>_examples.jsonl`. При падении API достаточно посмотреть
`run.stderr.log` и снова выполнить ту же команду; незавершённые jobs будут
повторены, завершённые — переиспользованы. Если нужно уменьшить concurrency,
остановите foreground-процесс через `Ctrl+C` и продолжите нужный dataset с
меньшим `--workers`: JSONL checkpoints сохраняются. Для hard negatives v3
sampler умеет создавать несколько разных attribute/value counterfactuals из
одной clean positive пары, чтобы добрать target.

## Подключение к Kaggle и запуск трёх сравнений

В Kaggle Dataset нужно загрузить финальные parquet-файлы и item-файлы. Удобно
создать отдельную папку с шестью файлами:

```powershell
$payload = "artifacts/kaggle_targeted_synthetic_v3"
New-Item -ItemType Directory -Force $payload | Out-Null
Copy-Item artifacts/targeted_synthetic_v3/hard_negatives_v3*.parquet $payload
Copy-Item artifacts/targeted_synthetic_v3/hard_positives_v1*.parquet $payload
Copy-Item artifacts/targeted_synthetic_v3/ood_style_positives_v1*.parquet $payload
```

Добавьте в эту папку `dataset-metadata.json` с вашим Kaggle username:

```json
{
  "title": "Product matching targeted synthetic v3",
  "id": "YOUR_USERNAME/product-matching-targeted-synthetic-v3",
  "licenses": [{"name": "CC0-1.0"}],
  "description": "Qwen-validated targeted synthetic product matching data"
}
```

Затем создайте/обновите Dataset:

```powershell
kaggle datasets create -p artifacts/kaggle_targeted_synthetic_v3 --keep-tabular
# для повторной версии:
kaggle datasets version -p artifacts/kaggle_targeted_synthetic_v3 -m "Qwen targeted synthetic v3"
```

После публикации запустите каждый notebook отдельно, подключив этот Dataset
(обычные validation/checkpoint Dataset берутся из `.env`):

```powershell
python scripts/run_kaggle_notebook.py notebooks/minilm_5ep_team_ablation/minilm_5ep_hard_negatives_v3_2xt4.ipynb `
  --dataset YOUR_USERNAME/product-matching-targeted-synthetic-v3 `
  --slug minilm-5ep-hard-negatives-v3 --title "MiniLM 5ep hard negatives v3" --no-wait

python scripts/run_kaggle_notebook.py notebooks/minilm_5ep_team_ablation/minilm_5ep_hard_positives_v1_2xt4.ipynb `
  --dataset YOUR_USERNAME/product-matching-targeted-synthetic-v3 `
  --slug minilm-5ep-hard-positives-v1 --title "MiniLM 5ep hard positives v1" --no-wait

python scripts/run_kaggle_notebook.py notebooks/minilm_5ep_team_ablation/minilm_5ep_ood_style_positives_v1_2xt4.ipynb `
  --dataset YOUR_USERNAME/product-matching-targeted-synthetic-v3 `
  --slug minilm-5ep-ood-style-positives-v1 --title "MiniLM 5ep OOD style positives v1" --no-wait
```

Не используйте `--no-env-sources`: baseline Dataset с frozen validation и
checkpoint должен остаться подключённым из `.env`. В текущей версии сравнения
human и synthetic rows получают `sample_weight=1.0`; frozen loss hook уже
умножает per-example BCE на этот вес.
