# Qwen semantic extraction v1.2: smoke-протокол

## Цель версии

V1.1 устранила обрезание JSON, но выявила систематические semantic errors:
односторонние значения в `differences`, дублирование facts между секциями,
ложные equal identity anchors и перегруженный missing context.

V1.2 проверяется на тех же первых 50 label-free парах из `RULE_DISCOVERY`.
Internal validation, ordinary, hard и OOD не используются. Human label не входит
в Qwen input и загружается кодом только после завершения всех API futures.

## Изменения

- `same` удалён из допустимых relations для `differences`;
- обычный difference обязан иметь value и evidence обеих сторон;
- односторонний факт имеет компактную форму `relation + known_value +
  known_evidence` и может находиться только в `missing_information`;
- один concept запрещено дублировать внутри секции или между секциями;
- identity anchor принимается только при консервативном совпадении raw A/B после
  удаления регистра, пробелов и пунктуации;
- добавлены отрицательные anchor-примеры `TT685IIC != TT685IIS` и
  `01303N0 != 44881N3`;
- strength больше не генерируется Qwen: код назначает strong для exact
  SKU/MPN/model, medium для family/line и weak для brand;
- Qwen больше не генерирует summary/warnings; summary строится кодом только из
  реально принятых facts;
- missing ограничен двумя фактами, явно исключены операционные поля вроде срока
  годности и размеров транспортной упаковки;
- raw Qwen response сохраняется без изменений, а стабильное A/B-представление
  missing создаётся отдельно кодом.

## Запуск smoke-run 50

Используется новый output directory; v1 и v1.1 не перезаписываются.

```powershell
python scripts/run_qwen_semantic_extraction.py `
  --dataset data/qwen_semantic_pilot_v1/pilot_inputs.parquet `
  --labels data/qwen_semantic_pilot_v1/pilot_labels.parquet `
  --prompt prompts/qwen_semantic_extraction_v1_2.md `
  --schema schemas/qwen_semantic_extraction_v1_2.schema.json `
  --prompt-version qwen_semantic_extraction_v1_2 `
  --api-base http://localhost:8194/v1 `
  --model Qwen3.5-397B-A17B-FP8 `
  --workers 2 `
  --max-pairs 50 `
  --max-tokens 4096 `
  --timeout 600 `
  --retries 3 `
  --review-size 50 `
  --output-dir artifacts/qwen_semantic_extraction_v1_2_smoke50
```

Прогресс:

```text
Qwen: 20/50 new; ok=18, invalid=2, request_errors=0, reused=0
```

`invalid` не повторяется: retries используются только для request/server errors.
Повтор той же команды продолжает незавершённые запросы и переиспользует ответы с
теми же model/prompt/schema hashes.

## Проверка после запуска

V1.2 сравнивается с v1.1 на тех же `pair_id` по:

- JSON/schema/semantic validity;
- отсутствию one-sided `differences`;
- отсутствию concept duplicates между секциями;
- ложным equal anchors;
- missing relevance;
- source-supported evidence;
- representative GOOD/BAD/AMBIGUOUS examples.

Даже при высокой технической validity переход на 500 пар выполняется только после
ручной проверки semantic signatures. Автоматического запуска на всём
`RULE_DISCOVERY` нет.

Smoke-run завершён и разобран в
`reports/qwen_semantic_extraction_v1_2_smoke50/analysis.md`. Следующая версия —
v1.3 с единым массивом `semantic_facts`; v1.2 больше не продолжается.
