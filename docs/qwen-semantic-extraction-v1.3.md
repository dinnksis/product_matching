# Qwen semantic extraction v1.3: unified smoke-протокол

## Цель версии

V1.2 показала, что строгий validator умеет отбрасывать one-sided differences и
ложные anchors, но Qwen продолжает ошибаться при одновременном выборе relation и
физической секции. V1.3 удаляет секции из model output: один request возвращает
один список `semantic_facts`, а код детерминированно строит стабильные
`identity_anchors`, `differences` и `missing_information`.

Используются те же первые 50 label-free пар из `RULE_DISCOVERY`.
`rule_internal_validation`, ordinary, hard и OOD не используются. Human label не
входит в prompt и загружается только после завершения всех API futures.

## Один request на пару

```text
pair A/B
   ↓ один Qwen request
semantic_facts[]
   ↓ deterministic code normalization
anchors / differences / missing / pair summary
```

Каждая пара обрабатывается один раз. V1.3 не запускает отдельные missing/difference
calls и пока не выполняет automatic repair pass. Raw Qwen response и
schema-valid `semantic_facts` сохраняются рядом с нормализованным pair object.

## Основные контракты

- один concept встречается в `semantic_facts` максимум один раз;
- relation сама определяет заполненность сторон A/B;
- `identity_same` требует `anchor_type`, консервативно одинаковые raw values и
  совместимую пару `anchor_type ↔ concept`;
- `different_value` отклоняется для одинаковых значений и очевидных unit
  equivalents (`1000 ml = 1 l`, `1000 g = 1 kg`);
- missing имеет одну заполненную сторону и ограничен двумя facts;
- strength и pair summary строятся кодом;
- evidence обязан быть точным source-supported fragment;
- после normalization downstream JSONL/parquet имеют тот же pair-level формат,
  что и предыдущие версии.

## Запуск smoke-run 50

```powershell
python scripts/run_qwen_semantic_extraction.py `
  --dataset data/qwen_semantic_pilot_v1/pilot_inputs.parquet `
  --labels data/qwen_semantic_pilot_v1/pilot_labels.parquet `
  --prompt prompts/qwen_semantic_extraction_v1_3.md `
  --schema schemas/qwen_semantic_extraction_v1_3.schema.json `
  --prompt-version qwen_semantic_extraction_v1_3 `
  --api-base http://localhost:8194/v1 `
  --model Qwen3.5-397B-A17B-FP8 `
  --workers 2 `
  --max-pairs 50 `
  --max-tokens 4096 `
  --timeout 600 `
  --retries 3 `
  --review-size 50 `
  --output-dir artifacts/qwen_semantic_extraction_v1_3_smoke50
```

Progress line:

```text
Qwen: 20/50 new; ok=18, invalid=2, request_errors=0, reused=0
```

`invalid` не расходует retries. Повтор команды переиспользует `ok` и `invalid` с
теми же model/prompt/schema hashes и повторяет только незавершённые/request-error
пары.

## Что сравниваем после запуска

- JSON/schema/semantic validity против v1.2;
- duplicate concepts;
- one-sided relation violations;
- false/equivalent `different_value`;
- false anchors и anchor type/concept compatibility;
- missing relevance;
- source evidence;
- coverage positives/negatives только после local label join;
- representative GOOD/BAD/AMBIGUOUS examples.

На 500 пар или весь `RULE_DISCOVERY` v1.3 автоматически не переходит. Targeted
repair pass для invalid pairs рассматривается только после анализа этого smoke.

## Локальная пофактовая очистка готовых ответов

После smoke-run повторный вызов Qwen не требуется. Скрипт читает неизменяемый
`raw_responses.jsonl`, проверяет каждый `semantic_fact` отдельно и сохраняет как
принятые, так и отброшенные факты вместе с машинной причиной решения:

```powershell
python scripts/sanitize_qwen_semantic_extraction.py `
  --raw-responses artifacts/qwen_semantic_extraction_v1_3_smoke50/raw_responses.jsonl `
  --dataset data/qwen_semantic_pilot_v1/pilot_inputs.parquet `
  --labels data/qwen_semantic_pilot_v1/pilot_labels.parquet `
  --schema schemas/qwen_semantic_extraction_v1_3.schema.json `
  --prompt prompts/qwen_semantic_extraction_v1_3.md `
  --max-pairs 50 `
  --output-dir artifacts/qwen_semantic_extraction_v1_3_sanitized50
```

Скрипт не имеет параметров API и не делает сетевых вызовов. Human label
присоединяется только к итоговым локальным файлам для анализа и не участвует в
решении о принятии факта.
