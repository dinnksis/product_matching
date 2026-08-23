# Qwen semantic extraction v1.4

## Что изменено

V1.4 исправляет классы ошибок, найденные при ручной проверке 50 результатов
v1.3. Одна пара по-прежнему обрабатывается одним Qwen request. Human label не
передаётся модели и присоединяется кодом только после завершения inference.

Новые требования prompt и исполняемые semantic checks:

- country-like значения вроде `КНР` запрещены как brand;
- `нет бренда` и другие absence markers не считаются известным значением;
- package quantity требует явного числа и не может извлекаться из веса,
  размеров или `с чехлом`;
- missing отклоняется, если известное значение найдено в title/attributes
  якобы пустой стороны;
- тип товара из attribute не считается межтоварным различием, если title этой
  же карточки поддерживает значение другой стороны: это вероятный source
  conflict;
- безопасные transliteration/script variants brand разрешены через одинаковый
  normalized value при сохранении всех raw tokens;
- добавлены инструкции не смешивать brand/country, model/product line/theme и
  package quantity/measurements.

Структура JSON не изменилась, поэтому намеренно используется уже
зафиксированная `qwen_semantic_extraction_v1_3.schema.json`. Версию эксперимента
определяют новый prompt и `validation_profile=v1_4`; оба значения и SHA-256
записываются в checkpoint/manifest.

## Проверка без API

Dry-run выполнен на 50 парах: input остаётся label-free и содержит только
category, title и attributes обеих карточек. Семь новых unit-тестов проходят.

Локальное применение v1.4 validators к старым 50 raw outputs дополнительно:

- удалило `КНР → brand`;
- удалило `с чехлом → package_quantity`;
- удалило `вес 5 г → package_quantity=5`;
- обнаружило два неоформленных source conflicts;
- обнаружило два ложных missing, включая аромат `лимон`;
- восстановило безопасные anchors `Fito Cosmetic ↔ Fito Косметик` и
  `Laima ↔ Лайма`.

Это diagnostic revalidation старых outputs, а не замена нового Qwen smoke-run.

## Команда сопоставимого smoke-run

Сначала следует снова прогнать те же первые 50 пар — это позволит сравнить v1.4
с v1.3 pair-by-pair:

```powershell
python scripts/run_qwen_semantic_extraction.py `
  --dataset data/qwen_semantic_pilot_v1/pilot_inputs.parquet `
  --labels data/qwen_semantic_pilot_v1/pilot_labels.parquet `
  --prompt prompts/qwen_semantic_extraction_v1_4.md `
  --schema schemas/qwen_semantic_extraction_v1_3.schema.json `
  --prompt-version qwen_semantic_extraction_v1_4 `
  --validation-profile v1_4 `
  --api-base http://localhost:8194/v1 `
  --model Qwen3.5-397B-A17B-FP8 `
  --workers 2 `
  --max-pairs 50 `
  --max-tokens 4096 `
  --timeout 600 `
  --retries 3 `
  --review-size 50 `
  --output-dir artifacts/qwen_semantic_extraction_v1_4_smoke50
```

Повтор той же команды продолжает checkpoint и не обрабатывает заново пары с
тем же model/prompt/schema/validation profile. На 200–300 пар переходить следует
только после сравнения этих 50 с ручным baseline v1.3.

## После запуска

Для fact-level очистки нового checkpoint:

```powershell
python scripts/sanitize_qwen_semantic_extraction.py `
  --raw-responses artifacts/qwen_semantic_extraction_v1_4_smoke50/raw_responses.jsonl `
  --dataset data/qwen_semantic_pilot_v1/pilot_inputs.parquet `
  --labels data/qwen_semantic_pilot_v1/pilot_labels.parquet `
  --schema schemas/qwen_semantic_extraction_v1_3.schema.json `
  --prompt prompts/qwen_semantic_extraction_v1_4.md `
  --validation-profile v1_4 `
  --max-pairs 50 `
  --output-dir artifacts/qwen_semantic_extraction_v1_4_sanitized50
```

Sanitizer не вызывает API.
