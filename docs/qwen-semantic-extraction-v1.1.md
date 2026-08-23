# Qwen semantic extraction v1.1: smoke-протокол

## Почему появилась v1.1

Остановленный v1 pilot содержит 142 уникальных ответа: 47 schema-valid и 95
ошибок. Наблюдаемые причины не требуют human labels:

- Qwen перечисляла слишком много односторонних attributes и длинных
  multi-value полей, поэтому часть JSON обрезалась;
- `raw_values` иногда был пустым либо превышал лимит массива;
- missing facts и differences попадали не в свои секции;
- identity anchor иногда использовался для различающихся значений;
- один невалидный ответ повторно запрашивался до трёх раз.

V1 и её checkpoint не изменяются. V1.1 запускается в новом каталоге и сначала
только на 50 первых парах того же детерминированного label-free pilot.

## Изменения v1.1

- output ограничен 5 anchors, 8 differences и 3 действительно значимыми missing
  facts;
- запрещён полный diff всех полей и разворачивание длинных составов/списков;
- `raw_values[]` заменён на одну строку `raw_value`;
- для отсутствующей стороны обязательны `value=null` и пустой evidence;
- relation identity anchor зафиксирован как `same`;
- удалены многословные `notes`, сокращены summary/warnings;
- добавлены короткие schema-совместимые примеры anchor, difference и missing;
- retries выполняются только для сетевых/server errors. Полученный невалидный
  ответ сохраняется после одного вызова как `status=invalid` и не повторяется при
  обычном resume.

Human label по-прежнему отсутствует в Qwen payload и загружается только после
завершения всех futures для локального join. Используются только пары
`RULE_DISCOVERY`; internal validation, ordinary, hard и OOD не затрагиваются.

## Smoke-run 50 пар

```powershell
python scripts/run_qwen_semantic_extraction.py `
  --dataset data/qwen_semantic_pilot_v1/pilot_inputs.parquet `
  --labels data/qwen_semantic_pilot_v1/pilot_labels.parquet `
  --prompt prompts/qwen_semantic_extraction_v1_1.md `
  --schema schemas/qwen_semantic_extraction_v1_1.schema.json `
  --prompt-version qwen_semantic_extraction_v1_1 `
  --api-base http://localhost:8194/v1 `
  --model Qwen3.5-397B-A17B-FP8 `
  --workers 2 `
  --max-pairs 50 `
  --max-tokens 4096 `
  --timeout 600 `
  --retries 3 `
  --review-size 50 `
  --output-dir artifacts/qwen_semantic_extraction_v1_1_smoke50
```

Прогресс печатается каждые 10 завершённых пар:

```text
Qwen: 20/50 new; ok=18, invalid=2, request_errors=0, reused=0
```

- `invalid` означает, что сервер ответил, но JSON/schema/semantic validation не
  пройдены; такой ответ не расходует retries;
- `request_errors` означает, что запрос не завершился после всех retries;
- повтор той же команды продолжает только request errors/незавершённые пары и
  переиспользует `ok` и `invalid` с теми же hashes.

После 50 пар эксперимент останавливается для сравнения JSON validity, длины
ответов, missing-vs-mismatch, duplicate facts, source evidence, anchors и ручных
примеров GOOD/BAD/AMBIGUOUS. До этого v1.1 нельзя автоматически запускать на всех
500 или на всём `RULE_DISCOVERY`.

Smoke-run завершён и разобран в
`reports/qwen_semantic_extraction_v1_1_smoke50/analysis.md`. Следующая версия —
v1.2; v1.1 больше не продолжается.
