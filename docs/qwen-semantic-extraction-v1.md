# Qwen semantic extraction v1: pilot-протокол

## Границы эксперимента

Pilot использует только `rule_discovery`. Запрещено использовать
`rule_internal_validation`, ordinary, hard и OOD как вход Qwen, источник prompt
examples или сигнал выбора canonical concepts.

Qwen не получает human label и не решает задачу match/non-match. Label находится
в отдельном parquet и читается inference-скриптом только после завершения всех
API-запросов.

## Артефакты до inference

- `prompts/qwen_semantic_extraction_v1.md` — versioned extraction prompt;
- `schemas/qwen_semantic_extraction_v1.schema.json` — строгая JSON Schema;
- `data/qwen_semantic_pilot_v1/pilot_inputs.parquet` — 500 label-free пар;
- `data/qwen_semantic_pilot_v1/pilot_labels.parquet` — отдельный local join;
- `data/qwen_semantic_pilot_v1/pilot_sampling_metadata.parquet` — эвристики
  покрытия для последующей ручной диагностики;
- `data/qwen_semantic_pilot_v1/qwen_request_preview.jsonl` — аудит фактических
  Qwen inputs без labels.

Pilot воспроизводится командой:

```powershell
python scripts/create_qwen_extraction_pilot.py --size 500 --seed 2026
```

## Запуск Qwen

```powershell
python scripts/run_qwen_semantic_extraction.py `
  --dataset data/qwen_semantic_pilot_v1/pilot_inputs.parquet `
  --labels data/qwen_semantic_pilot_v1/pilot_labels.parquet `
  --prompt prompts/qwen_semantic_extraction_v1.md `
  --schema schemas/qwen_semantic_extraction_v1.schema.json `
  --prompt-version qwen_semantic_extraction_v1 `
  --api-base http://localhost:8194/v1 `
  --model Qwen3.5-397B-A17B-FP8 `
  --workers 2 `
  --max-pairs 500 `
  --output-dir artifacts/qwen_semantic_extraction_v1_pilot
```

Версия v1 сохранена как диагностический baseline и не должна продолжаться после
остановленного pilot. Для следующего smoke-run используется v1.1 и отдельный
output directory, описанный в `docs/qwen-semantic-extraction-v1.1.md`.

`raw_responses.jsonl` является append-only checkpoint. В актуальном runner
успешные и полученные, но невалидные ответы с теми же model/prompt/schema hashes
повторно не запрашиваются; только request/server errors можно повторить resume.

## Результаты после inference

- `raw_responses.jsonl` — исходные Qwen outputs без human label;
- `parsed_extractions.jsonl` — полный pair-level semantic object с label,
  приклеенным после inference;
- `parsed_extractions.parquet` — плоский индекс и JSON-поля;
- `exploded_semantic_facts.parquet` — дополнительная fact-level таблица, где
  каждая строка сохраняет `pair_id` и полный semantic signature пары;
- `failed_extractions.csv` и `error_statistics.json`;
- `canonical_concept_frequency.csv`;
- `canonical_concept_raw_attribute_mapping.csv` и
  `canonical_mapping_consistency.csv`;
- `identity_anchor_frequency.csv`, `relation_frequency.csv`;
- `manual_review_queue.csv` — очередь для GOOD/BAD/AMBIGUOUS;
- `run_manifest.json` и `run_report.md`.

## Manual review

В `manual_review_queue.csv` нужно заполнить:

- `manual_review_status`: `GOOD`, `BAD` или `AMBIGUOUS`;
- `manual_review_notes`: конкретная причина.

Проверяются сценарии:

1. title-only и attribute-only facts;
2. один fact с evidence одновременно из title и attribute;
3. missing против ошибочного `different_value`;
4. conflicting title/attribute sources;
5. sparse descriptions;
6. multi-difference pairs;
7. hallucinated raw fragments или attribute names;
8. слишком общие или слишком раздробленные canonical concepts;
9. strength/type identity anchors;
10. отсутствие match/non-match verdict в ответе.

Human label можно видеть при ручном анализе пары, но нельзя использовать как
целевую переменную для настройки extraction prompt.

## Правила решения о следующей версии

До просмотра pilot не создаётся несколько конкурирующих ontologies. Возможные
изменения v1.1 принимаются только по наблюдаемым extraction errors:

- duplicate title/attribute facts → уточнить правило объединения evidence;
- missing-vs-mismatch → добавить минимальные schema examples для missing;
- один raw name распадается на много concepts → добавить небольшой
  category-aware glossary, полученный только из rule_discovery;
- разные raw names стабильно обозначают один concept → закрепить проверенный alias;
- hallucinated evidence → ужесточить source-span contract;
- неверные identity anchors → уточнить manufacturer/SKU/model hierarchy;
- низкая JSON validity → упростить schema, не меняя semantic taxonomy.

После pilot запуск на всём `rule_discovery` запрещён до отдельного решения.
