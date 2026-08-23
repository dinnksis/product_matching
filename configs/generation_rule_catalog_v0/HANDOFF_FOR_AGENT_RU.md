# Передача другому агенту: generation rule catalog v0

## Границы данных

Definitions правил и анализ positive anchors построены только по 60k
`RULE_DISCOVERY`. Stability взята из ранее замороженной проверки на 3k
`rule_internal_validation`; validation не создавала и не меняла definitions.
Ordinary, hard и OOD не читались.

## Машиночитаемые входы

- `generation_rules_v0.json/jsonl`: только исполняемые deterministic actions;
- `generation_rule_v0.schema.json`: JSON Schema одной записи;
- `generation_policy_v0.json`: обязательные preconditions/postconditions;
- `negative_mutation_rules_ready.csv`: 17 interventions для label=0;
- `positive_transformation_rules_ready.csv`: 4 transformations для label=1;
- `positive_anchor_statistics.csv`: identity guards, не самостоятельные labels;
- `anchor_conditioned_difference_statistics.csv`: диагностика; запись исполнима
  только при статусе связанного правила `READY_LABEL_0`;
- `negative_rule_evidence_all.csv`: строгие associations, включая неисполняемые
  строки `STRICT_ASSOCIATION_NEEDS_ISOLATION`.

## Обязательное поведение генератора

Для label=0 применить ровно одну intervention и взять категорию из
`allowed_categories`. Новое значение должно быть валидным для категории;
остальные latent facts сохраняются, а все упоминания в title/attributes
обновляются согласованно. Для label=1 обе записи создаются из одного latent item;
разрешены только surface, equivalent-format, source-redistribution и
information-omission transformations. Positive pair отклоняется, если появился
`different_value`, `incompatible` или source conflict.

Нельзя считать observational probabilities детерминированными labels для
естественных пар. Нельзя расширять definitions по validation/hard/OOD.
Дополнительный Qwen для v0 не требуется; позднее разрешён targeted pass только
по неиспользованным `RULE_DISCOVERY` pairs для конкретных непокрытых concepts.
