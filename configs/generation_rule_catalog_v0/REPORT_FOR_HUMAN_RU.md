# Готовые правила для генерации пар — v0

## Решение

Дополнительный запуск Qwen перед первой генерацией **не нужен**. На 60k
`RULE_DISCOVERY` уже есть **15515 positive** и
**44485 negative** пар. После проверки на отдельной выборке
получено **28** строгих negative associations; из них
**17** имеют достаточную поддержку в относительно чистых парах и
разрешены для контролируемой генерации label=0.

Во всех **18** current-train категориях наблюдаются готовые
negative rules; минимум на категорию — **4**.

## Как получать однозначный label

- `label=0`: взять структурированный item, изменить ровно один concept из
  `negative_mutation_rules_ready.csv` на другое валидное значение, не менять
  остальные latent facts и синхронно обновить title/attributes.
- `label=1`: создать два представления одного latent item и применять только
  операции из `positive_transformation_rules_ready.csv`. Явные semantic values
  изменять нельзя.

Это обеспечивает label **по конструкции генератора**. Статистические rules сами
по себе не дают логической гарантии label для произвольной естественной пары.

## Позитивы и identity anchors

Хотя бы один anchor найден в **66.2%** positive-пар.
Anchor сохраняется как guard/context: даже одинаковый SKU, MPN или exact model не
является самостоятельной гарантией match, если присутствует критическое
различие. Missing information также никогда не является mismatch.

### Статистика identity anchors

| anchor | сила | support | positive | P(match) |
| --- | --- | --- | --- | --- |
| brand | weak | 27312 | 7821.0 | 0.286 |
| exact_model | strong | 4834 | 2629.0 | 0.544 |
| exact_sku | strong | 111 | 97.0 | 0.874 |
| manufacturer_part_number | strong | 432 | 375.0 | 0.868 |
| model_family | medium | 1555 | 250.0 | 0.161 |
| product_line | medium | 6614 | 3024.0 | 0.457 |

Одинаковый MPN/SKU даёт сильный сигнал, но не 100% гарантию на естественных
парах. Для synthetic label=1 гарантия возникает из общего latent item, а anchors
служат проверкой сохранения identity.

## Готовые negative mutations

| concept | scope | discovery support | validation support | категорий |
| --- | --- | --- | --- | --- |
| optical_power | CATEGORY_SPECIFIC | 1782 | 75 | 1 |
| model_number | GLOBAL | 7366 | 370 | 16 |
| net_weight | GLOBAL | 1176 | 64 | 6 |
| package_weight | GLOBAL | 503 | 25 | 5 |
| model_name | GLOBAL_WITH_EXCEPTIONS | 1741 | 98 | 16 |
| brand | GLOBAL | 7385 | 356 | 16 |
| color | GLOBAL | 12159 | 607 | 18 |
| volume | GLOBAL | 920 | 35 | 9 |
| width | GLOBAL | 2070 | 111 | 9 |
| product_line | GLOBAL_WITH_EXCEPTIONS | 3224 | 136 | 16 |
| model_family | GLOBAL | 419 | 26 | 10 |
| length | GLOBAL | 1151 | 69 | 10 |
| manufacturer_part_number | GLOBAL_WITH_EXCEPTIONS | 2744 | 149 | 11 |
| scent | GLOBAL | 548 | 29 | 4 |
| package_quantity | GLOBAL_WITH_EXCEPTIONS | 4998 | 267 | 15 |
| product_type | GLOBAL_WITH_EXCEPTIONS | 4011 | 183 | 15 |
| color_code | GLOBAL | 628 | 28 | 5 |

Полный список разрешённых категорий для каждого правила хранится в
`negative_mutation_rules_ready.csv` и `ready_rule_category_scope.csv`.

## Что пока не использовать

- rules со статусом `STRICT_ASSOCIATION_NEEDS_ISOLATION`;
- provisional/uncertain/heterogeneous rules;
- positive statistical associations как автоматический label=1;
- категории, отсутствующие в `allowed_categories`;
- hard и OOD для дополнения или исправления catalog.
