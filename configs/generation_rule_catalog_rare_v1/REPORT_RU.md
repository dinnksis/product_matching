# Редкие правила генерации — v1

## Результат

Из frozen-таблиц отобрано **24 редких безопасных правил** и
**39 экспериментальных кандидатов**. Безопасные правила дают
**29** разрешённых комбинаций
`правило × категория` сверх основного каталога v0.

Ordinary, hard и OOD не читались. Определения правил не менялись по validation:
internal validation использована только для проверки направления и наличия clean-support.

## Что означает RARE_SAFE

- support в 60k RULE_DISCOVERY находится в диапазоне 20–500;
- relation — только `different_value`;
- направление относительно category baseline отрицательное в discovery и validation;
- есть минимум 5 discovery-пар с максимум двумя различиями;
- есть минимум 1 такая validation-пара;
- концепт допускает контролируемое однозначное вмешательство;
- правило применяется только в перечисленных категориях.

Это не означает, что правило детерминированно классифицирует произвольные естественные
пары. Label=0 допустим только при генерации из одного исходного item: меняется ровно
один явный факт, остальные latent facts сохраняются, а title и attributes обновляются
согласованно.

## Готовые редкие правила

| концепт | семейство | discovery support | validation support | clean validation | разрешённые категории |
| --- | --- | --- | --- | --- | --- |
| key_tuning | instrument_spec | 20 | 4 | 4 | Музыкальные инструменты |
| pasta_shape | consumable_variant | 24 | 3 | 1 | Продукты питания |
| compatible_phone_model | compatibility_target | 25 | 3 | 1 | Электроника |
| custom_name | personalization | 34 | 3 | 2 | Канцелярские товары |
| engraving_text | personalization | 38 | 3 | 1 | Галантерея и аксессуары |
| lash_thickness | beauty_variant | 51 | 4 | 1 | Красота и гигиена |
| lash_curl | beauty_variant | 59 | 5 | 1 | Красота и гигиена |
| lash_length | beauty_variant | 59 | 5 | 3 | Красота и гигиена |
| hook_size | dimensional_spec | 72 | 3 | 1 | Спорт и отдых |
| needle_diameter | dimensional_spec | 72 | 6 | 4 | Хобби и творчество |
| curl_type | beauty_variant | 73 | 5 | 1 | Красота и гигиена |
| sheet_size | dimensional_spec | 83 | 3 | 1 | Дом и сад |
| target_surface | target_compatibility | 102 | 5 | 2 | Бытовая химия |
| active_ingredient | composition_identity | 120 | 7 | 2 | Красота и гигиена, Спорт и отдых |
| design_pattern | categorical_variant | 141 | 7 | 1 | Хобби и творчество, Электроника |
| frame_size | dimensional_spec | 149 | 10 | 4 | Детские товары, Хобби и творчество |
| cable_length | dimensional_spec | 170 | 7 | 2 | Строительство и ремонт, Хобби и творчество, Электроника |
| load_index | vehicle_spec | 177 | 13 | 1 | Автотовары |
| target_animal | target_compatibility | 192 | 13 | 2 | Товары для животных |
| instrument_type | instrument_spec | 242 | 14 | 2 | Музыкальные инструменты |
| cylinder_power | optical_prescription | 294 | 13 | 2 | Аптека |
| oxygen_permeability | optical_prescription | 324 | 16 | 2 | Аптека |
| compatible_model | compatibility_target | 333 | 11 | 1 | Электроника |
| axis | optical_prescription | 368 | 14 | 2 | Аптека |

## Что осталось экспериментальным

Кандидаты в `rare_negative_rules_experimental.csv` не потеряны. Они разделены на:

- недостаточно clean-support на internal validation;
- неоднозначную семантику (упаковка, общий размер, материал, ориентация и т. п.);
- отсутствие надёжного category scope;
- алиасы уже существующих core-правил.

Их не следует смешивать с `RARE_SAFE` при автоматическом назначении label=0.
