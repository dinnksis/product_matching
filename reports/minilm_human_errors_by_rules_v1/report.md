# Ошибки MiniLM по semantic rules — human-only

## Что сделано

OOF-предсказания MiniLM S2 сопоставлены с label-free semantic assignments Qwen.
Целевые значения везде взяты только из human-разметки. Синтетика и Qwen-labels
не использовались; ordinary, hard и OOD не участвовали в выборе правил.

Из 306669 OOF human-пар semantic extraction покрывает
59184 пар. При диагностическом пороге
0.5: FP = 3103,
FN = 5362, OOF-hard = 7987.

## Самые частые ошибки по правилам

| концепт | каталог | human-пар | FP | FN | OOF-hard |
| --- | --- | --- | --- | --- | --- |
| brand | OTHER_CANDIDATE | 7965 | 548 | 822 | 1334 |
| color | OTHER_CANDIDATE | 5736 | 503 | 767 | 1309 |
| color | CORE | 12159 | 203 | 822 | 689 |
| model_number | OTHER_CANDIDATE | 3597 | 241 | 489 | 755 |
| brand | CORE | 7385 | 248 | 475 | 712 |
| manufacturer_part_number | OTHER_CANDIDATE | 4427 | 212 | 496 | 623 |
| product_line | OTHER_CANDIDATE | 3542 | 226 | 475 | 621 |
| sku | OTHER_CANDIDATE | 4016 | 190 | 408 | 499 |
| product_type | CORE | 4011 | 221 | 360 | 687 |
| country_of_origin | OTHER_CANDIDATE | 4138 | 197 | 329 | 557 |
| package_quantity | OTHER_CANDIDATE | 2497 | 192 | 332 | 516 |
| material | OTHER_CANDIDATE | 2415 | 226 | 286 | 518 |
| package_quantity | CORE | 4998 | 183 | 321 | 541 |
| product_line | CORE | 3224 | 161 | 343 | 533 |
| country_of_origin | OTHER_CANDIDATE | 2416 | 233 | 261 | 496 |

## Готовые редкие правила: достаточно ли human-примеров

| редкое правило | human negative | FP MiniLM | OOF-hard negative | clean human negative | рекомендация |
| --- | --- | --- | --- | --- | --- |
| target_surface | 68 | 14 | 26 | 12 | USE_EXISTING_HUMAN_ERRORS_FIRST |
| target_animal | 145 | 11 | 27 | 30 | USE_EXISTING_HUMAN_ERRORS_FIRST |
| cable_length | 152 | 3 | 11 | 30 | USE_EXISTING_HUMAN_HARD |
| active_ingredient | 55 | 3 | 10 | 15 | USE_EXISTING_HUMAN_HARD |
| design_pattern | 81 | 3 | 3 | 18 | LOW_OBSERVED_MODEL_ERROR |
| sheet_size | 69 | 2 | 7 | 1 | USE_EXISTING_HUMAN_HARD |
| instrument_type | 226 | 2 | 6 | 12 | USE_EXISTING_HUMAN_HARD |
| compatible_model | 281 | 1 | 4 | 74 | LOW_OBSERVED_MODEL_ERROR |
| hook_size | 67 | 1 | 3 | 23 | LOW_OBSERVED_MODEL_ERROR |
| lash_curl | 59 | 1 | 2 | 17 | LOW_OBSERVED_MODEL_ERROR |
| curl_type | 73 | 0 | 2 | 13 | LOW_OBSERVED_MODEL_ERROR |
| needle_diameter | 66 | 0 | 2 | 20 | LOW_OBSERVED_MODEL_ERROR |
| frame_size | 53 | 0 | 2 | 18 | LOW_OBSERVED_MODEL_ERROR |
| axis | 365 | 0 | 1 | 54 | LOW_OBSERVED_MODEL_ERROR |
| cylinder_power | 292 | 0 | 1 | 19 | LOW_OBSERVED_MODEL_ERROR |
| load_index | 168 | 0 | 1 | 4 | LOW_OBSERVED_MODEL_ERROR |
| lash_length | 56 | 0 | 1 | 17 | LOW_OBSERVED_MODEL_ERROR |
| oxygen_permeability | 312 | 0 | 0 | 29 | LOW_OBSERVED_MODEL_ERROR |
| lash_thickness | 50 | 0 | 0 | 12 | LOW_OBSERVED_MODEL_ERROR |
| engraving_text | 30 | 0 | 0 | 2 | LOW_OBSERVED_MODEL_ERROR |
| custom_name | 24 | 0 | 0 | 1 | LOW_OBSERVED_MODEL_ERROR |
| compatible_phone_model | 23 | 0 | 0 | 10 | LOW_OBSERVED_MODEL_ERROR |
| pasta_shape | 23 | 0 | 0 | 7 | LOW_OBSERVED_MODEL_ERROR |
| key_tuning | 19 | 0 | 0 | 7 | LOW_OBSERVED_MODEL_ERROR |

`USE_EXISTING_HUMAN_ERRORS_FIRST` означает, что для первого эксперимента уже есть
минимум пять реальных FP MiniLM: сначала следует дообучаться на них, а не генерировать.
`USE_EXISTING_HUMAN_HARD` означает, что ошибок меньше, но есть минимум пять трудных
human-negative. `HUMAN_DATA_SCARCE` — меньше десяти human-negative в доступных 60k.

## Следующий эксперимент

Сформировать только из human-разметки небольшой targeted train subset: реальные
OOF-ошибки и OOF-hard пары по приоритетным правилам. Затем продолжить fine-tuning
из весов frozen MiniLM S2 baseline и сравнить с тем же baseline на неизменённых
IID/hard/OOD. Qwen может отдельно объяснить выбранные ошибки, но не задаёт label.
