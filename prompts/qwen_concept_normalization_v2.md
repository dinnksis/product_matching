# Qwen concept normalization v2

Ты нормализуешь уже извлечённые semantic concepts для product matching.

Вход содержит только canonical concept names, категории, relation types и
примеры исходных attribute names. Human labels отсутствуют.

Для каждого entry верни ровно один mapping:

- `KEEP`: concept уже достаточно точный; `target_concept` равен `source_concept`.
- `MERGE`: это очевидный синоним другого `source_concept` из ЭТОГО ЖЕ batch;
  `target_concept` должен точно совпадать с именем этого concept.

Объединяй только одинаковый semantic meaning. Не объединяй просто связанные
понятия. В частности, сохраняй различия:

- `weight`, `net_weight`, `gross_weight`, `package_weight`;
- `model_name`, `model_number`, `model_family`, `model_year`;
- `color`, `frame_color`, `light_color`, `lens_color`;
- `material`, `frame_material`, `upper_material`, `lining_material`;
- размер товара, размер упаковки и compatibility size;
- package quantity и физическую capacity/volume.

Если есть сомнение — `KEEP`. Не создавай новую ontology и не придумывай новый
target. Не анализируй match/non-match. Верни только JSON по schema.
