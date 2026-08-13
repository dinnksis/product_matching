# Результаты аудита validation split

- Текущий component seed 42: macro AP `0.504450`.
- Самый сложный проверенный split: `component_seed_2026`, macro AP `0.498167`.

## Полная таблица

| split                         | notes                                                                         |   train_pairs |   validation_pairs |   validation_fraction |   train_positive_rate |   validation_positive_rate |   overlapping_item_ids |   macro_average_precision |   overall_average_precision |   best_iteration | catboost_backend   |
|:------------------------------|:------------------------------------------------------------------------------|--------------:|-------------------:|----------------------:|----------------------:|---------------------------:|-----------------------:|--------------------------:|----------------------------:|-----------------:|:-------------------|
| component_seed_13             | Exact reproduction of the existing connected-component split.                 |        310704 |              54950 |              0.150279 |              0.256466 |                   0.258508 |                      0 |                  0.514904 |                    0.600664 |             1199 | CPU                |
| brand_model_family_holdout    | Pair components merged by conservative category + brand + model/id signature. |        311978 |              53676 |              0.146795 |              0.256422 |                   0.258812 |                      0 |                  0.508698 |                    0.596796 |             1199 | CPU                |
| component_seed_42             | Exact reproduction of the existing connected-component split.                 |        310767 |              54887 |              0.150106 |              0.256662 |                   0.257402 |                      0 |                  0.50445  |                    0.596221 |             1192 | CPU                |
| exact_normalized_name_holdout | Pair components merged by exact category + normalized name.                   |        314973 |              50681 |              0.138604 |              0.255235 |                   0.266333 |                      0 |                  0.503171 |                    0.600905 |             1199 | CPU                |
| component_seed_77             | Exact reproduction of the existing connected-component split.                 |        310856 |              54798 |              0.149863 |              0.257109 |                   0.254863 |                      0 |                  0.500019 |                    0.589266 |             1188 | CPU                |
| component_seed_2026           | Exact reproduction of the existing connected-component split.                 |        310277 |              55377 |              0.151446 |              0.257167 |                   0.254564 |                      0 |                  0.498167 |                    0.580493 |             1198 | CPU                |

Это автоматически созданный отчёт. Интерпретацию и финальную рекомендацию нужно перенести в `docs/validation-split-audit.md` после просмотра overlaps и ошибок.