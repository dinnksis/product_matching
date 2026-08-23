# Аудит train/validation split

## Зачем проводится аудит

Текущий names-only lexical CatBoost полностью воспроизводит локальную
component-disjoint validation:

- локальный macro AP: `0.501538`;
- leaderboard: около `0.21`;
- максимальная разница между predictions исходного training pipeline и
  submission runner на локальной validation: `1.1e-16`.

существующая validation, вероятно, не воспроизводит распределение закрытых
retrieval-кандидатов.


## Что проверяет notebook

Основной notebook:
`notebooks/03_validation_split_audit.ipynb`.

Вычислительная логика:
`src/validation_audit.py`.

Результаты после исполнения:
`reports/validation_split_audit/`.

Notebook сравнивает одну и ту же lexical CatBoost при одинаковой примерно 15%
validation:

1. текущий component split с seed `13`, `42`, `77`, `2026`;
2. split, где pair-компоненты дополнительно объединены по точному
   `category + normalized name`;
3. split, где компоненты объединены по консервативной сигнатуре
   `category + brand + model/article`;
4. seller/source holdout, только если seller-поля имеют достаточное покрытие.

`component_seed_42` в notebook буквально воспроизводит существующую функцию
`src.data_pipeline.component_split`, а не является новой похожей реализацией.

## Параллельные представления товара

Исходное название сохраняется без изменений. Дополнительно строятся:

- Unicode NFKC;
- lowercase и `ё → е`;
- аккуратная очистка пунктуации и пробелов;
- разделение букв и цифр: `iphone15pro → iphone 15 pro`;
- отдельные словесные токены;
- числа;
- буквенно-цифровые идентификаторы;
- нормализованные измерения (`0.5 л → volume_ml:500`);
- значения семейств brand, model/SKU, seller и color.

Числа и идентификаторы не удаляются. Для воспроизводимости baseline восемь
лексических признаков CatBoost считаются по старой нормализации эксперимента 01;
новые представления используются для анализа групп и сложности validation.

## Что будет в итоговом отчёте

После исполнения notebook автоматически создаст:

- `split_comparison.csv` — macro/overall AP и размеры splits;
- `representation_overlap.csv` — доля brand/model/seller/name значений из
  validation, уже встречавшихся в train;
- `<split>_predictions.parquet` — честные validation predictions;
- `<split>_report.json` и `<split>.cbm`;
- `report.md` — автоматически собранная таблица;
- `manifest.json` и `COMPLETED`.

## Как выбрать development и final holdout

Решение принимается после получения чисел, а не заранее.

Development split должен:

- сохранять все 20 категорий и достаточное число позитивов;
- исключать пересечение товарных ID;
- заметно уменьшать повторение name/model/source шаблонов;
- быть достаточно большим для устойчивого macro AP;
- давать объяснимое, а не искусственно низкое качество.

Final holdout должен быть отдельным grouped split с другим seed и не
использоваться при выборе признаков. Его результат смотрится только после выбора
подхода на development.

Hard-negative subset не может заменить основную validation: он специально
меняет распределение классов и сложности. Он используется только как stress-test.

## Текущий статус рекомендаций

Оба notebook выполнены 13 августа 2026 года. Получены следующие macro AP для
одной и той же names-only lexical CatBoost:

| Split | Validation pairs | Macro AP |
|---|---:|---:|
| `component_seed_13` | 54 950 | 0.514904 |
| `brand_model_family_holdout` | 53 676 | 0.508698 |
| `component_seed_42` | 54 887 | 0.504450 |
| `exact_normalized_name_holdout` | 50 681 | 0.503171 |
| `component_seed_77` | 54 798 | 0.500019 |
| `component_seed_2026` | 55 377 | 0.498167 |

Разброс между четырьмя обычными component seeds равен примерно `0.017` macro
AP. Это заметно для сравнения близких экспериментов, но не объясняет падение с
локальных `~0.50` до leaderboard `~0.21`.

Во всех сценариях пересечение item ID равно нулю. В обычном component split
train уже содержит около 13% нормализованных названий, 10% model/id и 79–80%
брендов, встречающихся в validation. При `brand_model_family_holdout` доля
повторившихся family signatures равна нулю, но macro AP не падает. Значит,
главная проблема не похожа на простой leakage по ID, точному названию или
brand/model family.

Рекомендация:

- использовать `brand_model_family_holdout` как development split для выбора
  признаков: он жёстче контролирует повторение товарных семейств;
- зафиксировать `component_seed_2026` как отдельный final holdout и не смотреть
  на него при каждом подборе признаков;
- дополнительно показывать среднее и стандартное отклонение по component seeds
  для финальных кандидатов, если разница моделей меньше `0.01–0.02` macro AP.

Human-labelled данные подходят для относительного сравнения подходов и анализа
ошибок. Их текущая локальная метрика не является надёжной оценкой leaderboard:
вероятнее всего, скрытые retrieval-кандидаты отличаются по сложности или
механизму отбора. Увеличение одной validation-выборки это расхождение само по
себе не исправит.
