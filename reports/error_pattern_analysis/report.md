# Результаты анализа ошибок

- Split: `component_seed_42`.
- Порог FPR/FNR: `0.5`.
- FP: `2588`, FN: `9160`.
- Hard negatives сохранены: `40759` строк.

- Наибольший наблюдаемый FPR среди поддержанных комбинаций: `brand_conflict + measure_match` = `0.110` при support `1693`.

Полные таблицы: `word_ngram_error_rates.csv`, `char_ngram_error_rates.csv`, `semantic_combination_error_rates.csv`, `top_false_positives.csv`, `top_false_negatives.csv`.