# Clean-hard label audit

Hard subsets are selected only from targets, canonical duplicates, and fixed
lexical/attribute conflict flags. Model predictions never participate in selection.

## Subsets

| subset | pairs | positives | prevalence | categories | definite_label_conflicts | unordered_id_pair_target_conflicts | representation_pair_target_conflicts | suspicious_negative_identity | suspicious_positive_conflict | sku_vs_human_title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hard_all | 5814 | 1481 | 0.254730 | 18 | 3 | 0 | 3 | 204 | 679 | 622 |
| hard_clean | 4929 | 802 | 0.162710 | 18 | 0 | 0 | 0 | 0 | 0 | 494 |
| hard_suspicious | 882 | 678 | 0.768707 | 18 | 0 | 0 | 0 | 204 | 678 | 128 |
| hard_conflicting | 3 | 1 | 0.333333 | 1 | 3 | 0 | 3 | 0 | 1 | 0 |

## Competition macro AP

| subset | S0 | S2 | S2_CATBOOST |
| --- | --- | --- | --- |
| hard_all | 0.307657 | 0.307319 | 0.314517 |
| hard_clean | 0.221657 | 0.236681 | 0.247444 |
| hard_conflicting | 0.333333 | 0.333333 | 0.333333 |
| hard_suspicious | 0.886120 | 0.831469 | 0.855665 |
| iid | 0.683217 | 0.738074 | 0.736722 |
| ood | 0.503071 | 0.598185 | 0.600112 |

## Flag coverage

| flag | support | positive_count | prevalence |
| --- | --- | --- | --- |
| definite_label_conflict | 3 | 1 | 0.333333 |
| negative_identical_full_representation | 0 | 0 |  |
| negative_exact_normalized_title | 204 | 0 | 0.000000 |
| positive_numeric_conflict | 259 | 259 | 1.000000 |
| positive_code_conflict | 109 | 109 | 1.000000 |
| positive_model_code_conflict | 95 | 95 | 1.000000 |
| positive_critical_attribute_conflict | 403 | 403 | 1.000000 |
| sku_vs_human_title | 622 | 267 | 0.429260 |
| very_high_lexical_similarity_negative | 501 | 0 | 0.000000 |
| very_low_lexical_similarity_positive | 468 | 468 | 1.000000 |

AP values from subsets with different prevalence are not directly comparable.
The reliable comparison is between models within the same subset.
