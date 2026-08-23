# Быстрая проверка предсказания label по frozen rules

Модель обучена только на 60k `RULE_DISCOVERY`: L2 logistic regression с
категорией и бинарными признаками замороженных rules с discovery support не
меньше 10. Validation labels не использовались для
обучения, отбора features или выбора threshold.

| метрика | category-only baseline | frozen-rule classifier |
| --- | ---: | ---: |
| ROC-AUC | 0.6845 | 0.8272 |
| Average precision | 0.4149 | 0.6155 |
| Log loss | 0.5287 | 0.4305 |
| Brier | 0.1746 | 0.1378 |
| Accuracy, threshold=0.5 | 0.7530 | 0.8050 |
| F1, threshold=0.5 | 0.1972 | 0.5663 |

Это диагностический интерпретируемый baseline, а не финальный matcher. Его
коэффициенты учитывают совместное появление rules, но не доказывают причинность
каждого отдельного difference. Ordinary, hard и OOD не использовались.
