# MiniLM human-error curriculum v1: результат

## Что проверялось

Из `RULE_DISCOVERY` отобраны 9 311 human-пар: 5 951 OOF-ошибка MiniLM S2 и
3 360 корректно предсказанных, но отмеченных mining-пайплайном как сложные пары.
Их вес в loss увеличен с 1.0 до 2.0, вес остальных human-пар равен 1.0.

Qwen не подавалась в MiniLM и не создавала target. Её label-free semantic
extraction применялась только для анализа и фильтрации curriculum. Поэтому это
эксперимент с перевзвешиванием human-примеров, а не teacher-student distillation.

## Результаты

| Split | Baseline macro AP | Candidate macro AP | Delta | Holm p-value |
| --- | ---: | ---: | ---: | ---: |
| IID | 0.789388 | 0.788033 | -0.001355 | 0.3738 |
| Hard | 0.365501 | 0.366464 | +0.000963 | 0.3948 |
| OOD | 0.642660 | 0.640608 | -0.002052 | 0.0045 |

На IID и hard изменение статистически незначимо. На OOD получено небольшое,
но статистически значимое ухудшение. Равномерное удвоение веса всего curriculum
не рекомендуется продолжать как основную стратегию.

## Утечки

- Curriculum формировался только по `RULE_DISCOVERY`.
- Label в curriculum исключительно human.
- Synthetic pairs и Qwen labels отсутствуют.
- Frozen IID, hard и OOD не участвовали в выборе curriculum и обучении; они
  использовались только для итоговой оценки.

Полные predictions, checkpoint и Kaggle log не хранятся в Git. Они остаются в
локальном `artifacts/` и в output Kaggle kernel. Компактные метрики сохранены в
`summary.json`, а весь эксперимент воспроизводится скриптами и notebook.
