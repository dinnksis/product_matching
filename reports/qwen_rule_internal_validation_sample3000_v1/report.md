# Internal validation замороженных Qwen rules

Definitions были созданы только на `RULE_DISCOVERY 60k` и не менялись после
просмотра internal-validation labels.

- validation pairs: **3000**;
- label-free semantic assignments: **11550**;
- assignments, совпавшие с frozen registry: **11202**
  (97.0%);
- validation-only formulations: **348** — сохранены только как coverage
  diagnostics и не превращены в новые rules;
- frozen rules с validation support: **1951**.

## Stability: robust directional discovery rules

| результат | count |
| --- | ---: |
| SAME_DIRECTION_WEAK | 217 |
| REPLICATED_95 | 36 |
| OPPOSITE_DIRECTION | 31 |

`REPLICATED_95` означает совпадение discovery-класса и validation-класса при
заранее зафиксированной статистической политике. `SAME_DIRECTION_WEAK` означает
совпавшее направление posterior median, но недостаточную validation certainty.
`OPPOSITE_DIRECTION` здесь означает только смену знака posterior median. Число
robust rules со статистически подтверждённым противоположным 95% эффектом:
**0**. Definition правила
при этом не меняется.

Для следующего этапа сохранены **30** строго подтверждённых
rule candidates и **52** предварительно стабильных
rule candidates с validation support не меньше 10. `CONTEXT_ONLY` в эти списки
не включается.

Ordinary, hard и OOD не использовались.
