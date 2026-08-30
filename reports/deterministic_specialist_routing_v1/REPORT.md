# Deterministic specialist routing

Маршрутизация не использует IID/Hard/OOD labels. Labels читаются только после фиксации mask для расчёта macro AP.

## Leakage status

Совместимых train/OOF predictions для BGE, MiniLM и RuModernBERT нет. Поэтому ни одна стратегия или blend weight не может быть утверждена как OOF-selected. Результаты ниже — frozen validation benchmark заранее объявленных правил, а не основание для production tuning.

`abs(p-0.5)` и entropy дали одинаковые route masks во всех 15 проверках split × budget.
Empirically difficult score region и specialist slices из Experiment 1 не использовались: они были найдены по IID/Hard/OOD labels и нарушили бы текущий запрет.

## Full-model comparisons

| routing | IID AP | Hard AP | OOD AP | T4 private estimate | runtime vs BGE |
| --- | ---: | ---: | ---: | ---: | ---: |
| full_bge | 0.782222 | 0.375975 | 0.641270 | 78.4 min | 1.000x |
| full_bge_minilm_50_50 | 0.792857 | 0.373480 | 0.641315 | 91.7 min | 1.169x |
| full_bge_rumodernbert_50_50 | 0.788812 | 0.363232 | 0.649401 | 121.8 min | 1.554x |
| full_triple_equal | 0.796635 | 0.366133 | 0.648387 | 135.1 min | 1.723x |

## Predeclared 10% MiniLM protocol candidates

| routing | score | IID AP / delta | Hard AP / delta | OOD AP / delta | runtime vs BGE |
| --- | --- | ---: | ---: | ---: | ---: |
| uncertainty_abs_minilm | replace | 0.785001 / +0.002779 | 0.374913 / -0.001062 | 0.631984 / -0.009286 | 1.017x |
| uncertainty_abs_minilm | blend_50_50 | 0.785959 / +0.003737 | 0.376100 / +0.000125 | 0.639927 / -0.001343 | 1.017x |
| uncertainty_abs_minilm | blend_specialist_25 | 0.784710 / +0.002488 | 0.376228 / +0.000253 | 0.641485 / +0.000215 | 1.017x |
| domain_conflict_minilm | replace | 0.783861 / +0.001639 | 0.375709 / -0.000266 | 0.632617 / -0.008653 | 1.017x |
| domain_conflict_minilm | blend_50_50 | 0.786058 / +0.003836 | 0.376391 / +0.000416 | 0.640476 / -0.000794 | 1.017x |
| domain_conflict_minilm | blend_specialist_25 | 0.784852 / +0.002630 | 0.376342 / +0.000367 | 0.641672 / +0.000402 | 1.017x |

For the predeclared 25% specialist blend, captured Experiment 1 oracle headroom is:

- `uncertainty_abs_minilm`: IID 4.3%, Hard 0.6%, OOD 0.2%.
- `domain_conflict_minilm`: IID 4.5%, Hard 0.8%, OOD 0.4%.

## Runtime interpretation

T4 estimates use the measured one-T4 throughput and 275k private pairs. They are not H100 deadline predictions. The reliable portable quantity is the runtime multiplier relative to BGE; actual end-to-end H100 timing is still required.

## Selection status

Approved strategies: **0**. Two protocol candidates for a future neural OOF run are `uncertainty_abs_minilm` at 10% and `domain_conflict_minilm` at 10%. They are chosen in advance for simplicity and MiniLM runtime, not from the validation AP table. Their score mode remains unresolved until OOF predictions exist.

See `main_table.csv` for every policy/budget/fixed score mode and `routing_results.csv` for split-level rows.
