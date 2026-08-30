# MiniLM S2 targeted-hard training

Дата запуска: 2026-08-16. Kaggle kernel version 2 завершился со статусом
`COMPLETE` на 2×T4.

## Дизайн

- Модель: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`;
- serialization: S2, title + только values атрибутов;
- только human labels, 306,669 train pairs;
- 3 component/family-disjoint OOF folds;
- hardness выбрана только по OOF score, p85 отдельно для positive/negative;
- из mining исключены 80 definite conflicts и 25,781 strong-suspicion examples;
- 42,125 hard pairs (8,377 positive + 33,748 negative) продублированы один раз;
- итоговая hard exposure fraction: 24.15%;
- остальные model/training parameters не менялись.

## Основные результаты

| model | split | macro AP | ROC-AUC |
|---|---:|---:|---:|
| baseline S2 | IID | 0.738021 | 0.881540 |
| targeted hard S2 | IID | 0.738298 | 0.880372 |
| baseline S2 | hard_clean | 0.236673 | 0.639080 |
| targeted hard S2 | hard_clean | 0.243591 | 0.655995 |
| baseline S2 | OOD | 0.598181 | 0.831771 |
| targeted hard S2 | OOD | 0.600913 | 0.834077 |

Macro-AP deltas (targeted hard minus baseline):

- IID: **+0.000277**;
- hard_clean: **+0.006918**;
- OOD: **+0.002733**.

The predefined success gate passed: hard_clean improved and IID dropped by much
less than 0.01 (in fact it increased slightly).

## Interpretation

The causal result is positive but moderate. A single deterministic x2 oversampling
of OOF-hard, audit-eligible examples improves ranking on the clean hard benchmark
without an IID regression. The effect is not evidence that all hard-test failures
are solved: hard_clean AP remains only 0.244, and the experiment has no repeated
seeds or sampling sweep.

The improvement is ranking-based. No threshold or calibration was changed; on
some score-threshold diagnostics the positive scores moved downward, so those
diagnostics must not be read as classification-threshold gains. Hard-negative
false-positive rates did improve (for example 0.237 → 0.160 on the selected
hard-negative slice), while SKU↔human-title positive ranking improved from AP
0.334 to 0.355.

Artifacts from the run are in the local ignored directory
`artifacts/kaggle/product-matching-minilm-s2-targeted-hard-results2/`.
