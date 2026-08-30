# MiniLM coverage sweep

All routing masks and blend weights were selected on human-train neural OOF. IID/Hard/OOD labels were loaded only after policy selection.

Absolute runtime is intentionally omitted: the final production implementation uses SentenceTransformers, SDPA, FP16, max_length=192, batch=1024 and one-direction inference, while the frozen quality predictions use the earlier 384-token symmetric protocol.

| coverage | MiniLM weight | IID | Hard | OOD | mean | delta vs full mean | MiniLM stage saved |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5% | 75% | 0.785554 | 0.378029 | 0.639541 | 0.601041 | -0.001510 | 95% |
| 10% | 50% | 0.787380 | 0.377306 | 0.640971 | 0.601886 | -0.000665 | 90% |
| 15% | 50% | 0.787547 | 0.377739 | 0.642038 | 0.602441 | -0.000110 | 85% |
| 20% | 50% | 0.788477 | 0.377364 | 0.641849 | 0.602563 | +0.000012 | 80% |
| 30% | 50% | 0.790447 | 0.376574 | 0.641362 | 0.602795 | +0.000244 | 70% |
| 40% | 50% | 0.790349 | 0.377149 | 0.641795 | 0.603098 | +0.000547 | 60% |
| 50% | 50% | 0.791406 | 0.376180 | 0.641388 | 0.602991 | +0.000441 | 50% |
| 70% | 50% | 0.792591 | 0.374380 | 0.641720 | 0.602897 | +0.000346 | 30% |
| 100% | 50% | 0.792857 | 0.373480 | 0.641315 | 0.602551 | +0.000000 | 0% |

## Operating points

- mean_ap, loss <= 0.001: 10% coverage, mean delta -0.000665, worst split -0.005477.
- every_split, loss <= 0.001: 70% coverage, mean delta +0.000346, worst split -0.000267.
- mean_ap, loss <= 0.002: 5% coverage, mean delta -0.001510, worst split -0.007304.
- every_split, loss <= 0.002: 50% coverage, mean delta +0.000441, worst split -0.001452.

Pareto learned coverages: 5%, 10%, 15%, 20%, 30%, 40%

Runtime conversion after final H100 logs: T = T_fixed + T_BGE + T_MiniLM_load + coverage * T_MiniLM_inference.
