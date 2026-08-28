# BGE / MiniLM / RuModernBERT compute allocation

Routing targets, coverage masks and aggregation weights were frozen on component-disjoint human-train OOF before IID/Hard/OOD evaluation.
The sequential RuModern router never sees the RuModern score; it may use MiniLM only because it is evaluated inside the MiniLM-routed subset.

Absolute runtime is pending new end-to-end H100 measurements. The frontier below is therefore cost-independent: a row is dominated only when another row uses no more MiniLM and no more RuModern coverage while giving at least the same mean AP.

| architecture | Mini | Ru | scheme | Ru target | IID | Hard | OOD | mean | delta vs full BGE+Mini |
| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| hierarchical_m0.60_r0.05 | 60% | 5% | hierarchical | vs_bge_minilm | 0.794034 | 0.375256 | 0.643439 | 0.604243 | +0.001692 |
| hierarchical_m0.60_r0.10 | 60% | 10% | hierarchical | vs_bge_minilm | 0.794140 | 0.374533 | 0.643997 | 0.604223 | +0.001673 |
| hierarchical_m0.40_r0.05 | 40% | 5% | hierarchical | vs_bge_minilm | 0.792534 | 0.376139 | 0.643866 | 0.604180 | +0.001629 |
| hierarchical_m1.00_r0.05 | 100% | 5% | hierarchical | vs_bge_minilm | 0.795804 | 0.373080 | 0.643392 | 0.604092 | +0.001541 |
| hierarchical_m1.00_r0.10 | 100% | 10% | hierarchical | vs_bge_minilm | 0.795528 | 0.372380 | 0.644012 | 0.603973 | +0.001423 |
| exclusive_m0.60_r0.05 | 60% | 5% | exclusive | vs_bge | 0.794191 | 0.375463 | 0.642065 | 0.603906 | +0.001355 |
| hierarchical_m0.40_r0.10 | 40% | 10% | hierarchical | vs_bge_minilm | 0.792380 | 0.375216 | 0.643988 | 0.603861 | +0.001310 |
| hierarchical_m0.20_r0.10 | 20% | 10% | hierarchical | vs_bge | 0.789963 | 0.377376 | 0.643424 | 0.603588 | +0.001037 |
| exclusive_m0.60_r0.10 | 60% | 10% | exclusive | vs_bge | 0.793924 | 0.373697 | 0.642063 | 0.603228 | +0.000677 |
| hierarchical_m0.20_r0.05 | 20% | 5% | hierarchical | vs_bge | 0.789868 | 0.376858 | 0.642889 | 0.603205 | +0.000655 |
| mini_0.40 | 40% | 0% | mini_only | none | 0.790349 | 0.377149 | 0.641795 | 0.603098 | +0.000547 |
| exclusive_m0.40_r0.05 | 40% | 5% | exclusive | vs_bge | 0.792086 | 0.374933 | 0.642217 | 0.603078 | +0.000528 |
| exclusive_m0.40_r0.10 | 40% | 10% | exclusive | vs_bge | 0.793080 | 0.373684 | 0.642296 | 0.603020 | +0.000469 |
| exclusive_m0.20_r0.10 | 20% | 10% | exclusive | vs_bge | 0.790107 | 0.375275 | 0.643577 | 0.602986 | +0.000436 |
| mini_0.60 | 60% | 0% | mini_only | none | 0.791558 | 0.375633 | 0.641320 | 0.602837 | +0.000286 |
| exclusive_m0.20_r0.05 | 20% | 5% | exclusive | vs_bge | 0.790029 | 0.375556 | 0.642115 | 0.602567 | +0.000016 |
| mini_0.20 | 20% | 0% | mini_only | none | 0.788477 | 0.377364 | 0.641849 | 0.602563 | +0.000012 |
| full_bge_minilm | 100% | 0% | baseline | none | 0.792857 | 0.373480 | 0.641315 | 0.602551 | +0.000000 |
| mini_1.00 | 100% | 0% | mini_only | none | 0.792857 | 0.373480 | 0.641315 | 0.602551 | +0.000000 |
| bge_100 | 0% | 0% | baseline | none | 0.782222 | 0.375975 | 0.641270 | 0.599822 | -0.002728 |

## Pareto frontier

mini_0.20, hierarchical_m0.20_r0.05, hierarchical_m0.20_r0.10, mini_0.40, hierarchical_m0.40_r0.05, hierarchical_m0.60_r0.05

## Provisional candidates

- production_quality: `hierarchical_m0.60_r0.05` — mean 0.604243, MiniLM 60%, RuModern 5%.
- reserved_compute: `hierarchical_m0.40_r0.05` — mean 0.604180, MiniLM 40%, RuModern 5%.

Final 13-minute and reserved-runtime choices must be recomputed after filling `runtime_seconds` in the config with the new implementation timings.
