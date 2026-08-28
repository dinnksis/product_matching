# Quick frozen-checkpoint inference benchmark

Run `c9091bc7db92443ca36f5f34c2e1eb78` completed on 2026-08-25 using one Tesla
T4 for measured inference (`2 x T4` were visible), Transformers 4.57.3/PyTorch
2.10, FP16, unchanged `S2_VALUES_ONLY`, symmetric scoring and `max_length=384`.
The notebook completed in 671.66 seconds. It used 5,000 pairs and quick mode;
therefore it did not repeat the already frozen full validation experiment.

## Selected native runners

| model | runner | safe T4 batch | 5k total | pairs/s | baseline | reduction | peak VRAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BGE | pretokenized + exact-length buckets | 32 | 84.94 s | 58.87 | 93.02 s | 8.69% | 3.43 GiB |
| MiniLM | pretokenized + exact-length buckets | 192 | 14.46 s | 345.71 | 16.67 s | 13.23% | 2.28 GiB |
| RuModernBERT | pretokenized + exact-length buckets | 192 | 47.29 s | 105.74 | 73.98 s | 36.08% | 3.41 GiB |

MiniLM was 5.87x faster than BGE and 3.27x faster than RuModernBERT on this
T4 sample. RuModernBERT was 1.80x faster than BGE. These are relative T4
measurements, not H100 deadline predictions.

Raw catalogue reading took 7.92 seconds and exact S2 serialization took 1.58
seconds for 9,978 unique products. BGE and RuModernBERT are forward-bound;
tokenization is not their primary remaining bottleneck. MiniLM has a larger
relative tokenization share, but remains much faster overall.

## Equivalence

All selected runners preserve the saved validation scores:

| model | Pearson | Spearman | mean abs diff | max abs diff | sample AP delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| BGE | 0.999999977 | 0.999999931 | 0.0000117 | 0.002693 | -0.0000038 |
| MiniLM | 1.000000000 | 0.999999999 | 0.000000008 | 0.000029 | 0 |
| RuModernBERT | 0.999999637 | 0.999998784 | 0.0001299 | 0.005142 | +0.0004893 |

The RuModernBERT AP change is a small sample/ranking numerical effect, not a
model-quality improvement. No weights, serialization, max length or ensemble
logic changed.

## Interpretation for submissions

The T4 extrapolations are deliberately not used for a 6/13-minute verdict.
H100 timing must come from the competition Docker runner. The first useful
submission is optimized BGE alone: it has the strongest single-model validation
result and an existing 0.43 leaderboard result, while exact-length bucketing
reduced its T4 inference time by about 9%. If its private H100 time leaves at
least roughly 15-20% headroom, test optimized BGE + MiniLM next. MiniLM is the
only second model cheap enough to be a realistic near-term ensemble addition.

RuModernBERT remains useful for OOD diversity, but it is slower than MiniLM and
the BGE + RuModernBERT pair is weaker overall than BGE + MiniLM. Do not spend a
submission on BGE + RuModernBERT until BGE + MiniLM passes the runtime limit.
The three-model ensemble should wait until a two-model solution has measured
H100 headroom.

The reported T4 batches are safe benchmark settings only. The H100 runner must
use its own large-batch measurement; the quick sweep did not compare every
batch on an identical full sample.

Google Sheets synchronization succeeded: 26 rows were appended to the
`inference_benchmarks` worksheet.
