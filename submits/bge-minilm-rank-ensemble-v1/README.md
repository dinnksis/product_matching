# BGE + MiniLM normalized-rank ensemble

Offline two-model submission using the best pair from the frozen architecture
ensemble experiment. It runs the fine-tuned BGE and the five-epoch synthetic
pretrained + human-fine-tuned MiniLM sequentially, then returns the mean of
their normalized ranks.

Both models use shared S2 values-only serialization, `max_length=384`, FP16
autocast and symmetric A-to-B/B-to-A inference. The optimized runner tokenizes
each model in large CPU batches, sorts pairs by their exact token length and
uses dynamic padding with an H100-oriented token budget. BGE runs first and is
released before MiniLM is loaded. There is no fallback, partial inference, soft
deadline or Check bypass.

Default H100 limits are 2,048 pairs / 786,432 pair-tokens for BGE and 4,096
pairs / 1,572,864 pair-tokens for MiniLM. Environment variables with the
`PM_BGE_*`, `PM_MINILM_*` and `PM_TOKENIZATION_BATCH_SIZE` prefixes expose these
technical batch limits without changing model semantics.

The already published runtime image is reused:
`dinakepech/ecup26-bge-reranker-v2-m3:1.0`.

Build the ZIP from the repository root:

```powershell
.venv\Scripts\python.exe scripts\build_bge_minilm_ensemble_submit.py
```

Result: `submits/bge-minilm-rank-ensemble-optimized-v2.zip`.

The builder stages both immutable weight files as NTFS hard links, so the
working submission directory does not consume another 2.7 GB. Only the final
ZIP requires additional disk space.
