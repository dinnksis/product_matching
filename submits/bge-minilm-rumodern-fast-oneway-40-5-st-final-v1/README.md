# Final-checkpoint BGE100 + MiniLM40 + RuModernBERT5

This is a separate submission from the previous-checkpoint `40/5` archive.

- BGE scores 100% of pairs;
- the existing 7-feature CatBoost benefit router selects 40% for MiniLM;
- the existing 14-feature sequential router selects 5% for RuModernBERT inside
  the MiniLM subset;
- specialists use successive 50/50 probability blends;
- inference is one-way `A -> B`, SentenceTransformers, FP16, SDPA, batch 1024;
- combined pair `max_length=384`;
- product serialization is copied from final training: category, name and
  deterministic `key: value` attribute lines.

The CatBoost models are intentionally reused for this quick experiment. They were
trained on OOF scores from previous neural checkpoints, so their historical AP
numbers do not transfer to these final checkpoints. This submission tests how well
that routing policy survives the score-distribution change; it is not a freshly
trained OOF router.

The existing runtime image is reused:

```text
dinakepech/ecup26-bge-minilm-rumodern-router-sdpa:1.0
```

Build without overwriting the old `40/5` archive:

```powershell
.\.venv\Scripts\python.exe scripts\build_bge_minilm_rumodern_fast_oneway_final_40_5_submit.py
```
