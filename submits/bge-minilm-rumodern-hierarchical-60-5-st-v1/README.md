# Hierarchical BGE + MiniLM 60% + RuModernBERT 5%

Frozen production-quality candidate from the OOF-safe compute-allocation
experiment:

- BGE symmetric inference on 100%;
- CatBoost MiniLM benefit-router selects 60%;
- MiniLM symmetric inference on that 60%;
- a second CatBoost router, using BGE/MiniLM disagreement, selects a nested 5%;
- RuModernBERT symmetric inference on that 5%;
- both specialist stages use a 50/50 blend with the currently available score.

All cross-encoders use `S2_VALUES_ONLY`, max length 384, SentenceTransformers,
FP16 and SDPA. Frozen validation: IID 0.794034, Hard 0.375256, OOD 0.643439,
mean 0.604243.

This archive preserves the quality experiment exactly. It retains the full
134-feature first router and symmetric inference, so it is expected to be
slower than the previous BGE+MiniLM-40 runtime probe.
