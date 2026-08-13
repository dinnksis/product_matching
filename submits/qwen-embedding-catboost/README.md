# Qwen embedding + CatBoost submission

Offline inference for the best first-round experiment (`03_names_qwen_attributes`,
validation macro AP 0.5748). The builder adds the pinned Qwen snapshot, selected
train-only attribute keys, and CatBoost model. The runner embeds only products
referenced by test pairs. Inference uses the same `SentenceTransformer.encode`
path, float16 dtype, last-token pooling, truncation and L2 normalization as the
training pipeline. There is deliberately no silent fallback: incomplete Qwen
inference must fail rather than emit predictions from a different algorithm.

Before rebuilding the runtime image, run:

```powershell
python scripts/download_embedding_runtime_assets.py --validate-only
```

The snapshot must include the SentenceTransformer module and pooling configs,
not only the base Transformers files.
