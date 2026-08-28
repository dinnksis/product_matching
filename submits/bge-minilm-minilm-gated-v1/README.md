# Fast MiniLM-gated BGE ensemble

This submission runs the frozen MiniLM checkpoint on every pair and sends only
pairs with `MiniLM probability > 0.0602078252` to the frozen BGE checkpoint.
Rejected pairs retain the MiniLM probability; routed pairs receive the mean of
MiniLM and BGE probabilities.

The gate is the ordinary-validation MiniLM median and is fixed for all data.
It routes 50.0% of ordinary, 56.6% of hard, and 46.6% of OOD validation pairs
to BGE. Relative to full BGE+MiniLM mean probability, macro AP changes by
-0.0010, -0.0003, and -0.0021 respectively.

Both models retain S2 values-only serialization, `max_length=384`, symmetric
inference, FP16/SDPA, exact-length bucketing and dynamic padding. There is no
deadline fallback or partial output.

Build from the repository root:

```powershell
.venv\Scripts\python.exe scripts\build_bge_minilm_gated_submit.py
```

Output: `submits/bge-minilm-minilm-gated-v1.zip`.
