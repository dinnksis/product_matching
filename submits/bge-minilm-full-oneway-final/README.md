# Final full one-way BGE + MiniLM

- final BGE and MiniLM checkpoints score 100% of pairs;
- one `A -> B` pass, no symmetric pass;
- SentenceTransformers CrossEncoder, FP16, SDPA, batch 1024;
- combined pair `max_length=384`;
- final score is `0.5 * sigmoid(BGE logit) + 0.5 * sigmoid(MiniLM logit)`;
- no RuModernBERT, CatBoost or routing;
- product serialization matches final training: category, name and deterministic
  `key: value` attribute lines.

The runtime image is intentionally the same image already exercised by the passing
full-triple submission. Unused CatBoost/ModernBERT packages in that image are not
imported by this runner.

Build:

```powershell
.\.venv\Scripts\python.exe scripts\build_bge_minilm_full_oneway_final_submit.py
```

Output: `submits/bge-minilm-full-oneway-final.zip`.
