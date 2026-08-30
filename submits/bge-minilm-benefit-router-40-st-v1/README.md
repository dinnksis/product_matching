# BGE + CatBoost-routed MiniLM 40% (SentenceTransformers)

Runtime experiment that preserves the protocol used to train the frozen
MiniLM benefit router:

- BGE on 100% of pairs;
- frozen CatBoost classification benefit-router selects exactly the top 40%;
- MiniLM only on routed pairs;
- `S2_VALUES_ONLY`, `max_length=384`, symmetric AB/BA probabilities;
- routed score is `0.5 * BGE + 0.5 * MiniLM`;
- both cross-encoders use SentenceTransformers 5.1.2, FP16 and SDPA.

No model or router is retrained. The router uses the same label-free attribute
concept map and the same 134-feature contract as its OOF experiment. Its full
cheap-feature extraction is deliberately retained for quality equivalence;
earlier measurements suggest this CPU stage may cost roughly seven minutes on
the private pair count, so this archive is an end-to-end runtime probe.

Build the derived image and submission archive from the repository root:

```powershell
docker build --platform linux/amd64 -f docker\bge-minilm-benefit-router-sdpa\Dockerfile -t dinakepech/ecup26-bge-minilm-benefit-router-sdpa:1.0 .
docker push dinakepech/ecup26-bge-minilm-benefit-router-sdpa:1.0
.venv\Scripts\python.exe scripts\build_bge_minilm_benefit_router_submit.py
```
