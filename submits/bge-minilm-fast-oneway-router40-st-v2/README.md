# Fast one-way BGE + routed MiniLM 40%

Runtime-oriented replacement for the symmetric 134-feature router submission:

- BGE: one AB forward on 100%;
- compact 15-feature CatBoost benefit-router;
- MiniLM: one AB forward on the routed top 40%;
- routed score: 50/50 probability blend;
- `S2_VALUES_ONLY`, max length 384, SentenceTransformers FP16 + SDPA.

The router target is the OOF per-example logloss improvement of MiniLM AB over
BGE AB. It uses only BGE confidence, category and eight cheap title comparisons.
There is no second neural direction and no structured attribute/numeric feature
pipeline.

Frozen validation macro AP: IID 0.788195, Hard 0.377088, OOD 0.639255, mean
0.601513. This is 0.001585 below the symmetric 40% router, traded for roughly
half the neural forwards and removal of the previously dominant CPU feature cost.

The image is the same CatBoost/SentenceTransformers image as v1. Build only the
archive:

```powershell
.venv\Scripts\python.exe scripts\build_bge_minilm_fast_oneway_submit.py
```
