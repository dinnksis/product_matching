# Full one-way BGE + MiniLM + RuModernBERT

Each model runs one AB forward on 100% of pairs. The final score is the equal-weight mean of the three one-way probabilities. No CatBoost or routing is used.

Serialization is `S2_VALUES_ONLY`, maximum length 384, FP16 SDPA and batch size 1024. The 1/3, 1/3, 1/3 blend was fixed from human-train OOF only.
