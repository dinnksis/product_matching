# BGE + MiniLM + RuModernBERT fast hierarchical submission

- BGE: one AB forward on 100% of pairs.
- MiniLM: one AB forward on the top 40% selected by a 7-feature CatBoost router.
- RuModernBERT: one AB forward on the top 5% inside the MiniLM subset, selected by a 14-feature CatBoost router.
- Both specialists use a 50/50 blend with the currently available score.
- Serialization: `S2_VALUES_ONLY`; maximum length 384; FP16 SDPA; batch size 1024.
- The routers use only category and already available neural scores/confidence/disagreement. No attributes or fuzzy title features are computed for routing.

The neural checkpoints are unchanged. Both routers were trained from component-disjoint one-way OOF predictions on human train; IID, Hard and OOD were evaluation-only.
