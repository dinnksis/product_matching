# minilm_5ep_sft_hparam_search_v1

Completed entries: **4/4**.
Primary selection metric: IID macro AP. Hard and OOD are reported only as diagnostics.
`iid_p_holm_stage` uses the full planned hypothesis family, including reserved conditional extensions.
Selection statistics are recomputed from local IID parquets against the frozen stage anchor (the exact control for the initial LR stage).

| experiment | role | epochs | learning_rate | iid_macro_ap | iid_delta | iid_delta_vs_anchor | iid_p_value_vs_anchor | iid_p_holm_stage | hard_macro_ap | ood_macro_ap | training_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm5_sft_e3_lr8e5_v1 | candidate | 3 | 8e-05 | 0.808502 | 0.0191142 | 0.0104234 | 0.00149925 | 0.0029985 | 0.423286 | 0.647977 | 2966.82 | complete |
| minilm5_sft_e2_lr8e5_v1 | candidate | 2 | 8e-05 | 0.806157 | 0.0167692 | 0.00807838 | 0.0009995 | 0.0029985 | 0.403571 | 0.65556 | 1839.07 | complete |
| minilm5_sft_e4_lr8e5_v1 | candidate | 4 | 8e-05 | 0.805777 | 0.0163885 | 0.0076977 | 0.063968 | 0.063968 | 0.431732 | 0.642794 | 4053.39 | complete |
| minilm5_sft_learningrate8m5_0797e754_v1 | stage_anchor | 1 | 8e-05 | 0.798079 | 0.00869078 | 0 | 1 |  | 0.378718 | 0.653329 | 910.951 | complete |

## Stage decisions

```json
{
  "epoch_line": {
    "expected_runs": 3,
    "completed_runs": 3,
    "complete": true,
    "decision_status": "ready",
    "expected_entries_including_reused_anchor": 4,
    "completed_entries_including_reused_anchor": 4,
    "control_gate": "passed",
    "anchor_reuse": "validated",
    "selection_anchor_experiment": "minilm5_sft_learningrate8m5_0797e754_v1",
    "selection_anchor_run_id": "436af881fa5b4da28a5ce9187d44b5cf",
    "selection_anchor_iid_macro_ap": 0.7980789112007961,
    "recommendation": "select_challenger",
    "recommended_experiment": "minilm5_sft_e3_lr8e5_v1",
    "recommended_run_id": "2facdf6ad2b842a8a2f3e4c66410675e",
    "recommended_iid_macro_ap": 0.8085023440436755,
    "best_candidate_experiment": "minilm5_sft_e3_lr8e5_v1",
    "best_candidate_run_id": "2facdf6ad2b842a8a2f3e4c66410675e",
    "best_candidate_iid_macro_ap": 0.8085023440436755,
    "best_candidate_iid_delta": 0.010423432842879388,
    "best_candidate_iid_p_holm_stage": 0.0029985007496251873,
    "best_candidate_gain_over_anchor": 0.010423432842879388,
    "challenger_selected": true,
    "practical_shortlist": [
      "minilm5_sft_e3_lr8e5_v1"
    ],
    "boundary_extension_evidence": [],
    "needs_boundary_extension": false,
    "recommended_extension": null
  }
}
```
