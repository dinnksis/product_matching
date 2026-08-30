# minilm_5ep_sft_hparam_search_v1

Completed entries: **3/3**.
Primary selection metric: IID macro AP. Hard and OOD are reported only as diagnostics.
`iid_p_holm_stage` uses the full planned hypothesis family, including reserved conditional extensions.
Selection statistics are recomputed from local IID parquets against the frozen stage anchor (the exact control for the initial LR stage).

| experiment | role | epochs | learning_rate | iid_macro_ap | iid_delta | iid_delta_vs_anchor | iid_p_value_vs_anchor | iid_p_holm_stage | hard_macro_ap | ood_macro_ap | training_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm5_sft_e3_lr8e5_v1 | stage_anchor | 3 | 8e-05 | 0.808502 | 0.0191142 | 0 | 1 |  | 0.423286 | 0.647977 | 2966.82 | complete |
| minilm5_sft_wu0p1_55fd0a4d_v1 | candidate | 3 | 8e-05 | 0.807781 | 0.0183931 | -0.000721115 | 0.617691 | 1 | 0.427943 | 0.646917 | 3168.11 | complete |
| minilm5_sft_wu0_dcd97321_v1 | candidate | 3 | 8e-05 | 0.807545 | 0.0181567 | -0.000957502 | 0.618191 | 1 | 0.422933 | 0.647604 | 3043.25 | complete |

## Stage decisions

```json
{
  "regularization_coordinate_search__warmup_ratio": {
    "expected_runs": 2,
    "completed_runs": 2,
    "complete": true,
    "decision_status": "ready",
    "expected_entries_including_reused_anchor": 3,
    "completed_entries_including_reused_anchor": 3,
    "control_gate": "passed",
    "anchor_reuse": "validated",
    "selection_anchor_experiment": "minilm5_sft_e3_lr8e5_v1",
    "selection_anchor_run_id": "2facdf6ad2b842a8a2f3e4c66410675e",
    "selection_anchor_iid_macro_ap": 0.8085023440436755,
    "recommendation": "retain_current_recipe",
    "recommended_experiment": "minilm5_sft_e3_lr8e5_v1",
    "recommended_run_id": "2facdf6ad2b842a8a2f3e4c66410675e",
    "recommended_iid_macro_ap": 0.8085023440436755,
    "best_candidate_experiment": "minilm5_sft_wu0p1_55fd0a4d_v1",
    "best_candidate_run_id": "5a0d2a97d68c454dbb6cd7cae95cc5e3",
    "best_candidate_iid_macro_ap": 0.8077812288057772,
    "best_candidate_iid_delta": -0.0007211152378983066,
    "best_candidate_iid_p_holm_stage": 1.0,
    "best_candidate_gain_over_anchor": -0.0007211152378983066,
    "challenger_selected": false,
    "practical_shortlist": [
      "minilm5_sft_e3_lr8e5_v1",
      "minilm5_sft_wu0p1_55fd0a4d_v1",
      "minilm5_sft_wu0_dcd97321_v1"
    ],
    "boundary_extension_evidence": [],
    "needs_boundary_extension": false,
    "recommended_extension": null
  }
}
```
