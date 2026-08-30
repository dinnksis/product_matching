# minilm_5ep_sft_hparam_search_v1

Completed entries: **3/3**.
Primary selection metric: IID macro AP. Hard and OOD are reported only as diagnostics.
`iid_p_holm_stage` uses the full planned hypothesis family, including reserved conditional extensions.
Selection statistics are recomputed from local IID parquets against the frozen stage anchor (the exact control for the initial LR stage).

| experiment | role | epochs | learning_rate | iid_macro_ap | iid_delta | iid_delta_vs_anchor | iid_p_value_vs_anchor | iid_p_holm_stage | hard_macro_ap | ood_macro_ap | training_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm5_sft_e3_lr8e5_v1 | stage_anchor | 3 | 8e-05 | 0.808502 | 0.0191142 | 0 | 1 |  | 0.423286 | 0.647977 | 2966.82 | complete |
| minilm5_sft_drop0_e39bba04_v1 | candidate | 3 | 8e-05 | 0.807657 | 0.0182684 | -0.000845775 | 0.664668 | 0.664668 | 0.418935 | 0.650351 | 2806.28 | complete |
| minilm5_sft_drop0p2_4eb90112_v1 | candidate | 3 | 8e-05 | 0.806706 | 0.0173182 | -0.00179601 | 0.158921 | 0.317841 | 0.422323 | 0.648416 | 2957.34 | complete |

## Stage decisions

```json
{
  "regularization_coordinate_search__classifier_dropout": {
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
    "best_candidate_experiment": "minilm5_sft_drop0_e39bba04_v1",
    "best_candidate_run_id": "65fdfabbeb06489cb38c803c6c569d86",
    "best_candidate_iid_macro_ap": 0.8076565695347291,
    "best_candidate_iid_delta": -0.0008457745089464064,
    "best_candidate_iid_p_holm_stage": 0.6646676661669165,
    "best_candidate_gain_over_anchor": -0.0008457745089464064,
    "challenger_selected": false,
    "practical_shortlist": [
      "minilm5_sft_e3_lr8e5_v1",
      "minilm5_sft_drop0_e39bba04_v1",
      "minilm5_sft_drop0p2_4eb90112_v1"
    ],
    "boundary_extension_evidence": [],
    "needs_boundary_extension": false,
    "recommended_extension": null
  }
}
```
