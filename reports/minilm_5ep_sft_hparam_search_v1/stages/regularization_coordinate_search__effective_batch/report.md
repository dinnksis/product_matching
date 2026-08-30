# minilm_5ep_sft_hparam_search_v1

Completed entries: **3/3**.
Primary selection metric: IID macro AP. Hard and OOD are reported only as diagnostics.
`iid_p_holm_stage` uses the full planned hypothesis family, including reserved conditional extensions.
Selection statistics are recomputed from local IID parquets against the frozen stage anchor (the exact control for the initial LR stage).

| experiment | role | epochs | learning_rate | iid_macro_ap | iid_delta | iid_delta_vs_anchor | iid_p_value_vs_anchor | iid_p_holm_stage | hard_macro_ap | ood_macro_ap | training_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm5_sft_eb384_c9f42551_v1 | candidate | 3 | 8e-05 | 0.809527 | 0.020139 | 0.00102477 | 0.536232 | 1 | 0.420406 | 0.650533 | 2547.2 | complete |
| minilm5_sft_eb96_2c69ea29_v1 | candidate | 3 | 8e-05 | 0.808932 | 0.0195441 | 0.000429915 | 0.867066 | 1 | 0.415009 | 0.645179 | 3384.63 | complete |
| minilm5_sft_e3_lr8e5_v1 | stage_anchor | 3 | 8e-05 | 0.808502 | 0.0191142 | 0 | 1 |  | 0.423286 | 0.647977 | 2966.82 | complete |

## Stage decisions

```json
{
  "regularization_coordinate_search__effective_batch": {
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
    "best_candidate_experiment": "minilm5_sft_eb384_c9f42551_v1",
    "best_candidate_run_id": "491f6bf07f804f1bbf348368ca45765c",
    "best_candidate_iid_macro_ap": 0.809527116012489,
    "best_candidate_iid_delta": 0.0010247719688134538,
    "best_candidate_iid_p_holm_stage": 1.0,
    "best_candidate_gain_over_anchor": 0.0010247719688134538,
    "challenger_selected": false,
    "practical_shortlist": [
      "minilm5_sft_e3_lr8e5_v1",
      "minilm5_sft_eb384_c9f42551_v1",
      "minilm5_sft_eb96_2c69ea29_v1"
    ],
    "boundary_extension_evidence": [],
    "needs_boundary_extension": false,
    "recommended_extension": null
  }
}
```
