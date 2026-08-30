# minilm_5ep_sft_hparam_search_v1

Completed entries: **3/3**.
Primary selection metric: IID macro AP. Hard and OOD are reported only as diagnostics.
`iid_p_holm_stage` uses the full planned hypothesis family, including reserved conditional extensions.
Selection statistics are recomputed from local IID parquets against the frozen stage anchor (the exact control for the initial LR stage).

| experiment | role | epochs | learning_rate | iid_macro_ap | iid_delta | iid_delta_vs_anchor | iid_p_value_vs_anchor | iid_p_holm_stage | hard_macro_ap | ood_macro_ap | training_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm5_sft_e3_lr8e5_v1 | stage_anchor | 3 | 8e-05 | 0.808502 | 0.0191142 | 0 | 1 |  | 0.423286 | 0.647977 | 2966.82 | complete |
| minilm5_sft_ls0p05_fa9c6582_v1 | candidate | 3 | 8e-05 | 0.807339 | 0.0179505 | -0.00116376 | 0.65917 | 1 | 0.425097 | 0.647525 | 3115.07 | complete |
| minilm5_sft_ls0p02_721b2785_v1 | candidate | 3 | 8e-05 | 0.806175 | 0.0167872 | -0.00232697 | 0.207896 | 0.623688 | 0.421903 | 0.649656 | 3014.64 | complete |

## Stage decisions

```json
{
  "regularization_coordinate_search__label_smoothing": {
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
    "best_candidate_experiment": "minilm5_sft_ls0p05_fa9c6582_v1",
    "best_candidate_run_id": "9c759f7abbe74a209693766968d449b1",
    "best_candidate_iid_macro_ap": 0.8073385844319304,
    "best_candidate_iid_delta": -0.0011637596117450855,
    "best_candidate_iid_p_holm_stage": 1.0,
    "best_candidate_gain_over_anchor": -0.0011637596117450855,
    "challenger_selected": false,
    "practical_shortlist": [
      "minilm5_sft_e3_lr8e5_v1",
      "minilm5_sft_ls0p05_fa9c6582_v1"
    ],
    "boundary_extension_evidence": [],
    "needs_boundary_extension": false,
    "recommended_extension": null
  }
}
```
