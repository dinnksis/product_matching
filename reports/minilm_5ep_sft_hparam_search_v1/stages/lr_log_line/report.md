# minilm_5ep_sft_hparam_search_v1

Completed entries: **5/5**.
Primary selection metric: IID macro AP. Hard and OOD are reported only as diagnostics.
`iid_p_holm_stage` uses the full planned hypothesis family, including reserved conditional extensions.
Selection statistics are recomputed from local IID parquets against the frozen stage anchor (the exact control for the initial LR stage).

| experiment | role | epochs | learning_rate | iid_macro_ap | iid_delta | iid_delta_vs_anchor | iid_p_value_vs_anchor | iid_p_holm_stage | hard_macro_ap | ood_macro_ap | training_seconds | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| minilm5_sft_learningrate8m5_0797e754_v1 | candidate | 1 | 8e-05 | 0.798079 | 0.00869078 | 0.00844532 | 0.0009995 | 0.001999 | 0.378718 | 0.653329 | 910.951 | complete |
| minilm5_sft_e1_lr4e5_v1 | candidate | 1 | 4e-05 | 0.796411 | 0.0070228 | 0.00677735 | 0.00049975 | 0.001999 | 0.373749 | 0.647103 | 932.584 | complete |
| minilm5_sft_e1_lr2e5_control_v1 | current_protocol_control | 1 | 2e-05 | 0.789634 | 0.000245454 | 0 | 1 |  | 0.365776 | 0.642146 | 1029.5 | complete |
| minilm5_sft_e1_lr1e5_v1 | candidate | 1 | 1e-05 | 0.779403 | -0.00998545 | -0.0102309 | 0.00049975 | 0.001999 | 0.359717 | 0.630707 | 1007.25 | complete |
| minilm5_sft_e1_lr5e6_v1 | candidate | 1 | 5e-06 | 0.767937 | -0.0214511 | -0.0216966 | 0.00049975 | 0.001999 | 0.353825 | 0.617081 | 954.064 | complete |

## Stage decisions

```json
{
  "lr_log_line": {
    "expected_runs": 4,
    "completed_runs": 4,
    "complete": true,
    "decision_status": "ready",
    "expected_entries_including_reused_anchor": 5,
    "completed_entries_including_reused_anchor": 5,
    "control_gate": "passed",
    "control_experiment": "minilm5_sft_e1_lr2e5_control_v1",
    "control_checks": {
      "iid_delta": true,
      "hard_delta": true,
      "ood_delta": true,
      "runtime_drift": true
    },
    "control_runtime_relative_drift": 0.052960469073847216,
    "recommendation": "select_challenger",
    "recommended_experiment": "minilm5_sft_learningrate8m5_0797e754_v1",
    "recommended_run_id": "436af881fa5b4da28a5ce9187d44b5cf",
    "recommended_iid_macro_ap": 0.7980789112007961,
    "best_candidate_experiment": "minilm5_sft_learningrate8m5_0797e754_v1",
    "best_candidate_run_id": "436af881fa5b4da28a5ce9187d44b5cf",
    "best_candidate_iid_macro_ap": 0.7980789112007961,
    "best_candidate_iid_delta": 0.00844532422073796,
    "best_candidate_iid_p_holm_stage": 0.001999000499750125,
    "best_candidate_gain_over_anchor": 0.00844532422073796,
    "challenger_selected": true,
    "practical_shortlist": [
      "minilm5_sft_learningrate8m5_0797e754_v1",
      "minilm5_sft_e1_lr4e5_v1"
    ],
    "boundary_extension_evidence": [],
    "needs_boundary_extension": false,
    "recommended_extension": null
  }
}
```
