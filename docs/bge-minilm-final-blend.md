# Final BGE + MiniLM blend evaluator

`scripts/evaluate_bge_minilm_final_blend.py` is a local, read-only-input
evaluator to run only after the BGE loss screen and seed confirmation have
produced an authoritative final slim output. It does not import the Kaggle or
Google Sheets controllers and cannot launch, upload, sync or train anything.

The frozen contract is `configs/bge_minilm_final_blend_v1.json`. It permits
exactly two score combinations:

- 60% BGE / 40% MiniLM in logit space;
- 70% BGE / 30% MiniLM in logit space.

There is no grid extension, category-specific tuning, probability blend or
clipping. Both input scores must be finite and strictly inside `(0, 1)`.

## Input authority

The MiniLM side is byte-bound to the selected seed-42
`minilm5_sft_e3_lr8e5_v1` prediction artifacts:

- IID SHA-256
  `2fd79620059ad4425afe3998912b08f8cd6f9db06d54ca2fd042a9b5f7215e00`;
- hard SHA-256
  `da696070b57c310ba5378861780d624b62517c23fafd391deb84153e52700a28`.

The BGE side is not authorized by caller-supplied hashes. The evaluator requires
the authoritative `loss_confirmation_receipt.json` together with its exact
loss-screen, LR-selection and epoch-selection receipt chain. It rehashes every
receipt link, recomputes the seed-42 screen and two-seed final gate, derives the
selected seed-17 loss output, and obtains the IID/hard path, size and SHA-256
only from that selected `prediction_binding`. A baseline, LR/e2 candidate or
completed loss-screen output therefore cannot be substituted.

It also requires one raw `notebook_completed.json`, status `complete`, the
selected run/experiment/identity/recipe/loss binding, exact IID/hard evaluation
declarations, recomputed AP equal to the completion report and the BGE OOD
sentinel contract.

For each split the BGE and MiniLM files must have exactly the same row order,
`id1`, `id2`, target and category. Missing columns, duplicate/self pairs,
non-binary targets, category loss, invalid scores and an OOD parquet in the BGE
slim output are rejected.

## Run after final confirmation

Do not point this command at a running loss kernel directory. After the
authoritative final receipt identifies the selected seed-17 slim output, run:

```bash
PYTHONPATH="$PWD" .venv/bin/python \
  scripts/evaluate_bge_minilm_final_blend.py \
  --bge-dir artifacts/kaggle/FINAL_BGE_SLUG \
  --final-receipt reports/bge_2ep_sft_loss_confirmation_v1/confirm/loss_confirmation_receipt.json \
  --screen-receipt reports/bge_2ep_sft_loss_confirmation_v1/screen/loss_screen_receipt.json \
  --lr-receipt reports/bge_2ep_sft_candidate_v1/lr/lr_selection_receipt.json \
  --epoch-receipt reports/bge_2ep_sft_candidate_v1/e2/epoch_selection_receipt.json \
  --report-json reports/bge_minilm_final_blend_v1/report.json \
  --report-markdown reports/bge_minilm_final_blend_v1/report.md
```

Omit both report arguments for a strictly read-only run that prints the full
deterministic JSON report to stdout. When report paths are supplied, existing
files are never overwritten and outputs cannot be placed under either input
directory.

## Decision rule

IID macro AP is the sole selection metric. The evaluator first chooses between
the two blends; a tie within `1e-12` prefers 70% BGE. The final recommendation
then compares that frozen pair with the two single-model endpoints. A tie
prefers a single model to avoid ensemble cost (MiniLM, then BGE), followed by
the 70% and 60% BGE blends. Hard remains diagnostic and cannot change either
decision.

Former OOD categories were included in BGE supervised training. Consequently
OOD is never read or evaluated, its macro-AP sentinel is exactly `-1`, and the
report explicitly makes no OOD, hidden-test-gain or runtime-viability claim.
