# BGE post-LR/e2 loss and seed confirmation

This workflow is a deliberately small continuation of the frozen BGE v4
baseline and its LR/e2 candidate workflow. It lives in separate files and does
not modify either frozen implementation.

The controller is
`scripts/run_bge_2ep_sft_loss_confirmation.py`. Its default action prints a
network-free plan: it does not load `.env`, resolve the Kaggle CLI, stage a
notebook, contact Google Sheets, or mutate Kaggle.

## Required authority

No loss notebook can be staged until all of the following validate exactly:

1. the final v4 baseline local output and its private slim Dataset, positive
   Dataset version, raw manifest SHA-256, exact three-file payload and privacy;
2. the complete LR selection receipt and both raw LR outputs, paired summaries
   and exact `sft_exps` comparison markers;
3. the complete e2 receipt, its raw output, paired IID/hard comparison and exact
   comparison marker;
4. exact equality between the epoch summary splits, the standalone comparison
   and the comparison embedded in the Sheets-bound augmented completion, then a
   recomputation from that embedded IID delta: e2 is selected only when its gain
   versus selected e1 is strictly greater than `0.002`;
5. the selected e1/e2 raw output is plain `bce_finite_guard_v1`, seed 42, and
   binds the exact selected recipe, source, checkpoint, validation manifest,
   prediction hashes and initial Sheets marker.

Experiment names or recipes supplied on the command line are never selection
authority. The controller derives them from the frozen receipts and revalidates
the raw artifacts.

## Stage 1: one transferred loss

The selected seed-42 e1/e2 BCE output is reused as the matched anchor. Exactly
one new seed-42 run is permitted:

- `balanced_category_class_sqrt_bce`;
- fresh start from `model/pretrain_bge_2ep`;
- the selected LR and epoch count;
- all other training coordinates unchanged.

A matched seed-42 BCE materialization is allowlisted only as a fail-closed
fallback. Under the current authority it is not reachable: the LR/e2 receipt
cannot validate without the exact selected raw BCE output. If that output is
unavailable, the workflow stops. It will not spend an extra anchor kernel and
then discover that the worst-case two-run seed-17 branch exceeds the total cap.

The transferred weight for an example in category/class stratum `s` is
`n_s^-0.5`, normalized to global mean one. The hook requires exactly 347,840
rows, 20 categories and 40 non-empty category/class strata. With the frozen
combined train its expected weight range is
`0.6814300041898296 .. 2.753521186636443`.

IID macro AP is primary. Hard is diagnostic and cannot affect selection. The
challenger reaches seed confirmation only if its paired IID gain is strictly
greater than `0.002`; equality is a rejection. The one-member family still
records paired component permutation evidence, paired component bootstrap
intervals and Holm family size one.

Former OOD pairs stay in train. No OOD parquet is read, written or compared.
The run metric is `-1`; comparison delta, p-value, Holm value and CI are null in
JSON and blank in `sft_exps`.

## Stage 2: directional seed 17

The second seed is sequential and conditional:

- if BCE won at seed 42, run one matched BCE seed-17 model and stop;
- if the transferred loss passed at seed 42, first run matched BCE seed 17,
  then run the transferred loss seed 17.

The seed-17 pair differs only in loss. Both start from the initial checkpoint;
neither resumes or exports a trained checkpoint. The transferred loss is the
final choice only when its IID delta is positive on both seeds and the mean of
the two deltas is at least `0.002`. Otherwise the final loss is BCE. No third
seed, ensemble, boundary point, loss-LR refinement, ODS, runtime ablation or
checkpoint experiment is allowed.

## Kernel budget and failure semantics

The campaign ledger includes all four baseline slugs, including failures:

- `pm-b2-base-9c1f4648466b-s42-v1`;
- `pm-b2-base-6ad383889383-s42-v1`;
- `pm-b2-base-97335fa432bd-s42-v1`;
- `pm-b2-base-de25c35eabf4-s42-v1`.

The two LR kernels and e2 bring the pre-loss union to seven. The seed-42 loss
screen plus the largest seed-17 branch adds three, so the exact worst case is
10 unique BGE kernel slugs. The controller reserves this worst case before the
first loss push and rejects a larger union. Every loss-kernel push intent is
first appended to
`.kaggle/audit/bge_2ep_sft_loss_confirmation_v1/attempted_kernel_slugs.json`.
That local ledger is never rolled back after a failed API call. Before every
new push it must equal the expected preceding branch exactly, and two
authenticated owner-wide searches for `pm-b2-lbce-*` and `pm-b2-lsqrt-*` must
return that same set. Changing source code, Dataset version or another identity
coordinate therefore cannot silently create a replacement slug or hide a
spent slot.

Only one kernel is active at a time. A valid local output is reused; a
queued/running kernel is verified and waited for; a completed kernel is
verified and downloaded. A terminally failed slug is never resubmitted or
replaced automatically. Before a first push, the controller performs the
existing two-pass absence audit and rechecks the private baseline Dataset
version, status, manifest, file set and privacy. The soft kernel timeout cannot
exceed nine hours.

Every notebook attaches exactly validation, initial checkpoint, frozen private
baseline and Sheets-credentials Datasets, remains private, uses two T4s, writes
only slim reports and IID/hard predictions, and rejects model weights and OOD
predictions in downloaded output.

## Commands

Inspect the plan without side effects:

```bash
uv run python scripts/run_bge_2ep_sft_loss_confirmation.py
```

After LR/e2 are complete and comparison-Sheets-synced, stage only the next
seed-42 challenger locally:

```bash
uv run python scripts/run_bge_2ep_sft_loss_confirmation.py \
  --stage screen \
  --baseline-dataset-version VERSION \
  --baseline-manifest-sha256 MANIFEST_SHA256 \
  --dry-run
```

Execute it sequentially, download/validate the slim output, compare and sync
the augmented row to `sft_exps`:

```bash
uv run python scripts/run_bge_2ep_sft_loss_confirmation.py \
  --stage screen \
  --baseline-dataset-version VERSION \
  --baseline-manifest-sha256 MANIFEST_SHA256 \
  --execute
```

Then run the conditional seed-17 branch:

```bash
uv run python scripts/run_bge_2ep_sft_loss_confirmation.py \
  --stage confirm \
  --baseline-dataset-version VERSION \
  --baseline-manifest-sha256 MANIFEST_SHA256 \
  --screen-receipt reports/bge_2ep_sft_loss_confirmation_v1/screen/loss_screen_receipt.json \
  --execute
```

`--summarize-local` performs comparison from already strict local outputs and
does not contact Kaggle or Sheets. It writes the audit-only
`loss_screen_receipt.local.json` or `loss_confirmation_receipt.local.json`.
These are not default live downstream authority. A later `--execute` can add
the exact Sheets marker and write the separate authoritative receipt without
deleting or overwriting the local receipt.

No Kaggle kernel or Dataset was created while this workflow was implemented and
tested.
