# BGE 2ep SFT post-baseline candidate workflow

This workflow is intentionally separate from the frozen baseline notebook and
its source ledger. It reuses the exact baseline runtime cells and embedded
runtime sources, but every candidate has its own executable identity and also
binds the candidate-generator SHA-256.

The controller is `scripts/run_bge_2ep_sft_candidates.py`. Its default action
is plan-only: it does not load `.env`, resolve the Kaggle CLI, stage a notebook,
or make a network request.

## Hard prerequisites

Candidate staging and execution remain blocked until all of these exist:

1. the final baseline local output passes the complete baseline payload and
   exact `sft_exps` Sheets-marker validators;
2. `scripts/push_bge_2ep_sft_baseline_dataset.py` has built the private slim
   Dataset containing exactly the completion plus compact IID/hard prediction
   Parquets (no model and no OOD predictions);
3. the caller supplies the exact positive Kaggle Dataset version and the exact
   raw baseline-manifest SHA-256 printed by that uploader;
4. immediately before a new candidate kernel is pushed, the controller checks
   the remote Dataset twice for the exact version/readiness, downloads and
   hashes its manifest, verifies its exact file set, and verifies Kaggle still
   reports it private.

The dynamic loader rebuilds the current baseline entry and requires equality of
the campaign identity, source ledger, executable cells, recipe, loss hook,
initial checkpoint, validation manifest and baseline run ID. There is no
provisional baseline slug or source hash in the candidate code.

## Core experiment sequence

The first stage always runs sequentially:

1. `bge2_sft_oodtrain_e1_lr1e5_v1`;
2. `bge2_sft_oodtrain_e1_lr4e5_v1`.

Both start fresh from `model/pretrain_bge_2ep`; neither resumes baseline
weights. They change only learning rate from the exact baseline recipe. The
complete two-candidate family is compared against the frozen baseline with
paired component permutation tests and paired component bootstrap intervals,
separately on IID and hard. Holm correction is applied across the two planned
candidates within each split. IID is primary, hard is diagnostic, and the
practical tie margin is `0.002`, with tie order baseline, `1e-5`, then `4e-5`.

The LR stage writes an exact `lr_selection_receipt.json`. Only that receipt can
materialize e2. The e2 config is copied from the selected e1 recipe and changes
only `epochs: 1 -> 2`; it still starts fresh from the initial checkpoint. Thus,
if an LR candidate wins, e2 cannot silently revert to the baseline LR. E2 is
compared to the selected e1 run as a one-member Holm family. E1 is retained
unless the paired IID delta for e2 is strictly greater than `0.002`. Hard does
not select. Boundary LR expansion and epoch 3 are deliberately deferred.

Former OOD pairs are part of supervised training. Every training report and
Sheet row therefore keeps `ood_macro_ap = -1`, while OOD delta, p-value,
Holm-adjusted p-value and confidence-interval fields are `null` in comparison
JSON and blank in Sheets. No OOD paired comparison is run.

In live mode the LR receipt is written only after both augmented comparison
rows have exact comparison-sync markers. A live e2 launch rejects a receipt
created by `--summarize-local`, whose `comparison_sheets_synced` value is
`false`. Before e2, the loader rebuilds both canonical LR entries, runs the
strict raw-output validator on both, checks their run/identity/source/recipe
bindings against the hashed full family summary, and rehashes both declared
comparison markers against their augmented completions.

## Commands

Inspect the network-free plan:

```bash
uv run python scripts/run_bge_2ep_sft_candidates.py
```

After the final baseline Dataset is uploaded, locally generate and validate the
two LR notebooks and exact four-Dataset kernel metadata without contacting
Kaggle:

```bash
uv run python scripts/run_bge_2ep_sft_candidates.py \
  --stage lr \
  --baseline-dataset-version VERSION \
  --baseline-manifest-sha256 MANIFEST_SHA256 \
  --dry-run
```

The explicit live command performs the same local gates, runs `1e-5` and
`4e-5` sequentially, downloads and strictly validates each slim output,
materializes the paired family, and updates both exact comparison rows in
`sft_exps`:

```bash
uv run python scripts/run_bge_2ep_sft_candidates.py \
  --stage lr \
  --baseline-dataset-version VERSION \
  --baseline-manifest-sha256 MANIFEST_SHA256 \
  --execute
```

Then stage or execute the single parented e2 run:

```bash
uv run python scripts/run_bge_2ep_sft_candidates.py \
  --stage e2 \
  --baseline-dataset-version VERSION \
  --baseline-manifest-sha256 MANIFEST_SHA256 \
  --lr-receipt reports/bge_2ep_sft_candidate_v1/lr/lr_selection_receipt.json \
  --execute
```

Use `--summarize-local` instead of `--execute` to compare already validated
local outputs without Kaggle or Google Sheets calls.

## Resume and failure policy

A locally valid output is reused. A queued/running kernel is verified and
waited for. A completed kernel is verified and downloaded. A terminally failed
kernel raises an error and is never automatically replaced. An apparently
absent slug is submitted only after the existing two-pass absence audit and the
final remote baseline Dataset gate. Candidate outputs must preserve the exact
runtime baseline gate artifact, parent receipt, generator SHA and initial
Sheets marker. Post-comparison Sheets updates receive a separate exact,
idempotent `google_sheets_comparison_sync.json` marker.

No candidate kernel was submitted while this workflow was implemented or
tested.
