# BGE 2ep SFT: frozen baseline and paired comparisons

This support layer is intentionally separate from the MiniLM significance
pipeline.  BGE is trained on the 306,669 original human-train pairs plus all
41,171 pairs from the former OOD categories.  Consequently only IID and hard
can be compared; OOD is always recorded as `-1` and no OOD prediction artifact
is accepted.

## Freeze the completed baseline

After the exact baseline kernel has completed, been downloaded, passed the BGE
launcher validator and synced to `sft_exps`, build the private slim Dataset:

```bash
.venv/bin/python scripts/push_bge_2ep_sft_baseline_dataset.py
```

This command is a local dry-run by default and never resolves or calls the
Kaggle CLI.  It derives the current baseline slug and identity from the frozen
BGE builder, validates the downloaded run twice, and stages exactly:

- the byte-exact `notebook_completed.json`;
- compact `iid_validation_predictions.parquet`;
- compact `hard_validation_predictions.parquet`;
- the binding manifest and private Dataset metadata.

The manifest binds the baseline `run_id`, experiment, campaign identity, code
bundle, recipe, validation manifest and initial checkpoint/model SHA-256.  It
also records both source and compact prediction hashes.  Model weights, OOD
predictions, logs, texts and credentials are forbidden.

Inspect the local manifest and only then explicitly authorize the mutation in
the command itself:

```bash
.venv/bin/python scripts/push_bge_2ep_sft_baseline_dataset.py --upload
```

The target is the private Dataset
`alexproger23/product-matching-bge-2ep-sft-baseline-v1`.  The uploader rehashes
the complete stage immediately before upload and verifies the remote manifest
and privacy flag afterward.

## Compare a complete planned candidate family

The comparison tool requires every predeclared non-anchor candidate in a Holm
family.  It refuses partial families, which prevents sequential result peeking
from silently changing the multiplicity correction.

For the initial log-LR line:

```bash
.venv/bin/python scripts/summarize_bge_2ep_sft_comparisons.py \
  --baseline-dir /path/to/mounted-or-downloaded-bge-baseline-dataset \
  --candidate-dir artifacts/kaggle/<lr1e5-kernel> \
  --candidate-dir artifacts/kaggle/<lr4e5-kernel> \
  --planned-experiment bge2_sft_oodtrain_e1_lr1e5_v1 \
  --planned-experiment bge2_sft_oodtrain_e1_lr4e5_v1 \
  --family-name lr_log_line_v1 \
  --tie-break-order bge2_sft_oodtrain_e1_lr2e5_baseline_v1 \
  --tie-break-order bge2_sft_oodtrain_e1_lr1e5_v1 \
  --tie-break-order bge2_sft_oodtrain_e1_lr4e5_v1 \
  --output-dir reports/bge_2ep_sft/lr_log_line_v1
```

For each candidate, `compare_prediction_frames` runs separately on IID and
hard with paired component permutation and component bootstrap.  Holm is then
applied across the complete planned candidate family independently for each
split.  IID is the sole selection split; hard is diagnostic only.  The
practical-tie margin is `0.002`, with the supplied tie-break order deciding
among recipes within that margin.

Outputs are written outside the source run directories:

```text
<output-dir>/
├── family_summary.json
├── <candidate-a>/
│   ├── baseline_comparison.json
│   └── completion_with_comparison.json
└── <candidate-b>/
    ├── baseline_comparison.json
    └── completion_with_comparison.json
```

The augmented completions are compatible with the shared `sft_exps` logger.
Their explicit comparison status is `ready_ood_disabled`; OOD delta remains
blank/null together with OOD p-values and confidence intervals.  Only the run
and Sheet OOD macro-metric field carries the `-1` sentinel.  This support module
performs no Kaggle submissions and no Google Sheets mutation by itself.

Candidate execution and adaptive stage control remain separate work.  A future
controller should construct one planned family at a time, wait until every
member is validated locally, run this comparator, and only then make the stage
selection.
