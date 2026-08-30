# Surface-only positive augmentation

`scripts/build_surface_positive_pairs.py` creates a conservative label-1
ablation with no atomic semantic change.  It is deterministic and does not call
Qwen or any network service.

The default build selects 5,000 unique donors only from target-1 rows of the
frozen human **train** split.  Consequently, validation-only products and the
frozen OOD categories cannot enter the augmentation.  For every donor it emits
two new negative ids:

- the left card is a normalized-JSON clone of the real donor;
- the right card is another seller presentation of the same product facts.

Allowed right-side changes are movement of a known brand in the title,
punctuation/case variation, conservative attribute-key aliases, JSON key-order
variation, and omission of at most two non-identity fields.  Brand, model,
article/SKU, series, and product type are never omitted.  No retained attribute
value may gain, lose, or replace an alphanumeric token.  Obvious malformed
source cards such as `Бренд: фиолетовый` are rejected before sampling.

## Build and validate

```bash
UV_CACHE_DIR=/tmp/uv-cache-surface-positive uv run python \
  scripts/build_surface_positive_pairs.py

UV_CACHE_DIR=/tmp/uv-cache-surface-positive uv run python \
  scripts/build_surface_positive_pairs.py --validate-only
```

The default output is
`item_pipeline/artifacts/surface_positive_human_train_5k_v1/`.  The count is
configurable, for example `--count 2000 --output-dir <path>`.  A different
real or previously generated catalogue can be supplied with `--items`, but
`--eligible-pairs` must still be the exact training-only pair file whose
target-1 endpoints are authorized as donors.

The directory contains:

- `items.parquet` and `pairs.parquet`;
- `pair_provenance.parquet`, including the donor id, source train-pair row,
  title operation, key mapping, omitted fields, and fact hashes;
- `validation_report.json`, with the strict no-new/no-conflicting-facts result;
- `distribution_report.json`, comparing title length, attribute counts, key
  overlap, and category mix with human train positives;
- `examples.jsonl`, `summary.json`, and a SHA-256 `build_manifest.json`.

This dataset intentionally represents an easier regime than ordinary human
positive pairs: most logical attributes agree and only their presentation
differs.  It should therefore remain a separate ablation, not be mixed into the
atomic-rule experiment before its standalone effect is measured.
