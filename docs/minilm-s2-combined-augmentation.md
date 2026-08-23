# MiniLM S2 combined augmentation

## What `minilm_balanced_llm_2ep` means

The Google Sheets row `minilm_balanced_llm_2ep` is not a serialization-ablation
run. It used a balanced mixed dataset with 310,767 human pairs and 388,833
LLM-labelled pairs (699,600 examples per epoch), trained for two epochs with
`max_length=384`, and averaged A/B plus B/A during validation. Its human-only
validation also comes from an older 54,887-pair split. The reported macro AP
`0.726108` is therefore not directly comparable with the S2 serialization score
`0.690153`, which used 120k human-only training pairs, one epoch,
`max_length=256`, and the newer 51,470-pair family holdout.

## Controlled experiment

Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.

- `A_BASELINE`: deterministic S2 values-only serialization; no attribute
  shuffle and no pair reversal during training.
- `B_SHUFFLE_SWAP`: the same S2 serialization and normalization; intact
  attribute entries are permuted per pair/epoch before their keys are omitted,
  and the pair is reversed with probability 0.5 during training.

Both models start from the same upstream checkpoint and use the same full
human-only train pool, split, seed, optimizer, LR, batch size, `max_length=256`
and one epoch. The two jobs run independently on the two T4 GPUs. Validation
attributes use deterministic order and both models execute exactly one A-to-B
forward; no symmetric score averaging is used.

The fixed `brand_model_family_holdout` contains 314,184 train pairs and 51,470
validation pairs. Product-ID overlap and model-family signature overlap are
both zero.

## Extra validation slices

The existing lexical diagnostics define cheap hard-looking slices without new
labels:

- high name similarity: token-set ratio at least 0.8;
- critical variant conflict: an existing number, measurement, or model-code
  conflict flag;
- hard-looking: union of the two groups (42,702 validation pairs).

PR-AUC is not calculated on the positive-only subset because average precision
is undefined for a one-class slice. Positive support, mean score and score p10
are reported instead.

## Outputs

The Kaggle notebook writes the two training reports and predictions, a compact
comparison, the selected default-pipeline JSON, and only the winning checkpoint
under `/kaggle/working/minilm_s2_combined_augmentation/`. Both run rows and all
per-category AP values are upserted into Google Sheets by stable run IDs.

## Result (14 August 2026)

| Run | Macro PR-AUC | Overall PR-AUC | Hard-looking macro PR-AUC | Train time | Inference pairs/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_BASELINE` | 0.725369 | 0.783220 | 0.735125 | 1138.6 s | 1070.0 |
| `B_SHUFFLE_SWAP` | 0.721127 | 0.781447 | 0.731420 | 1102.0 s | 1219.9 |

Combined augmentation changed primary PR-AUC by `-0.004242` absolute or
`-0.585%` relative. The measured pair-swap rate was `50.007%`. Both validation
runs used one forward and identical token lengths (`avg=161.24`, `p95=256`), so
augmentation does not add inference work; the throughput difference is runtime
noise rather than a protocol difference.

The experiment rejects combined attribute shuffle plus pair swap. The default
remains deterministic S2 values-only training without either augmentation.

Both training processes completed successfully. The original notebook then
failed in its report-only cell because the summarizer did not add the project
root to `sys.path`. Reports and predictions were downloaded, aggregation was
recovered locally, and two experiment rows plus 40 category rows were verified
in Google Sheets. The import-path bug is fixed for future runs. Because the
failure happened before copying from `/kaggle/temp`, this run did not retain a
new full-human baseline checkpoint; no augmented checkpoint is selected.
