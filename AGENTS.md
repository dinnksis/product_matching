# Repository context for coding agents

Read this file before changing the repository. Also read the relevant document
under `docs/` for Kaggle or model-training work.

## Objective

This repository targets the E-CUP 2026 product-card matching competition. The
retrieval stage is already done: inference receives product cards and candidate
pairs and must return a continuous duplicate score for every pair.

Required CLI arguments:

- `--items_path` / `-i`: Parquet with `id`, `name`, `attributes`, `category`;
- `--matches_path` / `-m`: Parquet with `id1`, `id2`;
- `--output_path` / `-o`: CSV destination.

The output must contain exactly `id1,id2,predict`, preserve all pairs and use a
continuous numeric score. The metric is mean category-wise
`sklearn.metrics.average_precision_score` over 20 categories (macro AP).

The offline Docker runner has 20 CPU cores, 200 GB RAM and one H100 80 GB. Hard
limits are Check 1 minute, Public 6 minutes and Private 13 minutes. Startup,
Parquet reads, model loading and CSV writing all count. A model that does not
fit these limits is not a viable submission regardless of validation quality.

## Data currently in scope

Work currently uses only human labels:

- `data/items_human.parquet`: 711,304 unique products;
- `data/matches.parquet`: 365,654 pairs, 93,890 positives (25.68%);
- no missing IDs, nulls, duplicate unordered pairs, self-pairs or cross-category
  pairs; all `attributes` values parse as JSON objects;
- positive prevalence varies strongly by category (about 7.3% to 56.2%);
- 97.7% of products occur in only one labelled pair.

Use component-disjoint train/validation splits: a product must never appear in
both. Prepared outputs live under ignored `prepared/` and are not committed.
The 11M probabilistic LLM-labelled pairs are intentionally out of scope until a
separate decision is made.

## Established results and attempts

- Fast feature EDA: name `token_set_ratio` alone gives macro AP about 0.363.
- CPU `HistGradientBoostingClassifier` over lexical, numeric, brand and JSON
  pair features gives component-disjoint OOF macro AP 0.5351 (overall AP 0.6160).
- Qwen3-Reranker-0.6B names-only zero-shot gave local validation macro AP about
  0.4282, but neither the Transformers nor vLLM direction met the competition
  time constraints. LoRA training was abandoned. Do not resume Qwen reranker
  work without a new speed argument.
- `jinaai/jina-reranker-v2-base-multilingual` (278M), zero-shot, names only,
  direct Transformers inference, passed the competition time checks in roughly
  6--7 minutes and scored about 0.25 on the competition leaderboard. Treat it
  as the current proven neural runtime baseline, not as a quality baseline.
  It is CC-BY-NC-4.0; retain attribution and verify competition/license fit.
- MiniLM cross-encoder full fine-tuning is being improved by another
  collaborator. Avoid overwriting its config/training files without inspecting
  their latest Git state and coordinating scope.

## Current promising directions

1. Compact cross-encoders (Jina/MiniLM), then add carefully bounded attributes.
2. Qwen3-Embedding-0.6B item embeddings plus CatBoost/XGBoost pair features.
   Cache each item embedding once; never encode the same item per pair at
   inference. Candidate embedding features include cosine similarity, absolute
   difference, elementwise product summaries and optionally reduced vectors.
3. Structured attribute features: exact/conflicting values on category-specific
   frequent keys; semantic families for brand, model/SKU, size, quantity, color
   and material; counts/ratios of shared keys and numeric agreement. Do not make
   one global 34k-column table and do not blindly embed raw JSON.
4. Blend neural scores with the strong lightweight feature model. Macro metric
   makes per-category calibration/blending worth testing.
5. A stronger open-license generative judge may later handle only an uncertain
   subset, but routing must be deterministic and total inference must meet the
   hard deadline. Proprietary LLMs are forbidden for relabelling.

## Attribute representation principles

- Preserve original names, numbers, punctuation, SKUs, units and values.
- Parse JSON; skip empty values and serialize deterministic `key: value` lines.
- Key meaning is category-dependent. Prefer per-category support and signal
  estimates over one global frequency list.
- Distinguish missing/missing, one-side-missing, exact agreement and conflict.
- Treat seller/store/marketing/service metadata as possible noise until measured.
- For text encoders, put name and high-value fields first and enforce a tokenizer
  budget. For boosting, retain explicit comparisons rather than concatenating
  every attribute into one opaque string.

## Kaggle automation

Read `docs/kaggle-notebook.md` and model-specific docs before making changes.
The intended workflow is agent-operated from the local machine:

1. `uv sync` installs the pinned Kaggle CLI environment.
2. `.env` (never commit it) contains `KAGGLE_USERNAME`, `KAGGLE_API_TOKEN`,
   accelerator/private/internet settings and optional Kaggle sources.
3. A generator creates the experiment notebook and exact code/data bundle.
4. `scripts/push_kaggle_training_dataset.py` creates or versions a private
   Dataset containing Parquet and a SHA-256 source manifest.
5. A dry-run builds `.kaggle/staging/` and validates notebook metadata/GPU
   preflight without contacting Kaggle.
6. `scripts/run_kaggle_notebook.py` creates/updates a private kernel, requests
   two T4 GPUs, polls status, prints remote logs on failure and downloads
   `/kaggle/working` to `artifacts/kaggle/<kernel-slug>/`.

For background execution use `--no-wait`; the Kaggle kernel continues after the
local command returns. For an agent to monitor and download automatically, omit
`--no-wait` or use a model wrapper's `--wait`. Every notebook must write final
weights, reports, logs and a completion marker under `/kaggle/working`.

Never expose `.env`, `kaggle.json`, API tokens or private Dataset URLs in source,
notebook output or commits. External Kaggle mutations require user authorization.

## Submission discipline

- Keep every experimental submission isolated under `submits/<name>/`.
- Bundle pinned offline model weights and remote-code modules; the runner has no
  internet. Pin and verify revision, size and SHA-256 in the builder.
- `metadata.json` and the entry point must be at ZIP root.
- Configure all caches before importing ML libraries. The extracted workspace,
  `/root` and `/tmp` may be read-only; derive writable caches from the parent of
  the platform-provided `--output_path`.
- Perform an actual Check model load; do not hide load errors with a Check bypass.
- Use soft deadlines and a fast complete fallback rather than producing a
  partial CSV or being killed by the hard timeout.
- Preserve untracked user artifacts and unrelated collaborator changes.

## Important paths

- `src/data_pipeline.py`: serialization and component split;
- `src/product_matching/eda.py`: reproducible lightweight EDA/features;
- `docs/kaggle-notebook.md`: generic remote Kaggle workflow;
- `docs/cross-encoder-training.md`: MiniLM workflow;
- `submits/jina-reranker-v2-zero-shot/`: current passing neural runtime baseline;
- `notebooks/02_attributes_analysis.ipynb`: attribute investigation for the next
  embedding/boosting experiment.
- `docs/data-findings.md`: concise interpretation of the human data, attribute
  signals, and the existing CPU baseline;
- `docs/reranker-experiments.md`: chronological reranker/cross-encoder results,
  paths, runtime evidence and next model candidates;
- `docs/embedding-catboost-experiments.md`: the four confirmed boosting
  ablations and the unresolved 0.596 local versus 0.216 leaderboard gap;
- `notebooks/03_validation_split_audit.ipynb` and
  `docs/validation-split-audit.md`: compare exact component seeds and grouped
  name/model/seller holdouts with the same lexical CatBoost;
- `notebooks/04_error_pattern_analysis.ipynb` and
  `docs/error-pattern-analysis.md`: hard negatives, word/character n-grams,
  semantic FPR/FNR combinations and concrete FP/FN examples;
- `docs/prevalence-shift-diagnostic.md` and
  `reports/prevalence_shift_diagnostic/`: frozen lexical CatBoost weighting and
  bootstrap evidence that macro AP reaches 0.21 near 5.9% global effective
  prevalence without changing predictions; this is not an estimate of hidden
  prevalence;
- `src/embedding_boosting.py`: three controlled CatBoost ablations using names,
  Qwen item embeddings, and structured attributes;
- `scripts/run_embedding_boosting_kaggle.py`: background Kaggle launcher and
  later monitor/downloader for those ablations.
