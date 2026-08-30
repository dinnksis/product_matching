# Final human fine-tuning: MiniLM and RuModernBERT

This document covers only the final all-human stage. The competition-pretrain
scripts that create its two initial checkpoints are a separate team contribution
and must be present for a complete end-to-end audit.

## Inputs

The notebook generators require the organizer-provided human data locally:

- `data/items_human.parquet`: 711,304 rows with `id`, `name`, `attributes`,
  `category`;
- `data/matches.parquet`: 365,654 rows with `id1`, `id2`, `target`.

The final stage also receives one intermediate checkpoint:

- MiniLM: `alexproger23/product-matching-minilm-llm-pretrain-5ep-full`;
- RuModernBERT: `alexproger23/product-matching-rumodernbert-pretrain-3ep`.

These private Kaggle datasets are immutable transport caches, not the canonical
reproduction path. Each notebook verifies the attached checkpoint manifest before
training. The preceding pretrain code must independently recreate their contents.

## Serialization

Both models use `src.data_pipeline.serialize_product` with
`max_attribute_chars=6000`:

```text
Категория: <category>
Название: <name>
<attribute key>: <attribute value>
...
```

Whitespace is normalized without altering punctuation, numbers, SKUs or units.
Attributes are parsed as JSON, empty values are removed, nested values are encoded
deterministically, and important brand/model/code/type fields are emitted first.
The cross-encoder tokenizer then truncates the combined product pair to 384 tokens.

## Locked recipes

MiniLM uses
`configs/cross_encoder_minilm_5ep_full_human_final.json`:

- 3 epochs, 2 GPUs;
- batch 96 per GPU, no gradient accumulation, effective batch 192;
- AdamW, learning rate `8e-5`, weight decay `0.01`, warmup ratio `0.05`;
- classifier dropout `0.1`, max grad norm `1.0`;
- FP16, SDPA, max length 384.

RuModernBERT uses
`configs/cross_encoder_rumodernbert_3ep_full_human_final.json`:

- 3 epochs, 2 GPUs;
- batch 24 per GPU, accumulation 4, effective batch 192;
- AdamW, learning rate `4e-5`, cosine scheduler, weight decay `0.01`, warmup
  ratio `0.05`;
- max grad norm `0.5`;
- FP16, SDPA, max length 384, `trust_remote_code=false`.

Both runs use seed 42, standard BCE, no sampling, no sample weights, no label
smoothing and no validation. The final stage intentionally consumes all human
labels; model and hyperparameter selection happened in earlier component-disjoint
experiments.

## Notebook generation and launch

Install the repository environment and place Kaggle credentials in the ignored
`.env` file as documented in `docs/kaggle-notebook.md`.

Dry-run the complete notebook packaging without contacting Kaggle:

```powershell
.\.venv\Scripts\python.exe scripts\run_minilm_5ep_full_human_kaggle.py --dry-run
.\.venv\Scripts\python.exe scripts\run_rumodernbert_3ep_full_human_kaggle.py --dry-run
```

Launch in the background:

```powershell
.\.venv\Scripts\python.exe scripts\run_minilm_5ep_full_human_kaggle.py --no-wait --no-download
.\.venv\Scripts\python.exe scripts\run_rumodernbert_3ep_full_human_kaggle.py --no-wait --no-download
```

The wrappers regenerate these committed audit notebooks before each launch:

- `notebooks/minilm_5ep_full_human_final_2xt4.ipynb`;
- `notebooks/rumodernbert_3ep_full_human_final_2xt4.ipynb`.

## Output contract

Each notebook writes under `/kaggle/working`:

- the complete Hugging Face checkpoint and tokenizer;
- `training_config.json`;
- `training_report.json`;
- `checkpoint_manifest.json` with reconstruction parts and SHA-256;
- `notebook_completed.json` only after every invariant passes.

The notebook checks the expected world size, effective batch, FP16 dtype, absence
of custom loss/sample weights and successful completion of all three epochs.

## Final submission checkpoints

Downloaded inference files are stored locally under ignored directories:

```text
configs/minilm_final/
configs/rumodernbert_final/
```

The repository currently also recognizes the historical local spelling
`configs/rubertmodern_final/`. Checkpoint source files are never modified by the
submission builders. RuModern's compile flags are disabled only in the staged ZIP
copy because the offline runtime does not provide a compiler for its optional
TorchInductor path.
