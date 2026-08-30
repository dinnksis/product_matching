# Frozen three-model inference benchmark

This experiment does not train or modify any checkpoint. It benchmarks the
existing BGE v2-m3, MiniLM and RuModernBERT sequence classifiers with the frozen
`S2_VALUES_ONLY` serialization, symmetric pair scoring and `max_length=384`.
GTE is intentionally outside this runtime shortlist because the earlier
quality/diversity experiments found it redundant.

## Private checkpoint Dataset

The exact private Dataset slug is:

`dinakepecheva/product-matching-inference-checkpoints-v1`

The owner is read from `KAGGLE_USERNAME`, so the command also works if the
account is changed. The payload contains:

- the unchanged BGE, MiniLM and RuModernBERT `model.safetensors` files;
- each model's matching `config.json` and tokenizer files;
- the saved IID, hard and OOD reference predictions for score-equivalence checks;
- the frozen attribute-frequency table and a SHA-256 manifest.

Large weights are split into 128 MiB parts for upload and reconstructed
byte-for-byte inside the notebook. The Dataset must remain private.

```powershell
.venv\Scripts\python.exe scripts\push_inference_benchmark_checkpoints.py --dry-run
.venv\Scripts\python.exe scripts\push_inference_benchmark_checkpoints.py
```

## Notebook

Generate and validate the Kaggle staging area without submitting:

```powershell
.venv\Scripts\python.exe scripts\run_inference_benchmark_kaggle.py --dry-run
```

Submit in the background:

```powershell
.venv\Scripts\python.exe scripts\run_inference_benchmark_kaggle.py --no-wait --no-download
```

The kernel slug is
`dinakepecheva/product-matching-inference-backend-benchmark-v1`.

After it finishes, download only the generated working artifacts:

```powershell
.venv\Scripts\python.exe scripts\run_inference_benchmark_kaggle.py --download-existing
```

The default Kaggle notebook is a quick 5,000-pair native-runner shortlist. It
does not repeat full validation because the frozen validation scores already
exist, and it does not run slow compile/optional-backend probes. Results are
written to `inference_benchmark_results.csv` and the selected runners and rough
T4 extrapolations to `inference_benchmark_summary.json`. Every benchmark row is also appended to the
`inference_benchmarks` worksheet in the existing experiment spreadsheet. A
failed Sheets sync does not invalidate benchmark artifacts and is recorded in
`sheets_sync_pending.json`.

Kaggle T4 timing is useful for choosing an implementation but is not a valid
final 6/13-minute verdict for the H100 competition runner. The winning native
runner(s) must be copied into Docker and timed end-to-end on the platform.
