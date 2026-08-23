# MiniLM S2 values-only probing submission

This submission uses the winning checkpoint from the fixed 120k human-only
serialization screening run. It is a probing leaderboard submission, not a
full-data final model.

Inference reproduces the validation representation:

- shared NFKC/casefold/whitespace/unit normalization;
- title followed by attribute values only;
- train-derived attribute order;
- `max_length=256`, longest-first pair truncation;
- mean of sigmoid probabilities for A→B and B→A.

The submission archive has `metadata.json` and `run.py` at its root. Model
weights are bundled under `models/minilm-s2-values-only/`; the Docker image only
provides the offline runtime.

## Build and push the runtime image

Run from the repository root:

```powershell
docker build --platform linux/amd64 -f docker/minilm-s2-runtime/Dockerfile -t dinakepech/ecup26-minilm-s2:1.0 .
docker run --rm --gpus all dinakepech/ecup26-minilm-s2:1.0 python -c "import torch, transformers, pyarrow; print(torch.cuda.is_available(), transformers.__version__, pyarrow.__version__)"
docker login
docker push dinakepech/ecup26-minilm-s2:1.0
```

If the Docker Hub namespace is different, change both the tag above and
`metadata.json` before building the ZIP.

## Build the archive

```powershell
.\.venv\Scripts\python.exe scripts\build_minilm_s2_submit.py
```

Result: `submits/minilm-s2-values-only.zip`.

## Local format smoke test

```powershell
.\.venv\Scripts\python.exe submits\minilm-s2-values-only\run.py `
  -i data\items_human.parquet -m data\matches.parquet `
  -o artifacts\minilm-s2-smoke.csv --skip-model --limit 1000
```

The final GPU check must run without `--skip-model` inside the built image.
