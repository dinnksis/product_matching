# Jina Reranker v2 zero-shot submit

Names-only speed/quality experiment using the 278M multilingual cross-encoder
`jinaai/jina-reranker-v2-base-multilingual`. It uses direct batched logits,
without training and without an API call.

## Build runtime image

The image name in `metadata.json` must exist on Docker Hub before submission:

```bash
docker build -f docker/jina-reranker-runtime/Dockerfile \
  -t dinnksis/ecup26-jina-reranker:1.0 .
docker login
docker push dinnksis/ecup26-jina-reranker:1.0
```

If your Docker Hub username/tag differs, update `metadata.json` first.

## Build archive

```bash
python scripts/build_jina_submit.py
```

This downloads a pinned model revision, verifies the weight SHA-256 and creates:

```text
submits/jina-reranker-v2-zero-shot.zip
```

Rebuild from already downloaded assets without network:

```bash
python scripts/build_jina_submit.py --archive-only
```

## Local smoke test without CUDA

```bash
python submits/jina-reranker-v2-zero-shot/run.py \
  -i data/items_human.parquet -m data/matches.parquet \
  -o /tmp/jina-smoke.csv --skip-model --limit 1000
```

Runtime knobs: `PM_BATCH_SIZE` (default 512), `PM_MAX_LENGTH` (128),
`PM_SCORE_CHUNK_SIZE` (20000), `PM_PUBLIC_MAX_PAIRS` (200000),
`PM_PUBLIC_SOFT_LIMIT_SECONDS` (320), `PM_PRIVATE_SOFT_LIMIT_SECONDS` (740).

This is an experimental zero-shot submit. Validation quality and actual H100
throughput must be measured by the competition runner.
