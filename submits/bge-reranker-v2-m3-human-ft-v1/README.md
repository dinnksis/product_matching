# BGE reranker v2 m3 human fine-tuned submission

This is the one-human-epoch BGE architecture baseline packaged as an offline
competition submission. It reproduces validation inference:

- S2 values-only serialization;
- `max_length=384`;
- sigmoid of the one-logit sequence-classification output;
- mean of A-to-B and B-to-A probabilities;
- FP16 autocast;
- full coverage of every input pair.

There is deliberately no lexical fallback, soft deadline, Check bypass or
partial inference path. If BGE does not finish, the competition runtime must
report the genuine timeout.

## Build and push the runtime image

```powershell
docker build --platform linux/amd64 -f docker/bge-reranker-v2-m3-runtime/Dockerfile -t dinakepech/ecup26-bge-reranker-v2-m3:1.0 docker/bge-reranker-v2-m3-runtime
docker run --rm --gpus all dinakepech/ecup26-bge-reranker-v2-m3:1.0 python -c "import torch, transformers, pyarrow; print(torch.cuda.is_available(), transformers.__version__, pyarrow.__version__)"
docker login
docker push dinakepech/ecup26-bge-reranker-v2-m3:1.0
```

## Build the archive

Place the downloaded Kaggle checkpoint files under
`artifacts/kaggle/product-matching-architecture-bge-v2-m3-v1/bge_reranker_v2_m3_human_ft_v1/`,
then run:

```powershell
.venv\Scripts\python.exe scripts\build_bge_reranker_submit.py
```

Result: `submits/bge-reranker-v2-m3-human-ft-v1.zip`.

The archive root contains `metadata.json` and `run.py`; weights are bundled in
the archive, while the Docker image contains only the pinned runtime.
