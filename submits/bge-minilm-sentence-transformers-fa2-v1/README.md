# BGE + MiniLM SentenceTransformers SDPA ensemble

Experimental H100 submission requested after the native full ensemble missed
the deadline. Both frozen fine-tuned checkpoints run through
`sentence_transformers.CrossEncoder` with:

- one-direction pair inference;
- `max_length=192`;
- FP16 weights;
- `attn_implementation=sdpa` (XLM-R does not support FlashAttention 2);
- batch size 1,024;
- character-length bucketing and 20-thread Rust tokenizer parallelism;
- mean normalized-rank aggregation.

There is no attention fallback, timeout fallback, partial scoring or Check
bypass. The changed length and one-direction inference intentionally follow the
new requested runtime experiment and are not claimed equivalent to the earlier
384-token symmetric validation pipeline.

Build the image and archive from the repository root:

```powershell
docker build --platform linux/amd64 -f docker\bge-minilm-sentence-transformers-fa2\Dockerfile -t dinakepech/ecup26-bge-minilm-sentence-transformers-sdpa:1.0 .
docker push dinakepech/ecup26-bge-minilm-sentence-transformers-sdpa:1.0
.venv\Scripts\python.exe scripts\build_bge_minilm_sentence_transformers_submit.py
```
