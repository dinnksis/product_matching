# BGE reranker v2 m3, 3ep H100 submission

Offline competition bundle for the exact checkpoint in
`artifacts/server_bge_3ep_h100/run`.

The runner reproduces the training serializer (`category`, `name`, then
priority-sorted attributes), uses `max_length=384`, and scores one deterministic
orientation with the longer serialized card first. On the frozen validation
predictions this single-pass rule gives macro AP `0.823629` on IID and
`0.462631` on hard validation. The original symmetric two-pass scores were
`0.824975` and `0.461148`, respectively; the submit deliberately avoids nearly
doubling inference time for that small IID gain.

## H100 inference path

- one model replica on the single H100;
- BF16 weights and autocast;
- native Transformers SDPA, with PyTorch Flash-SDPA enabled when the kernel is
  eligible (no separately compiled `flash-attn` dependency);
- global length sorting and dynamic padding to a multiple of 8;
- initial pair batch `512`, automatically halved on CUDA OOM;
- a background Rust-tokenizer producer overlaps CPU tokenization with GPU work;
- 20 CPU/Rayon threads, asynchronous pinned-memory H2D copies, TF32 enabled;
- public/private soft deadlines with a complete lexical fallback for any tail.

Direct `transformers` is intentional here. `sentence-transformers.CrossEncoder`
loads the same XLM-R classifier and executes the same attention kernels, while
the already-published offline runtime image contains the exact Transformers
version used to export the checkpoint. Avoiding the extra wrapper/dependency is
the safer submit path and does not change the model computation.

The checkpoint is loaded for real even in the platform Check run. There is no
model-load bypass. `--skip-model` exists only for a local I/O/CSV smoke test.

Runtime image: `dinakepech/ecup26-minilm-s2:1.0` (PyTorch 2.5.1 CUDA 12.4,
Transformers 4.57.6, safetensors 0.6.2, pandas 2.2.3, PyArrow 21.0.0).

Environment overrides: `PM_PAIR_BATCH_SIZE`, `PM_MIN_PAIR_BATCH_SIZE`,
`PM_MAX_LENGTH`, `PM_PUBLIC_SOFT_LIMIT_SECONDS`,
`PM_PRIVATE_SOFT_LIMIT_SECONDS`, and `PM_DEADLINE_RESERVE_SECONDS`.
