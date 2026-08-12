# Qwen3 vLLM competition runtime

This image replaces the `vllm/vllm-openai:v0.14.0` server entrypoint with
`/usr/bin/env`. It adds no filesystem layers and keeps the same vLLM/CUDA
runtime.

Build and publish for the competition's `linux/amd64` worker:

```bash
docker buildx build \
  --platform linux/amd64 \
  --tag powpowpow12/ecup26-qwen3-vllm:0.14.0 \
  --push \
  docker/qwen3-vllm-runtime
```

The Docker Hub repository must be public. After publishing, put the complete
image reference into the submission's `metadata.json`.
