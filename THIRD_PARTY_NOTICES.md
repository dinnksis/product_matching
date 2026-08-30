# Third-party models and software

This repository contains participant-authored training, evaluation and submission
code for E-CUP 2026. Organizer datasets and large model weights are not committed.
Fine-tuned checkpoints remain derivative works of their upstream models and retain
the corresponding notices.

## Upstream models used in final submissions

| Component | Upstream model | Declared license | Use |
|---|---|---|---|
| BGE | [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3) | Apache-2.0 | mandatory full-coverage backbone |
| MiniLM | [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1/tree/1427fd652930e4ba29e8149678df786c240d8825) | Apache-2.0 | full or routed specialist |
| RuModernBERT | [`deepvk/RuModernBERT-base`](https://huggingface.co/deepvk/RuModernBERT-base) | Apache-2.0 | full or 5% routed specialist |

MiniLM was pinned during the original training chain to revision
`1427fd652930e4ba29e8149678df786c240d8825`. Exact BGE and RuModernBERT upstream
revisions must be copied from the competition-pretrain manifests when that team
contribution is merged; model names alone must not be treated as a complete
provenance record.

Apache License 2.0 permits use, modification and redistribution subject to its
conditions, including preservation of copyright, license and NOTICE information
where applicable. The upstream model cards and repositories remain authoritative.

## Main open-source runtime dependencies

| Project | License | Role |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | BSD-3-Clause | training and CUDA inference |
| [Transformers](https://github.com/huggingface/transformers) | Apache-2.0 | model/tokenizer implementations |
| [SentenceTransformers](https://github.com/huggingface/sentence-transformers) | Apache-2.0 | optimized CrossEncoder inference |
| [safetensors](https://github.com/huggingface/safetensors) | Apache-2.0 | checkpoint storage |
| [CatBoost](https://github.com/catboost/catboost) | Apache-2.0 | benefit routers |
| [pandas](https://github.com/pandas-dev/pandas) | BSD-3-Clause | tabular processing |
| [NumPy](https://github.com/numpy/numpy) | BSD-3-Clause | numerical processing |
| [Apache Arrow](https://github.com/apache/arrow) | Apache-2.0 | Parquet I/O |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn) | BSD-3-Clause | metrics and validation utilities |
| [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz) | MIT | optional lexical router features |

Dependency versions are fixed by the relevant requirements files and Dockerfiles.
The final inference pipeline does not call proprietary hosted APIs and runs fully
offline. NVIDIA CUDA is supplied by the competition runtime/base image; Docker
Desktop, IDEs and cloud notebook interfaces are development tools rather than
runtime dependencies of the submitted solution.

## Data and relabeling

The human and probabilistic LLM labels used for final training were supplied by
the competition organizers and are governed by the competition terms. They are
not redistributed here. Experimental open-model labeling code that did not feed
the final checkpoints is documented separately and must not be mistaken for the
provenance of the final submitted weights.
