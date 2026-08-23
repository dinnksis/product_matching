#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Pure symmetry ablation: both pair orientations, soft BCE, ELR=1 and a
# squared logit-gap penalty. Pairwise teacher-margin distillation is disabled.
exec torchrun \
  --standalone \
  --nproc_per_node="${LLM_NPROC:-1}" \
  scripts/train_llm_full.py \
  --data-dir "${LLM_DATA_DIR:-prepared/validation_splits_v1/llm}" \
  --model "${LLM_MODEL:-cross-encoder/mmarco-mMiniLMv2-L12-H384-v1}" \
  --human-validation-dir \
    "${LLM_HUMAN_VALIDATION_DIR:-prepared/validation_splits_v1/human}" \
  --human-items "${LLM_HUMAN_ITEMS:-data/items_human.parquet}" \
  --output-dir "${LLM_OUTPUT_DIR:-models/minilm_llm_full_elr1_sym_l01_5ep}" \
  --cache-dir "${LLM_CACHE_DIR:-artifacts/llm_full_cache}" \
  --epochs "${LLM_EPOCHS:-5}" \
  --batch-size "${LLM_BATCH_SIZE:-256}" \
  --eval-batch-size "${LLM_EVAL_BATCH_SIZE:-512}" \
  --learning-rate "${LLM_LEARNING_RATE:-5e-6}" \
  --elr-beta 0.7 \
  --elr-lambda 1.0 \
  --symmetry-lambda "${SYMMETRY_LAMBDA:-0.1}" \
  --pairwise-margin-lambda 0 \
  --max-length "${LLM_MAX_LENGTH:-512}" \
  --serialization-variant S1_KEY_VALUE \
  --num-workers "${LLM_NUM_WORKERS:-8}" \
  "$@"
