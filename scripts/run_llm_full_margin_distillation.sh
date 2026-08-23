#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Controlled alternative to the original ELR=3 run: soft BCE, weaker ELR=1,
# and same-category Margin-Huber distillation. Extra CLI arguments are appended
# last so a server launch can opt into --resume or override one setting.
exec torchrun \
  --standalone \
  --nproc_per_node="${LLM_NPROC:-1}" \
  scripts/train_llm_full.py \
  --data-dir "${LLM_DATA_DIR:-prepared/validation_splits_v1/llm}" \
  --model "${LLM_MODEL:-cross-encoder/mmarco-mMiniLMv2-L12-H384-v1}" \
  --human-validation-dir \
    "${LLM_HUMAN_VALIDATION_DIR:-prepared/validation_splits_v1/human}" \
  --human-items "${LLM_HUMAN_ITEMS:-data/items_human.parquet}" \
  --output-dir \
    "${LLM_OUTPUT_DIR:-models/minilm_llm_full_elr1_margin_l01_5ep}" \
  --cache-dir "${LLM_CACHE_DIR:-artifacts/llm_full_cache}" \
  --epochs "${LLM_EPOCHS:-5}" \
  --batch-size "${LLM_BATCH_SIZE:-256}" \
  --eval-batch-size "${LLM_EVAL_BATCH_SIZE:-512}" \
  --learning-rate "${LLM_LEARNING_RATE:-5e-6}" \
  --elr-beta 0.7 \
  --elr-lambda 1.0 \
  --pairwise-margin-lambda "${PAIRWISE_MARGIN_LAMBDA:-0.1}" \
  --pairwise-margin-temperature "${PAIRWISE_MARGIN_TEMPERATURE:-1.0}" \
  --pairwise-margin-huber-delta "${PAIRWISE_MARGIN_HUBER_DELTA:-1.0}" \
  --pairwise-margin-logit-epsilon \
    "${PAIRWISE_MARGIN_LOGIT_EPSILON:-1e-4}" \
  --pairwise-margin-min-teacher-gap \
    "${PAIRWISE_MARGIN_MIN_TEACHER_GAP:-0.0}" \
  --max-length "${LLM_MAX_LENGTH:-512}" \
  --serialization-variant S1_KEY_VALUE \
  --num-workers "${LLM_NUM_WORKERS:-8}" \
  "$@"
