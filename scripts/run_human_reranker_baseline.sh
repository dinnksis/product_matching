#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

usage() {
  cat <<'EOF'
Usage:
  scripts/run_human_reranker_baseline.sh --list
  scripts/run_human_reranker_baseline.sh MODEL_ALIAS [extra trainer arguments]
  scripts/run_human_reranker_baseline.sh all [extra trainer arguments]

Aliases:
  gte             Alibaba-NLP/gte-multilingual-reranker-base
  jina-v3.5       jinaai/jina-reranker-v3.5 (custom LBNL backend)
  jina-v2         jinaai/jina-reranker-v2-base-multilingual
  bge-v2-m3       BAAI/bge-reranker-v2-m3
  qwen-0.6b       Qwen/Qwen3-Reranker-0.6B (full fine-tune)
  qwen-4b         Qwen/Qwen3-Reranker-4B (LoRA on one H100 80GB)
  rumodernbert    deepvk/RuModernBERT-base (new randomly initialized binary head)

Useful environment overrides:
  PREPARED_DIR, OUTPUT_DIR, TOKEN_CACHE_DIR, PYTHON_BIN, CUDA_VISIBLE_DEVICES
  EXPERIMENT_NAME, RUN_ID, DATASET_REF, ENV_FILE, GOOGLE_SERVICE_ACCOUNT_JSON_PATH
  SYNC_GOOGLE_SHEETS=0   keep artifacts but skip automatic experiments_v2 sync
  DRY_RUN=1              print both commands without running them
EOF
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

if [[ ${1:-} == "--list" ]]; then
  usage
  exit 0
fi
if [[ ${1:-} == "--help" || ${1:-} == "-h" || $# -eq 0 ]]; then
  usage
  exit 0
fi

if [[ $1 == "all" ]]; then
  shift
  if [[ -n ${OUTPUT_DIR:-} || -n ${RUN_ID:-} || -n ${EXPERIMENT_NAME:-} ]]; then
    printf '%s\n' \
      "OUTPUT_DIR, RUN_ID and EXPERIMENT_NAME must be unset for the all profile." >&2
    exit 2
  fi
  profiles=(gte jina-v3.5 jina-v2 bge-v2-m3 qwen-0.6b qwen-4b rumodernbert)
  for profile_name in "${profiles[@]}"; do
    "$0" "$profile_name" "$@"
  done
  exit 0
fi

requested_profile=$1
shift

for argument in "$@"; do
  case "$argument" in
    --config|--config=*|--model|--model=*|--model-backend|--model-backend=*|\
    --prepared-dir|--prepared-dir=*|--validation-split|--validation-split=*|\
    --output-dir|--output-dir=*|--token-cache-dir|--token-cache-dir=*)
      printf 'Use the documented environment variable instead of overriding %s.\n' \
        "$argument" >&2
      exit 2
      ;;
  esac
done

backend="cross_encoder"
trust_remote_code=0
profile_args=()
case "$requested_profile" in
  gte|Alibaba-NLP/gte-multilingual-reranker-base)
    profile="gte"
    cache_name="gte_multilingual_reranker_base"
    model="Alibaba-NLP/gte-multilingual-reranker-base"
    experiment_default="gte_multilingual_reranker_base_human_ft_v1"
    trust_remote_code=1
    profile_args=(
      --batch-size 192 --eval-batch-size 512 --gradient-accumulation 1
      --learning-rate 2e-5 --attention-implementation eager
      --no-gradient-checkpointing --max-grad-norm 0.5
      --dataloader-workers 16 --prefetch-factor 4
      --tokenization-batch-size 1024 --log-every 50
    )
    ;;
  jina-v3.5|jinaai/jina-reranker-v3.5)
    profile="jina_v3_5"
    model="jinaai/jina-reranker-v3.5"
    experiment_default="jina_reranker_v3_5_lbnl_human_ft_v1"
    trust_remote_code=1
    profile_args=(
      --model-backend jina_lbnl
      --batch-size 48 --eval-batch-size 96 --gradient-accumulation 4
      --learning-rate 2e-5 --attention-implementation sdpa
      --gradient-checkpointing --max-grad-norm 0.5
      --dataloader-workers 16 --prefetch-factor 4
      --tokenization-batch-size 512 --log-every 50
    )
    ;;
  jina-v2|jinaai/jina-reranker-v2-base-multilingual)
    profile="jina_v2"
    model="jinaai/jina-reranker-v2-base-multilingual"
    experiment_default="jina_reranker_v2_base_multilingual_human_ft_v1"
    trust_remote_code=1
    profile_args=(
      --model-load-kwarg use_flash_attn=false
      --batch-size 192 --eval-batch-size 512 --gradient-accumulation 1
      --learning-rate 2e-5 --attention-implementation eager
      --no-gradient-checkpointing --max-grad-norm 0.5
      --dataloader-workers 16 --prefetch-factor 4
      --tokenization-batch-size 1024 --log-every 50
    )
    ;;
  bge-v2-m3|BAAI/bge-reranker-v2-m3)
    profile="bge_v2_m3"
    model="BAAI/bge-reranker-v2-m3"
    experiment_default="bge_reranker_v2_m3_human_ft_v1"
    profile_args=(
      --batch-size 64 --eval-batch-size 192 --gradient-accumulation 3
      --learning-rate 2e-5 --attention-implementation sdpa
      --no-gradient-checkpointing --max-grad-norm 0.5
      --dataloader-workers 16 --prefetch-factor 4
      --tokenization-batch-size 1024 --log-every 50
    )
    ;;
  qwen-0.6b|Qwen/Qwen3-Reranker-0.6B)
    profile="qwen_0_6b"
    model="Qwen/Qwen3-Reranker-0.6B"
    experiment_default="qwen3_reranker_0_6b_full_human_ft_v1"
    backend="qwen"
    profile_args=(
      --training-mode full
      --batch-size 32 --eval-batch-size 64 --gradient-accumulation 6
      --learning-rate 2e-5 --attention-implementation sdpa --max-grad-norm 0.5
      --gradient-checkpointing --dataloader-workers 16 --prefetch-factor 4
      --tokenization-batch-size 512 --log-every 50
    )
    ;;
  qwen-4b|Qwen/Qwen3-Reranker-4B)
    profile="qwen_4b"
    model="Qwen/Qwen3-Reranker-4B"
    experiment_default="qwen3_reranker_4b_lora_human_ft_v1"
    backend="qwen"
    profile_args=(
      --training-mode lora --lora-rank 16 --lora-targets attention_mlp
      --batch-size 8 --eval-batch-size 32 --gradient-accumulation 24
      --learning-rate 1e-4 --attention-implementation sdpa --max-grad-norm 0.5
      --gradient-checkpointing --dataloader-workers 16 --prefetch-factor 4
      --tokenization-batch-size 512 --log-every 50
    )
    ;;
  rumodernbert|ruModernBERT|deepvk/RuModernBERT-base)
    profile="rumodernbert"
    model="deepvk/RuModernBERT-base"
    experiment_default="rumodernbert_base_random_head_human_ft_v1"
    profile_args=(
      --batch-size 192 --eval-batch-size 512 --gradient-accumulation 1
      --learning-rate 2e-5 --attention-implementation sdpa
      --no-gradient-checkpointing --max-grad-norm 0.5
      --dataloader-workers 16 --prefetch-factor 4
      --tokenization-batch-size 1024 --log-every 50
    )
    ;;
  *)
    printf 'Unknown model alias: %s\n\n' "$requested_profile" >&2
    usage >&2
    exit 2
    ;;
esac

cache_name="${cache_name:-$profile}"

if [[ -z ${PYTHON_BIN:-} ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

if [[ -z ${PREPARED_DIR:-} ]]; then
  if [[ -d "$ROOT_DIR/prepared/validation_splits_v1/human" ]]; then
    PREPARED_DIR="$ROOT_DIR/prepared/validation_splits_v1/human"
  elif [[ -d "$ROOT_DIR/data/validation_splits_v1/human" ]]; then
    PREPARED_DIR="$ROOT_DIR/data/validation_splits_v1/human"
  else
    PREPARED_DIR="$ROOT_DIR/prepared/validation_splits_v1/human"
  fi
fi

utc_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
started_at_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/model/baseline_${profile}_${utc_stamp}}"
TOKEN_CACHE_DIR="${TOKEN_CACHE_DIR:-$ROOT_DIR/artifacts/token_cache/$cache_name}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$experiment_default}"
RUN_ID="${RUN_ID:-server-${profile}-${utc_stamp}}"
DATASET_REF="${DATASET_REF:-alexproger23/product-matching-validation-splits-v1}"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/configs/cross_encoder_minilm_validation_baseline.json}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

common_validation_args=(
  --prepared-dir "$PREPARED_DIR"
  --validation-split iid=iid_validation_pairs.parquet
  --validation-split hard=hard_validation_pairs.parquet
  --validation-split ood=ood_validation_pairs.parquet
  --output-dir "$OUTPUT_DIR"
  --token-cache-dir "$TOKEN_CACHE_DIR"
  --model "$model"
  --epochs 1
  --sampling none
  --max-length 384
  --symmetric-validation
)

if [[ $backend == "qwen" ]]; then
  train_command=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/train_qwen_names.py"
    "${common_validation_args[@]}"
    "${profile_args[@]}"
  )
  if (( $# )); then
    train_command+=("$@")
  fi
else
  train_command=(
    "$PYTHON_BIN" "$ROOT_DIR/scripts/train_cross_encoder.py"
    --config "$CONFIG_PATH"
    "${common_validation_args[@]}"
  )
  if [[ $trust_remote_code -eq 1 ]]; then
    train_command+=(--trust-remote-code)
  fi
  train_command+=("${profile_args[@]}")
  if (( $# )); then
    train_command+=("$@")
  fi
fi

sync_command=(
  "$PYTHON_BIN" "$ROOT_DIR/scripts/sync_local_experiment_to_google_sheet.py"
  --report "$OUTPUT_DIR/training_report.json"
  --experiment "$EXPERIMENT_NAME"
  --model "$model"
  --dataset-ref "$DATASET_REF"
  --run-id "$RUN_ID"
  --started-at-utc "$started_at_utc"
)
if [[ -n ${ENV_FILE:-} ]]; then
  sync_command+=(--env-file "$ENV_FILE")
fi

printf 'Training profile: %s\nModel: %s\nOutput: %s\n' "$profile" "$model" "$OUTPUT_DIR"
printf 'Training command:\n'
print_command "${train_command[@]}"
if [[ ${SYNC_GOOGLE_SHEETS:-1} != "0" ]]; then
  printf 'Post-training experiments_v2 sync:\n'
  print_command "${sync_command[@]}"
fi
if [[ ${DRY_RUN:-0} == "1" ]]; then
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
"${train_command[@]}" 2>&1 | tee "$OUTPUT_DIR/training.log"

if [[ ${SYNC_GOOGLE_SHEETS:-1} != "0" ]]; then
  if ! "${sync_command[@]}"; then
    printf '%s\n' \
      "Training completed, but Google Sheets sync is pending. See $OUTPUT_DIR/google_sheets_sync.json and rerun the printed sync command." >&2
    exit 3
  fi
else
  printf 'Training completed; Google Sheets sync was disabled.\n'
fi
