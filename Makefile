.PHONY: setup notebook attributes-notebook validation-audit-notebooks validation-split-audit error-pattern-audit report prepare-human prepare-mxbai-balanced analyze-validation train-qwen train-cross-encoder train-llm-full train-llm-full-margin kaggle-google-credentials kaggle-google-credentials-dry-run kaggle-significance-baseline kaggle-significance-baseline-dry-run kaggle-sheets-init kaggle-sheets-retry kaggle-train-build kaggle-train-data kaggle-cross-build kaggle-cross-dry-run kaggle-cross-run kaggle-mxbai-build kaggle-mxbai-data-dry-run kaggle-mxbai-dry-run kaggle-mxbai-run kaggle-minilm-balanced-build kaggle-minilm-balanced-dry-run kaggle-minilm-balanced-run kaggle-validation-data-dry-run kaggle-validation-data kaggle-minilm-validation-build kaggle-minilm-validation-dry-run kaggle-minilm-validation-run kaggle-run kaggle-dry-run architecture-baselines-build architecture-baselines-dry-run architecture-baselines-run architecture-baselines-download architecture-baselines-summary serialization-ablation-build serialization-ablation-dry-run serialization-ablation-run serialization-ablation-monitor submit-build submit-archive submit-jina-build submit-jina-archive submit-minilm-s2 submit-bge submit-bge-minilm embedding-boosting-dry-run embedding-boosting-run embedding-boosting-monitor submit-embedding-build submit-embedding-archive

CROSS_ENCODER_CONFIG ?= configs/cross_encoder_minilm.json
VALIDATION_PREDICTIONS ?= data/runs/validation_predictions_v1.parquet
LLM_NPROC ?= 1

setup:
	uv sync

notebook:
	uv run python scripts/create_eda_notebook.py

attributes-notebook:
	uv run python scripts/create_attributes_analysis_notebook.py

validation-audit-notebooks:
	uv run python scripts/create_validation_audit_notebooks.py

validation-split-audit: validation-audit-notebooks
	uv run jupyter nbconvert --execute --to notebook --inplace \
		--ExecutePreprocessor.timeout=7200 notebooks/03_validation_split_audit.ipynb

error-pattern-audit: validation-audit-notebooks
	uv run jupyter nbconvert --execute --to notebook --inplace \
		--ExecutePreprocessor.timeout=3600 notebooks/04_error_pattern_analysis.ipynb

report: notebook
	mkdir -p .cache/matplotlib reports
	MPLCONFIGDIR=.cache/matplotlib uv run jupyter nbconvert \
		--execute --to notebook --inplace \
		--ExecutePreprocessor.timeout=900 \
		notebooks/01_human_data_eda.ipynb
	MPLCONFIGDIR=.cache/matplotlib uv run jupyter nbconvert \
		--to html --no-input \
		--output human_data_eda --output-dir reports \
		notebooks/01_human_data_eda.ipynb

prepare-human:
	python scripts/prepare_human_data.py

prepare-mxbai-balanced:
	uv run python scripts/prepare_balanced_llm_data.py

analyze-validation:
	uv run python scripts/analyze_validation_predictions.py \
		"$(VALIDATION_PREDICTIONS)"

train-qwen:
	torchrun --standalone --nproc_per_node=2 scripts/train_qwen_names.py $(TRAIN_ARGS)

train-cross-encoder:
	torchrun --standalone --nproc_per_node=2 scripts/train_cross_encoder.py \
		--config "$(CROSS_ENCODER_CONFIG)" $(TRAIN_ARGS)

train-llm-full:
	torchrun --standalone --nproc_per_node="$(LLM_NPROC)" \
		scripts/train_llm_full.py $(TRAIN_ARGS)

train-llm-full-margin:
	LLM_NPROC="$(LLM_NPROC)" scripts/run_llm_full_margin_distillation.sh \
		$(TRAIN_ARGS)

kaggle-google-credentials:
	uv run python scripts/push_google_sheets_credentials_dataset.py

kaggle-google-credentials-dry-run:
	uv run python scripts/push_google_sheets_credentials_dataset.py --dry-run

kaggle-significance-baseline:
	uv run python scripts/push_minilm_significance_baseline_dataset.py

kaggle-significance-baseline-dry-run:
	uv run python scripts/push_minilm_significance_baseline_dataset.py --dry-run

kaggle-sheets-init:
	uv run python scripts/initialize_google_sheets_schema.py

kaggle-sheets-retry:
	@test -n "$(KERNEL)" || (echo "Usage: make kaggle-sheets-retry KERNEL=owner/slug"; exit 2)
	uv run python scripts/sync_kaggle_experiment_to_google_sheet.py "$(KERNEL)"

kaggle-train-build:
	uv run python scripts/create_qwen_training_notebook.py

kaggle-train-data:
	uv run python scripts/push_kaggle_training_dataset.py

kaggle-cross-build:
	uv run python scripts/create_cross_encoder_training_notebook.py \
		--config "$(CROSS_ENCODER_CONFIG)"

kaggle-cross-dry-run:
	uv run python scripts/run_cross_encoder_kaggle.py \
		--config "$(CROSS_ENCODER_CONFIG)" --dry-run

kaggle-cross-run:
	uv run python scripts/run_cross_encoder_kaggle.py \
		--config "$(CROSS_ENCODER_CONFIG)"

kaggle-mxbai-build:
	uv run python scripts/create_mxbai_training_notebook.py

kaggle-mxbai-data-dry-run:
	uv run python scripts/push_mxbai_training_dataset.py --dry-run

kaggle-mxbai-dry-run:
	uv run python scripts/run_mxbai_kaggle.py --dry-run

kaggle-mxbai-run:
	uv run python scripts/run_mxbai_kaggle.py

kaggle-minilm-balanced-build:
	uv run python scripts/create_minilm_balanced_training_notebook.py

kaggle-minilm-balanced-dry-run:
	uv run python scripts/run_minilm_balanced_kaggle.py --dry-run

kaggle-minilm-balanced-run:
	uv run python scripts/run_minilm_balanced_kaggle.py

kaggle-validation-data-dry-run:
	uv run python scripts/push_validation_splits_dataset.py --dry-run

kaggle-validation-data:
	uv run python scripts/push_validation_splits_dataset.py

kaggle-minilm-validation-build:
	uv run python scripts/create_minilm_validation_baseline_notebook.py

kaggle-minilm-validation-dry-run:
	uv run python scripts/run_minilm_validation_baseline_kaggle.py --dry-run

kaggle-minilm-validation-run:
	uv run python scripts/run_minilm_validation_baseline_kaggle.py

architecture-baselines-build:
	uv run python scripts/create_architecture_baseline_notebooks.py

architecture-baselines-dry-run:
	uv run python scripts/run_architecture_baseline_kaggle.py all --dry-run

architecture-baselines-run:
	@test -n "$(ARCH_PROFILE)" || (echo "Usage: make architecture-baselines-run ARCH_PROFILE=gte"; exit 2)
	uv run python scripts/run_architecture_baseline_kaggle.py "$(ARCH_PROFILE)"

architecture-baselines-download:
	@test -n "$(ARCH_PROFILE)" || (echo "Usage: make architecture-baselines-download ARCH_PROFILE=gte"; exit 2)
	uv run python scripts/run_architecture_baseline_kaggle.py "$(ARCH_PROFILE)" --download-existing

architecture-baselines-summary:
	uv run python scripts/summarize_architecture_baselines.py

kaggle-minilm-pretrain-checkpoint-dry-run:
	uv run python scripts/push_minilm_pretrain_checkpoint_dataset.py --dry-run

kaggle-minilm-pretrain-checkpoint:
	uv run python scripts/push_minilm_pretrain_checkpoint_dataset.py

kaggle-minilm-pretrain-human-ft-dry-run:
	uv run python scripts/run_minilm_llm_pretrain_human_ft_kaggle.py --dry-run

kaggle-minilm-pretrain-human-ft-run:
	uv run python scripts/run_minilm_llm_pretrain_human_ft_kaggle.py

kaggle-run:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make kaggle-run NOTEBOOK=notebooks/train.ipynb"; exit 2)
	uv run python scripts/run_kaggle_notebook.py "$(NOTEBOOK)"

kaggle-dry-run:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make kaggle-dry-run NOTEBOOK=notebooks/train.ipynb"; exit 2)
	uv run python scripts/run_kaggle_notebook.py "$(NOTEBOOK)" --dry-run

serialization-ablation-build:
	uv run python scripts/create_serialization_ablation_notebook.py

serialization-ablation-dry-run:
	uv run python scripts/run_serialization_ablation_kaggle.py --dry-run

serialization-ablation-run:
	uv run python scripts/run_serialization_ablation_kaggle.py

serialization-ablation-monitor:
	uv run python scripts/run_serialization_ablation_kaggle.py --monitor-existing

submit-build:
	uv run python scripts/build_qwen3_vllm_submit.py

submit-archive:
	uv run python scripts/build_qwen3_vllm_submit.py --archive-only

submit-jina-build:
	uv run python scripts/build_jina_submit.py

submit-jina-archive:
	uv run python scripts/build_jina_submit.py --archive-only

submit-minilm-s2:
	python scripts/build_minilm_s2_submit.py

submit-bge:
	python scripts/build_bge_reranker_submit.py

submit-bge-minilm:
	python scripts/build_bge_minilm_ensemble_submit.py

embedding-boosting-dry-run:
	python scripts/run_embedding_boosting_kaggle.py --dry-run

embedding-boosting-run:
	python scripts/run_embedding_boosting_kaggle.py

embedding-boosting-monitor:
	python scripts/run_embedding_boosting_kaggle.py --monitor-existing

submit-embedding-build:
	python scripts/build_embedding_catboost_submit.py

submit-embedding-archive:
	python scripts/build_embedding_catboost_submit.py --archive-only
