.PHONY: setup notebook report prepare-human prepare-mxbai-balanced analyze-validation train-qwen train-cross-encoder kaggle-google-credentials kaggle-google-credentials-dry-run kaggle-train-build kaggle-train-data kaggle-cross-build kaggle-cross-dry-run kaggle-cross-run kaggle-mxbai-build kaggle-mxbai-data-dry-run kaggle-mxbai-dry-run kaggle-mxbai-run kaggle-run kaggle-dry-run submit-build submit-archive

CROSS_ENCODER_CONFIG ?= configs/cross_encoder_minilm.json
VALIDATION_PREDICTIONS ?= data/runs/validation_predictions_v1.parquet

setup:
	uv sync

notebook:
	uv run python scripts/create_eda_notebook.py

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

kaggle-google-credentials:
	uv run python scripts/push_google_sheets_credentials_dataset.py

kaggle-google-credentials-dry-run:
	uv run python scripts/push_google_sheets_credentials_dataset.py --dry-run

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

kaggle-run:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make kaggle-run NOTEBOOK=notebooks/train.ipynb"; exit 2)
	uv run python scripts/run_kaggle_notebook.py "$(NOTEBOOK)"

kaggle-dry-run:
	@test -n "$(NOTEBOOK)" || (echo "Usage: make kaggle-dry-run NOTEBOOK=notebooks/train.ipynb"; exit 2)
	uv run python scripts/run_kaggle_notebook.py "$(NOTEBOOK)" --dry-run

submit-build:
	uv run python scripts/build_qwen3_vllm_submit.py

submit-archive:
	uv run python scripts/build_qwen3_vllm_submit.py --archive-only
