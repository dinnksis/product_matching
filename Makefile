.PHONY: setup notebook attributes-notebook report prepare-human train-qwen train-cross-encoder kaggle-train-build kaggle-train-data kaggle-cross-build kaggle-cross-dry-run kaggle-cross-run kaggle-run kaggle-dry-run submit-build submit-archive submit-jina-build submit-jina-archive

CROSS_ENCODER_CONFIG ?= configs/cross_encoder_minilm.json

setup:
	uv sync

notebook:
	uv run python scripts/create_eda_notebook.py

attributes-notebook:
	uv run python scripts/create_attributes_analysis_notebook.py

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

train-qwen:
	torchrun --standalone --nproc_per_node=2 scripts/train_qwen_names.py $(TRAIN_ARGS)

train-cross-encoder:
	torchrun --standalone --nproc_per_node=2 scripts/train_cross_encoder.py \
		--config "$(CROSS_ENCODER_CONFIG)" $(TRAIN_ARGS)

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

submit-jina-build:
	uv run python scripts/build_jina_submit.py

submit-jina-archive:
	uv run python scripts/build_jina_submit.py --archive-only
.PHONY: embedding-boosting-dry-run embedding-boosting-run embedding-boosting-monitor

embedding-boosting-dry-run:
	python scripts/run_embedding_boosting_kaggle.py --dry-run

embedding-boosting-run:
	python scripts/run_embedding_boosting_kaggle.py

embedding-boosting-monitor:
	python scripts/run_embedding_boosting_kaggle.py --monitor-existing
