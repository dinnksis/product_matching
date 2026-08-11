.PHONY: setup notebook report kaggle-run kaggle-dry-run submit-build submit-archive

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
