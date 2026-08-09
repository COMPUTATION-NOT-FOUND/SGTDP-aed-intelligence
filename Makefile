.PHONY: check gates fmt fetch pipeline app

## Full quality pass — the only command that has to stay green.
check:
	uv run ruff format --check .
	uv run ruff check .
	uv run pytest

## Phase gates only.
gates:
	uv run pytest -m gate -v

fmt:
	uv run ruff format .
	uv run ruff check --fix .

## One-shot: freeze every raw input and write the manifest.
fetch:
	uv run python scripts/fetch_sources.py

pipeline:
	uv run python -m aed_siting.pipeline --all-towns --seed 42

app:
	uv run streamlit run app.py
