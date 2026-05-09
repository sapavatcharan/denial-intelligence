.PHONY: install test lint eval demo synth analyze clean

install:
	python -m pip install -U pip
	python -m pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check src tests

eval:
	python -m gabeo.cli eval

demo:
	streamlit run demo/streamlit_app.py

synth:
	python scripts/generate_synthetic.py --n 40 --out data/synthetic/claims.jsonl

analyze:
	python -m gabeo.cli analyze --in data/synthetic/claims.jsonl

clean:
	rm -rf .pytest_cache .ruff_cache .gabeo build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
