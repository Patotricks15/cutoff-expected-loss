.PHONY: install test notebook clean

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/06_cutoff_economics.py -o notebooks/06_cutoff_economics.ipynb
	jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=credit-venv --ExecutePreprocessor.timeout=1200 notebooks/06_cutoff_economics.ipynb

clean:
	rm -rf .pytest_cache __pycache__ src/*.egg-info
