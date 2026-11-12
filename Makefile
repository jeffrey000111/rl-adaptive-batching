PY ?= .venv/bin/python

.PHONY: setup lint test docs legacy-repro sweep check

setup:  ## create .venv (Python 3.10+) and install the package with dev extras
	python3.10 -m venv .venv
	$(PY) -m pip install -e ".[dev,legacy,plots,rl]"

lint:
	$(PY) -m ruff check .

test:
	$(PY) -m pytest -q

docs:  ## regenerate every generated block in README.md / REPORT.md from results/
	$(PY) scripts/render_docs.py

legacy-repro:  ## re-run the fixed-batch check against the archived environment
	$(PY) legacy/repro_fixed_batch.py
	$(PY) scripts/render_docs.py

sweep:  ## fixed-batch-size sweep on the train split, figure, README table
	$(PY) scripts/fixed_batch_sweep.py
	$(PY) scripts/plot_fixed_batch.py
	$(PY) scripts/render_docs.py

check: lint test
