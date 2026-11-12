PY ?= .venv/bin/python

.PHONY: setup lint test docs legacy-repro sweep baselines train figures cpu-latency check

setup:  ## create .venv (Python 3.10+) and install the package with dev extras
	python3.10 -m venv .venv
	$(PY) -m pip install -e ".[dev,legacy,plots,rl,measure]"

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

baselines:  ## grid-search every rule on dev (minutes)
	$(PY) scripts/tune_baselines.py --group primary
	$(PY) scripts/tune_baselines.py --group secondary
	$(PY) scripts/tune_baselines.py --group ablation_len128

train:  ## RL tuning, sufficiency check, all training runs, dev selection (about an hour)
	$(PY) scripts/run_rl.py --stage tune
	$(PY) scripts/check_training_sufficiency.py
	$(PY) scripts/run_rl.py --stage primary
	$(PY) scripts/run_rl.py --stage secondary
	$(PY) scripts/summarize_dev.py --group primary
	$(PY) scripts/summarize_dev.py --group secondary
	$(PY) scripts/plot_learning_curves.py --group primary
	$(PY) scripts/plot_learning_curves.py --group secondary
	$(PY) scripts/run_ablations.py --stage train

# Test evaluations are run once and refuse to overwrite existing results:
#   scripts/evaluate_test.py --group primary|secondary
#   scripts/run_ablations.py --stage test
#   scripts/posthoc.py --group primary [--stress]; --group secondary

figures:  ## figures that are drawn from results/ (no simulation)
	$(PY) scripts/plot_summary.py
	$(PY) scripts/plot_learning_curves.py --group primary
	$(PY) scripts/plot_learning_curves.py --group secondary

cpu-latency:  ## post-hoc: time bert-base-uncased on this CPU and fit the model (minutes)
	$(PY) scripts/measure_cpu_latency.py
	$(PY) scripts/fit_cpu_latency.py
	$(PY) scripts/render_docs.py

check: lint test
