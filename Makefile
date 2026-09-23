.PHONY: install dev test lint format clean docs

# ─────────────────────────────────────────────────────────────────────────────
# Installation
# ─────────────────────────────────────────────────────────────────────────────

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	pre-commit install || true

all:
	pip install -e ".[all]"

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src/spectral_agent --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

# ─────────────────────────────────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

check: lint test

# ─────────────────────────────────────────────────────────────────────────────
# Cleaning (cross-platform via Python)
# ─────────────────────────────────────────────────────────────────────────────

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(d, True) for d in ['.pytest_cache', '.mypy_cache', '.ruff_cache', 'htmlcov', 'build', 'dist']]; [p.unlink() for p in pathlib.Path('.').rglob('*.pyc')]; [shutil.rmtree(str(d), True) for d in pathlib.Path('.').rglob('__pycache__')]"

# ─────────────────────────────────────────────────────────────────────────────
# Development Helpers
# ─────────────────────────────────────────────────────────────────────────────

run-example:
	python -m spectral_agent.examples.basic_ingestion

streamlit:
	streamlit run app/main.py

# ─────────────────────────────────────────────────────────────────────────────
# Documentation
# ─────────────────────────────────────────────────────────────────────────────

docs:
	@echo "Documentation is in docs/ folder"
