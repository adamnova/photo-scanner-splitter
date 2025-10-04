.PHONY: help install install-dev format lint type-check test test-coverage clean all

help:  ## Show this help message
	@echo 'Usage: make [target]'
	@echo ''
	@echo 'Available targets:'
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install runtime dependencies
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install development dependencies
	pip install -r requirements.txt
	pip install -r requirements-dev.txt
	pip install -e .

format:  ## Format code with black
	black photo_splitter/ tests/

format-check:  ## Check code formatting without modifying
	black --check --diff photo_splitter/ tests/

lint:  ## Run ruff linter with auto-fix
	ruff check photo_splitter/ tests/ --fix

lint-check:  ## Run ruff linter without auto-fix
	ruff check photo_splitter/ tests/

type-check:  ## Run mypy type checker
	mypy photo_splitter/ --install-types --non-interactive || true

test:  ## Run tests
	python -m unittest discover tests -v

test-coverage:  ## Run tests with coverage report
	coverage run -m unittest discover tests -v
	coverage report -m
	coverage html
	@echo "Coverage report generated in htmlcov/index.html"

clean:  ## Clean build artifacts and cache
	rm -rf build/ dist/ *.egg-info
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete

all: format lint type-check test  ## Run all quality checks and tests

pre-commit-install:  ## Install pre-commit hooks
	pip install pre-commit
	pre-commit install

pre-commit-run:  ## Run pre-commit on all files
	pre-commit run --all-files

ci: format-check lint-check test  ## Run CI checks locally
