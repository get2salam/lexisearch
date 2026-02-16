.PHONY: help install dev test lint format typecheck clean docs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

dev: ## Install with all dev dependencies
	pip install -e ".[dev,all]"
	pre-commit install

test: ## Run tests with coverage
	pytest --tb=short -q --cov=lexisearch --cov-report=term-missing

test-verbose: ## Run tests with verbose output
	pytest -v --tb=long --cov=lexisearch

lint: ## Run linter
	ruff check lexisearch/ tests/

format: ## Format code
	ruff format lexisearch/ tests/

format-check: ## Check code formatting
	ruff format --check lexisearch/ tests/

typecheck: ## Run type checker
	mypy lexisearch/ --ignore-missing-imports

clean: ## Remove build artifacts
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -f coverage.xml .coverage

all: lint format-check typecheck test ## Run all checks
