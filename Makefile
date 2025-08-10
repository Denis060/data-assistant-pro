.PHONY: help install install-dev test lint format type-check security clean run docker-build docker-run

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install production dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  test             Run tests"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black and isort"
	@echo "  type-check       Run type checking with mypy"
	@echo "  security         Run security checks"
	@echo "  clean            Clean up cache and temp files"
	@echo "  run              Start the Streamlit application"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  setup-pre-commit Setup pre-commit hooks"

# Installation
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -r requirements-dev.txt

# Testing
test:
	pytest --cov=modules --cov-report=term-missing --cov-report=html

test-verbose:
	pytest -v --cov=modules --cov-report=term-missing

# Code quality
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics

format:
	black .
	isort .

format-check:
	black --check .
	isort --check-only .

type-check:
	mypy modules/ --ignore-missing-imports

# Security
security:
	bandit -r . -x tests/
	safety check

# Maintenance
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build

# Development
run:
	streamlit run app.py

run-dev:
	ENVIRONMENT=development streamlit run app.py --server.runOnSave=true

# Docker
docker-build:
	docker build -t data-assistant-pro .

docker-run:
	docker run -p 8501:8501 data-assistant-pro

docker-compose-up:
	docker-compose up --build

docker-compose-down:
	docker-compose down

# Pre-commit
setup-pre-commit:
	pre-commit install
	pre-commit install --hook-type commit-msg

run-pre-commit:
	pre-commit run --all-files

# CI/CD simulation
ci-check: format-check lint type-check security test
	@echo "All CI checks passed!"

# Quick development setup
setup: install-dev setup-pre-commit
	@echo "Development environment setup complete!"

# Production deployment preparation
build-prod: clean ci-check docker-build
	@echo "Production build complete!"
