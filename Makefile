.PHONY: help setup test lint format run docker-build docker-up clean conda-setup conda-update

help:
	@echo "Available commands:"
	@echo "  conda-setup  Create conda environment"
	@echo "  conda-update Update conda environment"
	@echo "  setup        Install dependencies (after activating conda)"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  run          Run the API locally"
	@echo "  docker-build Build Docker images"
	@echo "  docker-up    Start services with Docker Compose"
	@echo "  clean        Clean up generated files"

conda-setup:
	conda env create -f environment.yml
	@echo "Activate environment with: conda activate customer_churn"

conda-update:
	conda env update -f environment.yml --prune

setup:
	pip install -e .
	pre-commit install

test:
	pytest tests/ -v --cov=src

lint:
	ruff check src/ tests/
	black --check src/ tests/

format:
	black src/ tests/
	ruff check --fix src/ tests/

run:
	uvicorn src.api.main:app --reload

run-dashboard:
	streamlit run dashboard/streamlit_app.py

run-mlflow:
	mlflow server --host 0.0.0.0 --port 8000

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov .pytest_cache