.PHONY: help install clean run dev test lint format check setup dirs

# Configuration
PYTHON := python
VENV := venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
UVICORN := $(VENV_BIN)/uvicorn
PYTEST := $(VENV_BIN)/pytest
APP := app.main:app
HOST := 0.0.0.0
PORT := 8000

# Default target
.DEFAULT_GOAL := help

help:
	@echo ""
	@echo " VideoMind - available commands:"
	@echo ""
	@echo "  Setup"
	@echo "    make setup        Create venv + install all dependencies"
	@echo "    make install      Install dependencies into active venv"
	@echo "    make dirs         Create required runtime directories"
	@echo ""
	@echo "  Run"
	@echo "    make run          Start FastAPI in production mode"
	@echo "    make dev          Start FastAPI with hot reload"
	@echo "    make gradio       Launch Gradio demo"
	@echo ""
	@echo "  Test"
	@echo "    make test         Run all tests"
	@echo "    make test-v       Run tests with verbose output"
	@echo ""
	@echo "  Code quality"
	@echo "    make lint         Run ruff linter"
	@echo "    make format       Auto-format with ruff"
	@echo "    make check        lint + format + test (full CI check)"
	@echo ""
	@echo "  Docker"
	@echo "    make docker-build Build Docker image"
	@echo "    make docker-run   Run container locally"
	@echo "    make docker-stop  Stop running container"
	@echo ""
	@echo "  Utils"
	@echo "    make clean        Remove cache and temp files"
	@echo "    make clean-all    Remove venv too"
	@echo ""

# Setup
setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	@echo "Setup complete. Activate the virtual environment with: source $(VENV)/bin/activate"

install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

dirs:
	mkdir -p /tmp/videomind/uploads /tmp/videomind/outputs
	@echo "Runtime directories created at /tmp/videomind/uploads and /tmp/videomind/outputs"

# Run
run:
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT)

dev:
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT) --reload

gradio:
	$(VENV_BIN)/python gradio_demo/app.py

# Test
test:
	$(PYTEST) tests/
test-v:
	$(PYTEST) tests/ -v

# Code quality
lint:
	$(VENV_BIN)/ruff check app/ tests/ gradio_demo/
format:
	$(VENV_BIN)/ruff format app/ tests/ gradio_demo/
	$(VENV_BIN)/ruff check --fix app/ tests/
check: lint format test
	@echo "All checks passed!"

# MLflow
mlflow:
	$(VENV_BIN)/mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns > mlflow.log 2>&1 &
	@echo "MLflow UI running at http://localhost:5000 (logs in mlflow.log)"
	
# Docker
docker-build:
	docker build -t videomind:latest .
docker-run:
	docker run -d --name videomind -p ${PORT}:$(PORT) videomind:latest
docker-stop:
	docker stop videomind && docker rm videomind

# Docker Compose
compose-up:
	docker compose up --build -d
	@echo "Services starting..."
	@echo "API:    http://localhost:8000"
	@echo "Gradio: http://localhost:7860"
	@echo "MLflow: http://localhost:5000"
compose-down:
	docker compose down
compose-logs:
	docker compose logs -f
compose-ps:
	docker compose ps

# Clean
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	@echo "Cache cleaned."

clean-all: clean
	rm -rf $(VENV)
	@echo "venv removed."