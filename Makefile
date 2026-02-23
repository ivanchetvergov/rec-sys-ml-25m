# Makefile for RecSys Project
# Python environment: /Users/ivan/myvenv

PYTHON = /Users/ivan/myvenv/bin/python
PIP = /Users/ivan/myvenv/bin/pip

# Default dataset tag (use latest or set via environment variable)
DATASET_TAG ?= ml_v_20260215_184134
MLFLOW_PORT ?= 5000

.PHONY: help install preprocess train-popularity train-popularity-sample mlflow-ui mlflow-ui-alt mlflow-stop clean \
        backend frontend web web-down

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install                - Install dependencies"
	@echo ""
	@echo "Data preparation:"
	@echo "  make preprocess             - Run MovieLens preprocessing pipeline"
	@echo ""
	@echo "Web:"
	@echo "  make backend                - Run FastAPI backend locally (port 8000)"
	@echo "  make frontend               - Run Next.js frontend locally (port 3000)"
	@echo "  make web                    - Run backend + frontend via Docker Compose"
	@echo "  make web-down               - Stop Docker Compose"
	@echo ""
	@echo "Model training:"
	@echo "  make train-popularity       - Train popularity baseline (full data)"
	@echo "  make train-popularity-sample - Train popularity baseline (10% sample for testing)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make mlflow-ui              - Start MLflow UI on http://localhost:5000"
	@echo "  make mlflow-ui-alt          - Start MLflow UI on http://localhost:5001 (if 5000 is busy)"
	@echo "  make mlflow-stop            - Stop all MLflow processes"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean                  - Clean processed data and artifacts"

install:
	$(PIP) install -r requirements.txt

preprocess:
	$(PYTHON) src/pipeline/preprocess_pipeline.py

train-popularity:
	$(PYTHON) src/training/train_popularity.py \
		--dataset-tag $(DATASET_TAG) \
		--min-ratings 10 \
		--rating-weight 1.0 \
		--count-weight 1.0 \
		--relevance-threshold 4.0 \
		--k-values 5 10 20

train-popularity-sample:
	$(PYTHON) src/training/train_popularity.py \
		--dataset-tag $(DATASET_TAG) \
		--sample-frac 0.1 \
		--min-ratings 10 \
		--rating-weight 1.0 \
		--count-weight 1.0 \
		--relevance-threshold 4.0 \
		--k-values 5 10 20

mlflow-ui:
	@echo "Checking if port $(MLFLOW_PORT) is available..."
	@lsof -ti:$(MLFLOW_PORT) > /dev/null 2>&1 && \
		echo "⚠️  Port $(MLFLOW_PORT) is already in use. Stopping existing process..." && \
		kill -9 $$(lsof -ti:$(MLFLOW_PORT)) && \
		sleep 1 || true
	@echo "Starting MLflow UI at http://localhost:$(MLFLOW_PORT)"
	@echo "Press Ctrl+C to stop"
	@cd $(shell pwd) && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)

mlflow-ui-alt:
	@echo "Starting MLflow UI at http://localhost:5001 (alternative port)"
	@echo "Press Ctrl+C to stop"
	@lsof -ti:5001 > /dev/null 2>&1 && kill -9 $$(lsof -ti:5001) && sleep 1 || true
	@cd $(shell pwd) && mlflow ui --host 0.0.0.0 --port 5001

mlflow-stop:
	@echo "Stopping all MLflow processes..."
	@pkill -f "mlflow ui" || true
	@lsof -ti:5000 > /dev/null 2>&1 && kill -9 $$(lsof -ti:5000) || true
	@lsof -ti:5001 > /dev/null 2>&1 && kill -9 $$(lsof -ti:5001) || true
	@echo "✓ All MLflow processes stopped"

clean:
	rm -rf data/processed/feature_store/*
	rm -rf mlruns/
	@echo "✓ Cleaned processed data and artifacts"

# ─── Web ──────────────────────────────────────────────────────────────────────
backend:
	@echo "Starting FastAPI backend on http://localhost:8000"
	cd backend && pip install -r requirements.txt -q && uvicorn app.main:app --reload --port 8000

frontend:
	@echo "Starting Next.js frontend on http://localhost:3000"
	cd frontend && npm install && npm run dev

web:
	docker compose up --build

web-down:
	docker compose down
