# Makefile for RecSys Project
# Python environment: /Users/ivan/myvenv

PYTHON = /Users/ivan/myvenv/bin/python
PIP = /Users/ivan/myvenv/bin/pip

# Default dataset tag (use latest or set via environment variable)
DATASET_TAG ?= ml_v_20260215_184134

.PHONY: help install preprocess train-popularity train-popularity-sample mlflow-ui clean

help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install                - Install dependencies"
	@echo ""
	@echo "Data preparation:"
	@echo "  make preprocess             - Run MovieLens preprocessing pipeline"
	@echo ""
	@echo "Model training:"
	@echo "  make train-popularity       - Train popularity baseline (full data)"
	@echo "  make train-popularity-sample - Train popularity baseline (10% sample for testing)"
	@echo ""
	@echo "Monitoring:"
	@echo "  make mlflow-ui              - Start MLflow UI on http://localhost:5000"
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
	@echo "Starting MLflow UI at http://localhost:5000"
	@echo "Press Ctrl+C to stop"
	cd $(shell pwd) && mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -rf data/processed/feature_store/*
	rm -rf mlruns/
	@echo "✓ Cleaned processed data and artifacts"
