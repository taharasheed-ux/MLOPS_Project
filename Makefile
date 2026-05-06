# MLOps Drift-Aware Pipeline Automation

.PHONY: help setup test train train-acs drift drift-acs experiments-acs ablation-acs mlflow build up down clean

help:
	@echo "Available commands:"
	@echo "  make setup    - Setup environment (install dependencies)"
	@echo "  make test     - Run test suite"
	@echo "  make train    - Run baseline training pipeline"
	@echo "  make train-acs - Run ACS baseline training pipeline"
	@echo "  make drift    - Run drift simulation and detection"
	@echo "  make drift-acs - Run ACS drift simulation and detection"
	@echo "  make experiments-acs - Run final ACS experiment suite"
	@echo "  make ablation-acs - Run ACS noise ablation diagnostics"
	@echo "  make mlflow   - Start local MLflow server"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start local infrastructure (API, Prometheus, Grafana)"
	@echo "  make down     - Stop local infrastructure"
	@echo "  make clean    - Remove cached files and pyc"

setup:
	pip install --upgrade pip
	pip install -r requirements.txt

test:
	python -m pytest tests/ -v

train:
	python -m src.train --run-name baseline_training

train-acs:
	SETTINGS_FILE=configs/settings_acs.yaml python -m src.train --run-name acs_baseline

retrain:
	python -m src.retrain

drift:
	python -m src.drift_simulation
	python -m src.drift_detection

drift-acs:
	SETTINGS_FILE=configs/settings_acs.yaml python -m src.drift_simulation
	SETTINGS_FILE=configs/settings_acs.yaml python -m src.drift_detection

experiments-acs:
	SETTINGS_FILE=configs/settings_acs.yaml python -m experiments.run_experiments

ablation-acs:
	SETTINGS_FILE=configs/settings_acs.yaml python -m experiments.run_noise_ablation

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

build:
	cd docker && docker compose build

up:
	cd docker && docker compose up -d

down:
	cd docker && docker compose down

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	find . -type d -name ".pytest_cache" -exec rm -r {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
