# Drift-Aware MLOps Pipeline

An end-to-end MLOps system for tabular binary classification that implements **feature-aware drift detection** and **policy-based automated retraining**.

## Research Problem

Production ML models degrade when input distributions change over time. This project studies whether a drift-aware retraining policy can preserve predictive performance more efficiently than naive or static retraining.

## Key Features

- **Two-layer drift detection**: Feature-level (KS test / Chi-square) + global severity score
- **Policy-based retraining**: Retrain only when severity, data, and cooldown conditions are met
- **Experiment tracking**: MLflow for parameters, metrics, and model versioning
- **Monitoring**: Prometheus + Grafana dashboards for drift scores, latency, and retraining events
- **Containerized deployment**: Docker + FastAPI inference service with model hot-swap
- **CI/CD**: GitHub Actions (lint/test) + local Makefile (retraining triggers)

## Dataset

UCI Adult Income dataset (~48,842 records, 14 features, binary classification).

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run data pipeline
python -m src.data_processing

# Train baseline model
python -m src.train

# Run drift simulation
python -m src.drift_simulation

# Start inference API
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Project Structure

```
├── configs/          # YAML configuration files
├── src/              # Core source code (data, drift, policy, training)
├── api/              # FastAPI inference service
├── docker/           # Dockerfiles and compose
├── monitoring/       # Prometheus + Grafana configs
├── tests/            # Test suite
├── reports/          # Experiment results and figures
├── data/             # Raw, processed, and batch data (gitignored)
└── mlflow/           # MLflow tracking (gitignored)
```

## Milestones

1. Data loading, preprocessing & drift simulation
2. Baseline model training & MLflow logging
3. Feature-level drift detection
4. Policy engine for retraining decisions
5. Inference API & Dockerization
6. CI/CD automation
7. Prometheus & Grafana monitoring
8. Experimental comparison & report
