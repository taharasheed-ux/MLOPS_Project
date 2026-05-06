# Drift-Aware MLOps Pipeline

An end-to-end MLOps and research pipeline for tabular income prediction under dataset shift. The project integrates **MLflow**, **Docker**, **Prometheus**, **Grafana**, **GitHub Actions CI/CD**, FastAPI deployment, drift detection, policy-based retraining, and a multi-year ACS research study.

## Research Problem

Production ML models degrade when input distributions and feature-label relationships change over time. This project studies when adaptive retraining improves over static deployment, and when retraining becomes harmful because incoming labels behave like synthetic noise rather than a learnable new regime.

## Key Features

- **Multi-year ACS research pipeline**: 2016 baseline training with 2017-2018 temporal evaluation
- **Two-layer drift detection**: Feature-level KS/chi-square tests plus global severity scoring
- **Policy-based retraining**: Persistent-drift, cooldown, sample-count, and rolling-window gates
- **Experiment tracking**: MLflow logging for baseline and retrained models
- **Monitoring**: Prometheus metrics and provisioned Grafana dashboards
- **Containerized deployment**: Docker Compose stack for API, MLflow, Prometheus, and Grafana
- **CI/CD**: GitHub Actions for linting, tests, artifact smoke checks, and Docker validation
- **Research diagnostics**: Noise ablation and temporal regime feature-importance analysis

## Dataset

The project supports both UCI Adult and Folktables ACS Income. The final research run uses multi-year ACS data:

- 2016 baseline training pool
- 2017-2018 temporally shifted evaluation pool
- 12 drift batches under learnable temporal/covariate/feature drift
- separate label-flip stress-test ablation

## Quick Start

```bash
# Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run data pipeline
python -m src.data_processing

# Run ACS pipeline (programmatic fetch + local snapshot cache)
SETTINGS_FILE=configs/settings_acs.yaml python -m src.data_processing

# Train baseline model
SETTINGS_FILE=configs/settings_acs.yaml python -m src.train --run-name acs_baseline

# Run drift simulation
SETTINGS_FILE=configs/settings_acs.yaml python -m src.drift_simulation

# Run final ACS experiment
SETTINGS_FILE=configs/settings_acs.yaml python -m experiments.run_experiments

# Run noise ablation diagnostic
SETTINGS_FILE=configs/settings_acs.yaml python -m experiments.run_noise_ablation

# Start inference API
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Full Local MLOps Stack

```bash
cd docker
docker compose up -d --build
```

Services:

- FastAPI: `http://localhost:8080/docs`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

Grafana default login:

- username: `admin`
- password: `admin`

## Project Structure

```
├── configs/          # YAML configuration files
├── src/              # Core source code (data, drift, policy, training)
├── api/              # FastAPI inference service
├── docker/           # Dockerfiles and compose
├── monitoring/       # Prometheus + Grafana configs
├── tests/            # Test suite
├── reports/          # Experiment results and figures
├── paper/            # IEEE research paper and generated figures
├── data/             # Raw, processed, and batch data (gitignored)
└── mlruns/           # MLflow tracking artifacts (gitignored)
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

## Requirements Mapping

See [MLOPS_REQUIREMENTS_AUDIT.md](MLOPS_REQUIREMENTS_AUDIT.md) for the full evaluator-facing checklist that maps project requirements to repository evidence.

## Final Research Artifacts

- Paper: `paper/ieee_drift_aware_mlops.tex`
- Main report: `reports/results_acs.md`
- Noise ablation: `reports/diagnostics/acs_noise_ablation/`
- Temporal regime diagnostic: `reports/diagnostics/acs_regime/`
- Paper figures: `paper/figures/`

## ACS Dataset

- ACS loading uses `folktables` with profile `configs/settings_acs.yaml`.
- The multi-year run uses `survey_years: [2016, 2017, 2018]`.
- Data and model artifacts are intentionally gitignored.
