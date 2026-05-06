# MLOps Project Requirements Audit

This document maps the submitted project requirements to concrete repository evidence. It is intended as an evaluator-facing checklist and as a final sanity check before submission.

## 1. Project Overview

**Chosen problem:** Binary income prediction under dataset shift using Adult and ACS/Folktables data.

**Research track:** Track-II, technical research with implementation and methodological improvement.

**Core research problem:** Determine when drift-aware adaptive retraining improves over static deployment, and when retraining becomes harmful because new labels represent synthetic noise rather than a learnable regime.

**Primary final finding:** Policy-gated retraining improves over static deployment under learnable ACS drift while reducing retraining frequency compared with immediate retraining. Noise ablation shows retraining degrades under subgroup label-flip stress.

## 2. Mandatory Technical Requirements

| Requirement | Status | Repository evidence |
|---|---:|---|
| MLflow experiment tracking | Complete | `src/train.py`, `src/retrain.py`, `docker/Dockerfile.mlflow`, `docker/docker-compose.yml` |
| Docker containerization | Complete | `docker/Dockerfile`, `docker/Dockerfile.mlflow`, `docker/docker-compose.yml`, `.dockerignore` |
| Prometheus metrics collection | Complete | `api/app.py`, `docker/prometheus.yml`, `/metrics` endpoint |
| Grafana dashboard | Complete | `monitoring/grafana/dashboards/mlops_dashboard.json`, `monitoring/grafana/provisioning/` |
| CI/CD automation | Complete | `.github/workflows/ci.yml` runs linting, tests, artifact generation, compile checks, and Docker validation/build steps |
| Deployment | Complete locally | FastAPI service deploys through Docker Compose with API, MLflow, Prometheus, and Grafana |
| AWS deployment | Not used | Optional in the project guideline; local containerized deployment is implemented instead |

## 3. Experiment Tracking

MLflow is integrated into the training and retraining lifecycle:

- baseline training logs parameters, metrics, and model artifact in `src/train.py`
- retraining logs model version, number of samples, metrics, and artifacts in `src/retrain.py`
- MLflow can run locally through Docker Compose as service `mlflow`
- local MLflow can also be started with `make mlflow`

Primary URL when the stack is active:

```bash
http://localhost:5000
```

## 4. Model Packaging and Deployment

The inference service is packaged with Docker:

- `docker/Dockerfile` builds the FastAPI API image
- `.dockerignore` prevents local raw data, processed data, models, logs, reports, virtual environments, and MLflow artifacts from being copied into the Docker build context
- `docker/docker-compose.yml` starts API, MLflow, Prometheus, and Grafana
- API model artifacts are mounted from `models/` as read-only volumes

Run:

```bash
cd docker
docker compose up -d --build
```

Service URLs:

- API: `http://localhost:8080/health`
- API docs: `http://localhost:8080/docs`
- Prometheus metrics: `http://localhost:8080/metrics`
- MLflow: `http://localhost:5000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## 5. Monitoring and Logging

The API exports Prometheus metrics:

- `prediction_requests_total`
- `prediction_latency_seconds`
- `current_model_version`
- `drift_severity`
- `retrain_events_total`

Prometheus scrapes the API through `docker/prometheus.yml`. Grafana is provisioned automatically from `monitoring/grafana/provisioning/` and loads the dashboard JSON from `monitoring/grafana/dashboards/`.

Python modules also use structured loggers through `src/utils.py`, writing operational logs under `logs/` during local execution.

## 6. CI/CD Automation

GitHub Actions workflow:

```bash
.github/workflows/ci.yml
```

The workflow performs:

- dependency installation with pip caching
- flake8 syntax and quality checks
- small Adult dataset download for CI-safe testing
- data processing and baseline training smoke test
- pytest regression suite
- Python compile checks
- Docker Compose config validation
- Docker image builds for API and MLflow

This satisfies CI/CD expectations for automated validation before merge/push.

## 7. Research Component

| Research requirement | Status | Evidence |
|---|---:|---|
| Clear research problem | Complete | `paper/ieee_drift_aware_mlops.tex`, README, this audit |
| Literature review with 8-10 papers | Complete | `paper/references.bib` includes dataset shift, concept drift, MLOps, XGBoost, Folktables, production ML references |
| Research questions/hypotheses | Complete | Paper introduction and methodology |
| Evaluation metrics | Complete | Accuracy, F1, precision, recall, drift severity, retrain count, retrain time, latency, degradation area, recovery gain |
| Experimental validation | Complete | `reports/results_acs.md`, `reports/diagnostics/acs_noise_ablation/`, `reports/diagnostics/acs_regime/` |
| Statistical/performance analysis | Complete | Drift tests, corrected p-values, effect sizes, severity scoring, ablation analysis, temporal regime diagnostics |

## 8. Final Experiment Artifacts

Main ACS result:

```bash
reports/results_acs.md
reports/figures/acs/
```

Noise ablation:

```bash
reports/diagnostics/acs_noise_ablation/
```

Temporal regime diagnostic:

```bash
reports/diagnostics/acs_regime/
```

Research paper:

```bash
paper/ieee_drift_aware_mlops.tex
paper/references.bib
paper/figures/
```

## 9. Remaining Limitations

The implementation is strong for a course project, but the following limitations should be acknowledged honestly:

- AWS deployment is not implemented because it is optional.
- The API loads local model artifacts rather than pulling from a formal MLflow Model Registry stage.
- Labeled feedback is assumed to be available per batch.
- Only XGBoost is evaluated as the learner.
- Docker validation in CI is stronger than local validation if Docker Desktop is not integrated with WSL.

## 10. Scoring Summary

The project satisfies the mandatory technical stack and goes beyond a basic implementation by adding:

- multi-year ACS data processing
- drift-aware retraining policy
- rolling-window plus anchor retraining
- persistence gating
- noise ablation
- temporal regime diagnostics
- recovery-oriented evaluation
- full IEEE research paper
- Dockerized API, MLflow, Prometheus, and Grafana stack
- GitHub Actions CI/CD validation
