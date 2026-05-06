# System Documentation: Drift-Aware MLOps Pipeline

This document provides a comprehensive technical overview of the implementation and architecture of the Drift-Aware MLOps Pipeline.

---

## 1. Project Objective
To build a research-grade MLOps system that maintains model integrity in non-stationary environments and evaluates when adaptive retraining is actually useful. The system uses drift detection, policy-gated retraining, experiment tracking, containerized deployment, monitoring, and CI/CD automation.

---

## 2. System Architecture

### 🛡️ Layer 1: Core Research Logic
The brain of the system is contained in two modules that operate independently of the model type.

#### **A. Drift Detection Engine (`src/drift_detection.py`)**
*   **Statistical Tests**: Uses Kolmogorov-Smirnov (KS) for numerical features and Chi-Square for categorical features.
*   **Layer 1 (Feature-Level)**: Individual p-values are calculated for every feature.
*   **Correction Logic**: Implements **Bonferroni** and **Benjamini-Hochberg** corrections to prevent false positives in high-dimensional feature spaces.
*   **Layer 2 (Global Severity)**: Aggregates individual feature shifts into a single **Severity Score [0 to 1]** using effect sizes (Cramér's V and KS-Statistic).

#### **B. Policy Engine (`src/policy_engine.py`)**
*   **Cost-Aware Triggers**: Instead of retraining on every alert, this engine evaluates:
    1.  **Severity Gate**: Is feature drift large enough?
    2.  **Concept Gate**: Has labeled performance degraded?
    3.  **Data Gate**: Are enough new labeled samples accumulated?
    4.  **Cooldown Gate**: Has enough time passed since the last retrain?
    5.  **Persistence Gate**: Did the signal persist across consecutive batches?
*   **Retraining Data Policy**: Uses a recent rolling window plus a stratified historical anchor.

---

### 🚀 Layer 2: Production Infrastructure

#### **C. Inference Service (`api/app.py`)**
*   **Framework**: FastAPI.
*   **Endpoints**:
    *   `POST /predict`: Real-time inference with Pydantic validation.
    *   `POST /reload-model`: **Zero-Downtime Hot-Swapping**. Reloads artifacts in memory without restarting the container.
    *   `GET /metrics`: Native Prometheus exporter for real-time monitoring.

#### **D. Observability & Monitoring**
*   **Prometheus**: Backend time-series database that scrapes API health and drift metrics.
*   **Grafana**: Provisioned dashboard in `monitoring/grafana/` showing prediction volume, latency, model version, drift severity, and retraining events.

---

### 📦 Layer 3: Tracking & Lifecycle

#### **E. Experiment Tracking (MLflow)**
*   **Artifacts**: Every training run logs the model (.pkl), the encoder pipeline, and the training data footprint.
*   **Metrics**: Accuracy, F1-Score, Precision, Recall, Train Time, and Drift Severity are tracked per batch.
*   **Deployment**: MLflow is available as a Docker Compose service and through `make mlflow`.

#### **F. CI/CD & Automation**
*   **Makefile**: Unified entry point for testing, ACS training, drift generation, experiments, ablation, MLflow, and Docker stack commands.
*   **GitHub Actions**: Automated pipeline for linting, regression testing, smoke training, compile checks, Docker Compose validation, and Docker image builds.

---

## 3. Data Strategy
*   **Development Dataset**: UCI Adult Income for lightweight CI and local tests.
*   **Final Research Dataset**: Folktables ACS Income across 2016, 2017, and 2018.
*   **Temporal Split**: 2016 is used as baseline training data; 2017-2018 are used as shifted evaluation data.
*   **Main Drift Design**: Temporal sourcing, covariate shifts, categorical redistribution, and feature-noise shifts.
*   **Stress-Test Drift Design**: Subgroup label flips are isolated in `configs/drift_config_acs_label_flip_stress.yaml` and used only for noise-ablation diagnostics.

---

## 4. Implementation Workflow
1.  **Baseline**: Model v1 is trained on "clean" data and registered in MLflow.
2.  **Production Traffic**: API, MLflow, Prometheus, and Grafana are deployed in Docker Compose.
3.  **Drift Event**: New batches arrive; `DriftDetector` computes feature-level tests and global severity.
4.  **Policy Decision**: If gates are cleared, `retrain.py` triggers a new MLflow run (Model v2).
5.  **Hot-Swap**: The API receives a `/reload-model` signal and begins serving v2 immediately.
6.  **Cycle Complete**: The system updates Grafana to reflect the new performance baseline.

## 5. Final Research Outputs

*   `reports/results_acs.md`: final ACS learnable-drift experiment.
*   `reports/diagnostics/acs_noise_ablation/`: clean-vs-label-noise ablation.
*   `reports/diagnostics/acs_regime/`: temporal regime and feature-importance diagnostics.
*   `paper/ieee_drift_aware_mlops.tex`: final IEEE-format research paper.

---
