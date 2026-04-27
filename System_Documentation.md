# System Documentation: Drift-Aware MLOps Pipeline

This document provides a comprehensive technical overview of the implementation and architecture of the Drift-Aware MLOps Pipeline.

---

## 1. Project Objective
To build a production-grade MLOps system that maintains model integrity in non-stationary environments. The system uses **Two-Layer Drift Detection** and a **Policy-Based Retraining Engine** to minimize operational costs while maximizing predictive performance.

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
*   **Cost-Aware Triggers**: Instead of retraining on every alert, this engine evaluates three gates:
    1.  **Severity Gate**: Is the drift score >= 0.30?
    2.  **Data Gate**: Are there >= 1000 new labeled samples accumulated?
    3.  **Cooldown Gate**: Have at least 1-2 batches passed since the last update?

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
*   **Grafana**: Visual dashboard (located in `monitoring/grafana/`) showing the drift timeline and model version history.

---

### 📦 Layer 3: Tracking & Lifecycle

#### **E. Experiment Tracking (MLflow)**
*   **Artifacts**: Every training run logs the model (.pkl), the encoder pipeline, and the training data footprint.
*   **Metrics**: Accuracy, F1-Score, Precision, Recall, Train Time, and Drift Severity are tracked per batch.

#### **F. CI/CD & Automation**
*   **Makefile**: Unified entry point for `make train`, `make test`, and `make drift-check`.
*   **GitHub Actions**: Automated pipeline for linting (`flake8`) and regression testing on every push.

---

## 3. Data Strategy
*   **Base Dataset**: UCI Adult Income (Binary Classification).
*   **Simulated Time-Batches**: 5 sequential batches with controlled perturbations:
    *   **Shift**: Modifying numerical means (Age).
    *   **Resample**: Altering categorical distributions (Education).
    *   **Conditional**: Changing label relationships (Subgroup flipping).
    *   **Noise**: Simulating sensor failure or feature importance distortion.

---

## 4. Implementation Workflow
1.  **Baseline**: Model v1 is trained on "clean" data and registered in MLflow.
2.  **Production Traffic**: API is deployed in Docker.
3.  **Drift Event**: New batches arrive; `DriftDetector` computes the Batch Severity score.
4.  **Policy Decision**: If gates are cleared, `retrain.py` triggers a new MLflow run (Model v2).
5.  **Hot-Swap**: The API receives a `/reload-model` signal and begins serving v2 immediately.
6.  **Cycle Complete**: The system updates Grafana to reflect the new performance baseline.

---
*Created by Antigravity AI Implementation Agent - April 2026*
