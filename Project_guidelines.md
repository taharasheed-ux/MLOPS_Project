# Drift-Aware MLOps Pipeline Project Spec

## 1) Objective

Build an end-to-end MLOps system for a tabular binary classification problem where the central research problem is **feature-aware drift detection plus policy-based automated retraining**.

The system must support:

* Experiment tracking
* Containerized deployment
* Drift monitoring
* Retraining automation
* Metrics collection and visualization
* Comparative experiments against baselines

This is a **research-driven engineering project**. The core contribution is not the model itself; it is the drift-aware operational pipeline and the evaluation of retraining policies.

---

## 2) Final Research Framing

### Research problem

Production ML models degrade when the input distribution changes over time. The project studies whether a drift-aware retraining policy can preserve predictive performance more efficiently than naive or static retraining.

### Research questions

1. Can feature-level drift detection identify distribution shifts earlier or more precisely than a single global drift signal?
2. Does policy-based retraining outperform static deployment and naive immediate retraining?
3. What is the trade-off between model recovery, retraining frequency, and system overhead?

### Core contribution

A modular MLOps pipeline that:

* detects drift at both feature and batch level,
* computes a drift severity score,
* triggers retraining only when policy conditions are satisfied,
* logs all experiments and deployed versions,
* exposes operational metrics for monitoring.

---

## 3) Data Strategy

### Final dataset choice

Use the **UCI Adult Income dataset** as the primary real-world tabular dataset.

### Why this dataset

* Contains both numerical and categorical features.
* Is suitable for classification and monitoring over time.
* Allows realistic drift simulation by perturbing feature distributions and label relationships.

### Data handling approach

* Clean and preprocess the dataset once.
* Split into training and sequential evaluation batches.
* Simulate drift across batches instead of relying on a synthetic-only setup.

### Drift simulation design

Create time-ordered batches with increasing non-stationarity:

* Batch 1: baseline training data (no perturbation)
* Batch 2: mild covariate shift (e.g., shift `age` by +5 years, `hours-per-week` by +3)
* Batch 3: stronger covariate shift (e.g., shift `age` by +10, resample `education` distribution)
* Batch 4: conditional shift (flip 15% of labels for specific subgroups)
* Batch 5: mixed drift scenario (covariate + conditional + feature importance changes)

All perturbation parameters must be defined in `configs/drift_config.yaml` for reproducibility.

### Drift types to simulate

* Covariate drift: feature distributions change
* Conditional drift: label relationship changes
* Feature importance drift: predictive power of key features changes

### Reference distribution management

After each successful retrain, the new training data becomes the reference distribution for future drift comparisons. This mirrors realistic production behavior.

### Dataset sourcing note

The agent should download the dataset manually from a public source such as the **UCI Machine Learning Repository** or a mirrored Kaggle version if necessary.

---

## 4) Model Choice

### Final model

Use **XGBoost Classifier** as the main production candidate.

### Why

* Strong performance on tabular data
* Handles mixed feature types well after encoding
* Gives a meaningful baseline for retraining comparisons
* Is practical for deployment and experimentation

### Encoding strategy

* Categorical features: `OrdinalEncoder` (XGBoost-native friendly)
* Binary features: `LabelEncoder` fallback
* Avoid one-hot encoding to keep dimensionality manageable for drift detection

### Baseline comparison models

* Static trained model with no retraining
* Immediate retraining model
* Policy-based retraining model

---

## 5) Drift Detection Design

### Drift detection should be two-layered

#### Layer 1: Feature-level drift detection

For each feature, compare the current batch against the reference distribution.

Suggested methods:

* Numerical features: Kolmogorov-Smirnov test
* Categorical features: Chi-square test

Statistical correction:

* Apply **Bonferroni correction** (or Benjamini-Hochberg FDR control) when testing multiple features to control false positive rate.
* Report both raw p-values and corrected p-values.
* Report effect sizes (KS statistic / Cramér's V) alongside p-values.

Output:

* Which features drifted
* P-values (raw and corrected) or test scores
* Drift magnitude (effect size) by feature

#### Layer 2: Global drift severity score

Aggregate feature drift into a single severity signal.

Suggested logic:

* count of drifted features
* weighted drift magnitude
* optional normalization by batch size

Decision rule:

* if severity exceeds threshold, raise drift event

### Implementation goal

The drift detector should be readable, modular, and configurable through thresholds.

---

## 6) Retraining Policy

### Core policy idea

Retraining must not happen on every small signal.

### Conditions for retraining

Retrain only when all of the following hold:

1. Drift severity exceeds threshold
2. Enough new labeled data has accumulated
3. Cooldown period has passed since the last retrain

### Labeled data assumption

For this project, labels are assumed to be available immediately (simulated batch scenario). This simplification should be documented in the final report as a known limitation.

### Why this matters

This creates a researchable policy trade-off:

* too sensitive → too many retrains
* too conservative → slow recovery

### Policy variants to evaluate

* No retraining
* Immediate retraining on any detected drift
* Policy-based retraining with cooldown and data threshold

The policy-based version is the proposed contribution.

---

## 7) Experimental Plan

### Required comparisons

The agent must implement experiments that compare:

#### Experiment A: Static vs adaptive system

* Static model deployed once
* Drift-aware retraining system

#### Experiment B: Immediate retraining vs policy-based retraining

* Immediate retraining after drift
* Retraining only when policy conditions are satisfied

#### Experiment C: Threshold sensitivity

Evaluate multiple drift thresholds to show:

* detection sensitivity
* retraining frequency
* performance recovery

#### Experiment D: Cost vs performance

Report trade-offs between:

* accuracy / F1
* retraining frequency
* inference latency
* retraining time

### Required outputs

* batch-wise metric curves
* retraining event timeline
* drift score timeline
* model version history

### Output format

* Plots: matplotlib/seaborn charts saved as PNGs in `reports/figures/`
* Summary: `reports/results.md` with tables comparing all experiment variants
* MLflow: all individual run metrics logged for drill-down

---

## 8) Metrics

### Model metrics

* Accuracy
* F1-score
* Precision
* Recall
* Batch-wise performance degradation / recovery

### Drift metrics

* Feature drift flags
* Drift severity score
* Number of drift events
* Drift detection timing

### System metrics

* Inference latency
* Retraining duration
* Number of retraining triggers
* Deployment frequency

---

## 9) Mandatory Tooling and Role of Each Tool

### MLflow

Use for:

* experiment tracking
* parameter logging
* metric logging
* model artifact logging
* model version comparison

### Docker

Use for:

* packaging inference service
* packaging training/retraining jobs
* making the system reproducible

### FastAPI inference service

* Support both single and batch prediction endpoints
* Include `/reload-model` endpoint for in-memory model hot-swap after retraining (avoids container restarts)
* Expose Prometheus metrics via `/metrics` endpoint

### Prometheus

Use for:

* runtime metrics collection
* request count
* latency
* retraining count
* drift score exposure (via Prometheus `Gauge` updated by batch evaluation loop)

### Grafana

Use for:

* dashboards
* metric visualization
* drift timeline
* latency monitoring
* retraining event visualization

### CI/CD

Use for:

* automated pipeline execution (GitHub Actions for lint/test on push)
* local `Makefile` / shell scripts for retraining trigger (practical for local WSL setup)
* container rebuild and redeployment

GitHub Actions handles lint/test automation. Retraining triggers run locally since model training on free-tier CI is impractical. Document the CI/CD as both "what would happen in production" and "what we actually automate locally."

---

## 10) Architecture

### Core flow

1. Preprocess and train initial model
2. Log training run in MLflow
3. Deploy inference service in Docker
4. Receive sequential data batches
5. Predict and monitor metrics
6. Run drift detection on each batch
7. Compute drift severity
8. Evaluate retraining policy
9. Trigger retraining pipeline if needed
10. Register new model version in MLflow
11. Redeploy updated container
12. Expose metrics to Prometheus and Grafana

### Architecture principles

* Keep training, inference, drift detection, and policy logic modular
* Use config files for thresholds and policy parameters
* Store artifacts and logs consistently
* Make every stage reproducible

---

## 11) Repository Structure

```text
mlops-drift-project/
├── data/
├── notebooks/
├── src/
│   ├── data_processing.py
│   ├── feature_encoding.py
│   ├── drift_detection.py
│   ├── policy_engine.py
│   ├── train.py
│   ├── retrain.py
│   ├── evaluate.py
│   └── utils.py
├── api/
│   └── app.py
├── configs/
│   ├── settings.yaml
│   └── thresholds.yaml
├── mlflow/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── ci_cd/
│   └── github_actions.yml
├── monitoring/
│   ├── prometheus.yml
│   └── grafana/
├── experiments/
├── logs/
├── reports/
└── README.md
```

---

## 12) Environment and Setup Guidance

### Recommended OS

Use **Ubuntu** if possible.

Reasons:

* Docker behaves more naturally
* Prometheus/Grafana setup is easier
* CI/CD tooling and server-like workflows are simpler

Windows is still acceptable for development, but Ubuntu should be preferred for the deployment and monitoring parts.

### Required setup tasks

The agent should prepare or guide through:

* Python virtual environment
* package installation
* MLflow tracking setup
* Docker build/run validation
* Prometheus configuration
* Grafana dashboard connection
* GitHub Actions workflow configuration
* local API testing

### Important note for the agent

Do not introduce extra orchestration frameworks unless they are absolutely necessary. Keep the stack aligned with the assignment requirements.

---

## 13) CI/CD Expectations

The CI/CD workflow should support:

* lint/test on push
* training or retraining job execution
* model artifact registration
* container rebuild
* redeployment

The exact trigger can be:

* manual at first
* automated later when drift is detected

The retraining trigger should be part of the research pipeline, not just a generic DevOps demo.

---

## 14) Monitoring Expectations

### Prometheus should expose

* request count
* inference latency
* drift score
* retraining count
* model version identifier if convenient

### Grafana dashboards should show

* drift score over time
* batch-wise model performance
* latency over time
* retraining events
* current deployment status

---

## 15) Development Milestones

### Milestone 1

Data loading, preprocessing, and drift simulation

### Milestone 2

Baseline model training and MLflow logging

### Milestone 3

Feature-level drift detection

### Milestone 4

Policy engine for retraining decisions

### Milestone 5

Inference API and Dockerization

### Milestone 6

CI/CD automation

### Milestone 7

Prometheus and Grafana monitoring

### Milestone 8

Experimental comparison and report outputs

---

## 16) Desired Final Demonstration

The final system should visibly demonstrate:

* initial training and logging
* sequential batch processing
* drift spike detection
* policy-triggered retraining
* new model version registration
* redeployment
* updated monitoring dashboard

---

## 17) Dataset Download Hint

Search for:

* **UCI Adult Income dataset**
* **Adult census income dataset**

Likely sources:

* UCI Machine Learning Repository
* Kaggle mirror if needed

---

## 18) Non-Negotiable Constraints for the Agent

* Keep the pipeline modular
* Keep the research logic explicit
* Keep experiment comparisons visible
* Do not build a toy-only system
* Do not rely on a single static evaluation split
* Simulate drift across time batches
* Preserve reproducibility through logging and configuration
