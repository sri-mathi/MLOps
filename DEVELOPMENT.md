# 🚀 AQI Real-Time Anomaly Detection — MLOps Development Log

Welcome to your MLOps journey! This document tracks the technical progress, key learnings, and architectural decisions made during the development of the AQI Anomaly Detection system.

---

## 🎯 Project Vision
To build a "living" ML system that doesn't just predict, but manages its own lifecycle: ingesting data, detecting anomalies, monitoring drift, and self-healing through retraining.

---

## 🛠 Tech Stack Overview
- **Data**: Open-Meteo Air Quality API (Chennai, Bangalore, Delhi)
- **ML**: Isolation Forest (Multivariate Time Series)
- **Reproducibility**: DVC (Data Version Control)
- **Tracking**: MLflow (Experiments & Model Registry)
- **Serving**: FastAPI (Dockerized)
- **Monitoring**: Prometheus, Grafana, Evidently AI
- **CI/CD**: GitHub Actions
- **Cloud**: AWS (App Runner, S3, ECR)

---

## 📅 Development Timeline

### Phase 1: Foundation & Data Engineering
- [x] Project Structure Setup
- [x] Data Ingestion from Open-Meteo
- [x] DVC Initialization
- [x] Basic Feature Engineering (Rolling windows, Cyclical encoding)

**Key Learnings:**
- Why DVC is better than Git for large datasets.
- Handling API rate limits and time-series gaps.

### Phase 2: The ML Pipeline
- [x] Isolation Forest Implementation
- [x] MLflow Tracking Setup
- [x] DVC Pipeline Definition (`dvc.yaml`)
- [x] Model Registry & Aliasing (`staging` vs `production`)

**Key Learnings:**
- Unsupervised anomaly detection vs Supervised learning.
- The importance of a reproducible DAG.

### Phase 3: Production Serving
- [x] FastAPI implementation
- [x] Model Hot-swapping logic
- [x] Dockerization (Multi-stage builds)
- [x] Local Orchestration (Docker Compose)

**Key Learnings:**
- Pydantic for robust incoming sensor payload validation.
- Building high-performance, multi-stage Docker images to keep final image size under 200MB.
- Orchestrating multi-service environments with Docker Compose.

### Phase 4: Observability & Drift
- [x] Prometheus metrics export
- [x] Grafana Dashboard configuration
- [x] Evidently AI Drift Detection
- [x] Automated Retraining Trigger

**Key Learnings:**
- Separating system performance telemetry (latency, memory) from ML metrics (anomaly scores).
- Detecting statistical covariate shifts in sensor inputs using evidently AI's Kolmogorov-Smirnov statistical tests.

### Phase 5: CI/CD & Cloud Deployment
- [x] GitHub Actions for Linting & Testing
- [x] AWS Infrastructure setup (Architecture Blueprint)
- [x] Automated Verification & Docker build CI tests

**Key Learnings:**
- Running automated CI/CD checks (Ruff, Pytest) on GitHub Actions.
- Validating Docker image builds before registering them in ECR.

---

## 📝 Daily Progress Notes

### Day 1: Project Kickoff & Foundation
*Date: 2026-05-16*
- Defined project scope, finalized ingestion pipelines, initialized DVC repository.

### Day 2: Training & Serving Pipelines
*Date: 2026-05-17*
- Registered `AQI_Anomaly_Detector` model in SQLite DB with creator custom attributes.
- Built FastAPI serving layer with stateful, rolling sliding cache to compute features on-the-fly.

### Day 3: Observability, Metrics & CI/CD
*Date: 2026-05-18*
- Integrated Prometheus custom counters/histograms and configured scraped targets.
- Created live, persistent Grafana analytics dashboard tracking throughput and anomaly alerts.
- Configured Evidently AI offline drift detection reports.
- Set up unit testing with `pytest` and configured automated GitHub Actions workflows (`pipeline.yml`).

---

## 🧠 Expert Corner: MLOps Concepts
*As we build, we will define advanced concepts here.*

1. **Model Registry**: A central store for managing model versions, tracking metadata, and model state transitions.
2. **Data Drift**: Shifts in the statistical distribution of independent input features (e.g. seasonal changes boosting PM2.5 levels), checked via offline metrics (Evidently AI).
3. **Continuous Training (CT)**: Automated workflow executing DVC pipeline retraining triggered when drift thresholds are breached.

