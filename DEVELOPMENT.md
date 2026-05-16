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
- [ ] Project Structure Setup
- [ ] Data Ingestion from Open-Meteo
- [ ] DVC Initialization
- [ ] Basic Feature Engineering (Rolling windows, Cyclical encoding)

**Key Learnings:**
- Why DVC is better than Git for large datasets.
- Handling API rate limits and time-series gaps.

### Phase 2: The ML Pipeline
- [ ] Isolation Forest Implementation
- [ ] MLflow Tracking Setup
- [ ] DVC Pipeline Definition (`dvc.yaml`)
- [ ] Model Registry & Aliasing (`staging` vs `production`)

**Key Learnings:**
- Unsupervised anomaly detection vs Supervised learning.
- The importance of a reproducible DAG.

### Phase 3: Production Serving
- [ ] FastAPI implementation
- [ ] Model Hot-swapping logic
- [ ] Dockerization (Multi-stage builds)
- [ ] Local Orchestration (Docker Compose)

**Key Learnings:**
- Pydantic for data validation.
- Building lean Docker images.

### Phase 4: Observability & Drift
- [ ] Prometheus metrics export
- [ ] Grafana Dashboard configuration
- [ ] Evidently AI Drift Detection
- [ ] Automated Retraining Trigger

**Key Learnings:**
- Infrastructure metrics vs ML metrics.
- Concept Drift vs Data Drift.

### Phase 5: CI/CD & Cloud Deployment
- [ ] GitHub Actions for Linting & Testing
- [ ] AWS Infrastructure setup
- [ ] Automated Deployment to App Runner

---

## 📝 Daily Progress Notes

### Day 1: Project Kickoff
*Date: 2026-05-16*
- Defined project scope and architecture.
- Created `implementation_plan.md` and `DEVELOPMENT.md`.
- *Status: Initiated.*

---

## 🧠 Expert Corner: MLOps Concepts
*As we build, we will define advanced concepts here.*

1. **Model Registry**: A central store for managing model versions and stages.
2. **Feature Store (Concept)**: While we use a pipeline, in larger systems, this would be a Feature Store.
3. **Continuous Training (CT)**: The system automatically retrains when drift is detected.
