---
title: LoanGuard API
emoji: 🛡️
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# LoanGuard MLOps — Loan Fraud Detection System

A production-grade MLOps pipeline for detecting fraudulent loan applications. Built end-to-end with automated drift detection, self-healing retraining, and real-time monitoring.

---

## Live Demo

| Interface | Link | Description |
|---|---|---|
| Prediction UI | `Coming soon` | Submit a loan application, get instant fraud risk |
| MLOps Dashboard | `Coming soon` | Real-time model monitoring and drift detection |

---

## Architecture

```
Lending Club Dataset (2.26M rows)
        ↓
Data Versioning (DVC)
        ↓
Airflow Training Pipeline
  ├── check_drift       → Is retraining needed?
  ├── load_data         → Load versioned dataset
  ├── train_model       → XGBoost + MLflow tracking
  └── evaluate_and_promote → A/B comparison gate
        ↓
MLflow Model Registry (versioned, auditable)
        ↓
FastAPI REST API (Dockerized)
        ↓
Simulation Script (fake bank traffic)
        ↓
Evidently AI (feature-level drift detection)
        ↓
Drift Detected? → Airflow triggers retraining
        ↓
Streamlit Dashboard (real-time monitoring)
```

---

## Stack

| Component | Tool |
|---|---|
| Model | XGBoost |
| Experiment Tracking | MLflow |
| Model Registry | MLflow Model Registry |
| API | FastAPI + Uvicorn |
| Containerization | Docker |
| Pipeline Orchestration | Apache Airflow |
| Drift Detection | Evidently AI |
| Data Versioning | DVC |
| Monitoring Dashboard | Streamlit + Plotly |
| Frontend | HTML / CSS / JavaScript |

---

## What Makes This Different

Most ML projects stop at training a model. This project goes further:

- **Data versioning** — every model is linked to the exact data version that trained it
- **Automated drift detection** — Evidently AI monitors feature distributions continuously
- **Self-healing** — when drift is detected, Airflow triggers retraining automatically
- **A/B model promotion** — new model only replaces old if it actually performs better
- **Feature-level drift reports** — shows exactly which features drifted and by how much
- **Full predictions log** — every API call timestamped and stored

---

## Project Structure

```
ML_project/
├── src/
│   ├── api/           → FastAPI application
│   ├── data/          → Data preprocessing
│   ├── models/        → Model training
│   ├── monitoring/    → Drift detection + dashboard
│   └── simulate_traffic.py → Fake bank traffic simulator
├── airflow/
│   └── dags/          → Retraining DAG
├── frontend/          → Prediction web UI
├── data/
│   ├── raw/           → Raw Lending Club data (DVC tracked)
│   └── processed/     → Cleaned, feature-engineered data
├── models/            → Saved model artifacts (DVC tracked)
├── Dockerfile         → API container
├── docker-compose.yml → Airflow + services
└── requirements.txt
```

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo>
cd ML_project
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# 2. Train model
python src/models/train.py

# 3. Start API
uvicorn src.api.app:app --reload --port 8000

# 4. Start dashboard
streamlit run src/monitoring/dashboard.py

# 5. Run simulation
python src/simulate_traffic.py

# 6. Start Airflow (Docker required)
docker compose up -d
```

---

## Model Performance

| Metric | Value |
|---|---|
| ROC-AUC | 0.7257 |
| F1 Score | 0.4754 |
| Recall | 0.6826 |
| Precision | 0.3647 |

High recall is intentional — in fraud detection, missing a fraud (false negative) is more costly than a false alarm.

---

## Dataset

**Lending Club Loan Dataset** — 2.26M loan applications (2007–2015)
- Source: Kaggle
- Features: 145 columns including loan amount, interest rate, DTI, income, credit history
- Target: Loan status (Fully Paid / Charged Off / Default)