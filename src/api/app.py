"""
FastAPI Prediction API — Loan Fraud Detection
=============================================
What this file does:
1. Loads the trained model from MLflow Model Registry
2. Exposes a /predict endpoint — send loan data, get fraud prediction back
3. Exposes a /health endpoint — check if API is alive
4. Exposes a /model-info endpoint — see which model version is running
5. Logs every prediction for monitoring later

How it works:
  Client sends loan data (JSON)
       ↓
  FastAPI receives it
       ↓
  We convert it to the right format
       ↓
  Model predicts: fraud or not fraud
       ↓
  We send back the prediction + confidence score
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

import pandas as pd
import uvicorn
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_NAME      = "loan-fraud-model"
MODEL_VERSION   = "1"
MODEL_PATH      = "models/loan_fraud_model.json"
FEATURE_PATH    = "data/processed/feature_columns.json"
PREDICTIONS_LOG = "logs/predictions.csv"

os.makedirs("logs", exist_ok=True)

# ─────────────────────────────────────────────
# LOAD MODEL AT STARTUP
# ─────────────────────────────────────────────
# Why load at startup and not per-request?
# Loading a model takes ~1-2 seconds. If we load per request,
# every user waits 2 seconds. Load once → serve instantly.
log.info(f"Loading model from {MODEL_PATH} ...")
model = XGBClassifier()
model.load_model(MODEL_PATH)
log.info("Model loaded successfully.")

# Load expected feature columns
with open(FEATURE_PATH) as f:
    FEATURE_COLUMNS = json.load(f)
log.info(f"Expecting {len(FEATURE_COLUMNS)} features")

# Load optimal prediction threshold
THRESHOLD_PATH = "models/threshold.json"
THRESHOLD = 0.5  # fallback default
if os.path.exists(THRESHOLD_PATH):
    with open(THRESHOLD_PATH) as f:
        THRESHOLD = json.load(f).get("threshold", 0.5)
log.info(f"Using prediction threshold: {THRESHOLD}")


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Loan Fraud Detection API",
    description="Production ML API for detecting fraudulent loan applications",
    version="1.0.0",
)

# Allow requests from any origin (needed for dashboards / frontends)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# REQUEST / RESPONSE SCHEMAS
# ─────────────────────────────────────────────
# Pydantic validates incoming data automatically
# If someone sends wrong types → FastAPI rejects with clear error
class LoanApplication(BaseModel):
    loan_amnt: float        = Field(..., description="Loan amount requested")
    funded_amnt: float      = Field(..., description="Amount funded")
    term: float             = Field(..., description="Loan term in months (36 or 60)")
    int_rate: float         = Field(..., description="Interest rate (%)")
    installment: float      = Field(..., description="Monthly installment amount")
    annual_inc: float       = Field(..., description="Annual income")
    dti: float              = Field(..., description="Debt-to-income ratio")
    delinq_2yrs: float      = Field(0, description="Delinquencies in past 2 years")
    inq_last_6mths: float   = Field(0, description="Credit inquiries last 6 months")
    open_acc: float         = Field(..., description="Number of open credit accounts")
    pub_rec: float          = Field(0, description="Number of public records")
    revol_bal: float        = Field(..., description="Revolving balance")
    revol_util: float       = Field(..., description="Revolving line utilization (%)")
    total_acc: float        = Field(..., description="Total credit accounts")
    mort_acc: float         = Field(0, description="Number of mortgage accounts")
    pub_rec_bankruptcies: float = Field(0, description="Number of bankruptcies")

    class Config:
        json_schema_extra = {
            "example": {
                "loan_amnt": 15000,
                "funded_amnt": 15000,
                "term": 36,
                "int_rate": 12.5,
                "installment": 502.5,
                "annual_inc": 65000,
                "dti": 18.5,
                "delinq_2yrs": 0,
                "inq_last_6mths": 1,
                "open_acc": 8,
                "pub_rec": 0,
                "revol_bal": 12000,
                "revol_util": 45.2,
                "total_acc": 15,
                "mort_acc": 1,
                "pub_rec_bankruptcies": 0,
            }
        }


class PredictionResponse(BaseModel):
    fraud_prediction: int       # 0 = legit, 1 = fraud
    fraud_probability: float    # confidence score (0-1)
    risk_level: str             # LOW / MEDIUM / HIGH
    model_version: str
    timestamp: str


# ─────────────────────────────────────────────
# HELPER: Align Input to Model Features
# ─────────────────────────────────────────────
# Our model was trained on 127 features (after one-hot encoding)
# But the API receives only ~16 raw features
# We need to fill in all the missing one-hot columns with 0
def align_features(input_dict: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_dict])

    # Add all missing columns as 0 (absent category = 0 in one-hot encoding)
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0

    # Keep only the columns the model expects, in the right order
    df = df[FEATURE_COLUMNS]
    return df


# ─────────────────────────────────────────────
# HELPER: Log Prediction
# ─────────────────────────────────────────────
def log_prediction(input_data: dict, prediction: int, probability: float):
    record = {
        "timestamp": datetime.now().isoformat(),
        "prediction": prediction,
        "probability": probability,
        **input_data,
    }
    record_df = pd.DataFrame([record])

    if os.path.exists(PREDICTIONS_LOG):
        record_df.to_csv(PREDICTIONS_LOG, mode="a", header=False, index=False)
    else:
        record_df.to_csv(PREDICTIONS_LOG, index=False)


# ─────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/predictions")
def get_predictions(limit: int = 100):
    """Return recent predictions log as JSON — used by the Streamlit dashboard"""
    if os.path.exists(PREDICTIONS_LOG):
        df = pd.read_csv(PREDICTIONS_LOG).tail(limit)
        return df.to_dict(orient="records")
    return []


@app.get("/health")
def health_check():
    """Check if API is alive and model is loaded"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "version": MODEL_VERSION,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info")
def model_info():
    """Return info about the currently loaded model"""
    return {
        "model_name": MODEL_NAME,
        "model_version": MODEL_STAGE,
        "n_features": len(FEATURE_COLUMNS),
        "mlflow_uri": MLFLOW_URI,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    """
    Predict whether a loan application is fraudulent.

    - fraud_prediction = 1 → HIGH RISK, likely fraud
    - fraud_prediction = 0 → LOW RISK, likely legit
    - fraud_probability → confidence score (closer to 1 = more likely fraud)
    """
    try:
        input_dict = application.model_dump()
        df = align_features(input_dict)

        probability = float(model.predict_proba(df)[0][1])
        prediction  = int(probability >= THRESHOLD)

        # Risk level bucketing relative to threshold
        if probability < THRESHOLD * 0.6:
            risk_level = "LOW"
        elif probability < THRESHOLD:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        log_prediction(input_dict, prediction, probability)

        log.info(f"Prediction: {prediction} | Probability: {probability:.4f} | Risk: {risk_level}")

        return PredictionResponse(
            fraud_prediction=prediction,
            fraud_probability=round(probability, 4),
            risk_level=risk_level,
            model_version=MODEL_VERSION,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
