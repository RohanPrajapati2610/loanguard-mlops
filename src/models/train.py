"""
Model Training with MLflow Tracking — Loan Fraud Detection
===========================================================
What this file does:
1. Loads processed training data
2. Splits into train / validation sets
3. Handles class imbalance (fraud is rare — we tell the model to pay attention)
4. Trains XGBoost model
5. Logs EVERYTHING to MLflow (params, metrics, model artifact)
6. Registers best model in MLflow Model Registry
7. Saves model locally as backup
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
)
from xgboost import XGBClassifier
import os
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH       = "data/processed/train.csv"
MLFLOW_URI      = "mlruns"
EXPERIMENT_NAME = "loan-fraud-detection"
MODEL_NAME      = "loan-fraud-model"
TARGET_COL      = "fraud_label"

# XGBoost hyperparameters
# Why these values?
# - n_estimators=300: enough trees to learn complex patterns
# - max_depth=6: deep enough to capture interactions, not too deep to overfit
# - learning_rate=0.05: slow learning = better generalization
# - scale_pos_weight: handles class imbalance (more legit loans than fraud)
PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.5,
    "random_state": 42,
    "eval_metric": "auc",
    "tree_method": "hist",
    "early_stopping_rounds": 30,
}


# ─────────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────────
def load_data(path: str):
    log.info(f"Loading training data from {path} ...")
    df = pd.read_csv(path)
    log.info(f"Loaded {len(df):,} rows")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    log.info(f"Features: {X.shape[1]}")
    log.info(f"Fraud rate: {y.mean() * 100:.2f}%")
    return X, y


# ─────────────────────────────────────────────
# STEP 2: Train / Validation Split
# ─────────────────────────────────────────────
def split_data(X, y):
    log.info("Splitting into train/validation (80/20) ...")
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # stratify=y ensures both splits have same fraud ratio


# ─────────────────────────────────────────────
# STEP 3: Calculate Class Weight
# ─────────────────────────────────────────────
# Why do we need this?
# Dataset has ~78% legit, ~22% fraud
# Without adjustment, model learns: "always say legit" → 78% accuracy but useless
# scale_pos_weight tells XGBoost: "pay 3x more attention to fraud cases"
def get_scale_pos_weight(y_train):
    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    weight = n_legit / n_fraud
    log.info(f"Class weight (scale_pos_weight): {weight:.2f}")
    return weight


# ─────────────────────────────────────────────
# STEP 4: Train Model
# ─────────────────────────────────────────────
def train_model(X_train, y_train, X_val, y_val, scale_pos_weight):
    log.info("Training XGBoost model ...")
    params = PARAMS.copy()
    params["scale_pos_weight"] = round(scale_pos_weight, 2)

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model, params


# ─────────────────────────────────────────────
# STEP 5: Evaluate Model
# ─────────────────────────────────────────────
def find_best_threshold(y_val, y_proba):
    """Find threshold that maximizes F1 score on validation set."""
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = f1_scores.argmax()
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    log.info(f"Best threshold: {best_threshold:.4f} (F1={f1_scores[best_idx]:.4f})")
    return round(float(best_threshold), 4)


def evaluate_model(model, X_val, y_val):
    log.info("Evaluating model ...")
    y_proba = model.predict_proba(X_val)[:, 1]

    threshold = find_best_threshold(y_val, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "roc_auc":   round(roc_auc_score(y_val, y_proba), 4),
        "f1_score":  round(f1_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred), 4),
        "recall":    round(recall_score(y_val, y_pred), 4),
        "threshold": threshold,
    }

    log.info(f"ROC-AUC:   {metrics['roc_auc']}")
    log.info(f"F1 Score:  {metrics['f1_score']}")
    log.info(f"Precision: {metrics['precision']}")
    log.info(f"Recall:    {metrics['recall']}")
    log.info(f"\nClassification Report:\n{classification_report(y_val, y_pred)}")

    return metrics, threshold


# ─────────────────────────────────────────────
# STEP 6: Log to MLflow
# ─────────────────────────────────────────────
# MLflow tracks: params used, metrics achieved, the model itself
# So you can always answer: "which model is in prod? how was it trained?"
def log_to_mlflow(model, params, metrics, X_train, threshold):
    log.info("Logging to MLflow ...")

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="xgboost-baseline") as run:
        # Log hyperparameters
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics(metrics)

        # Log model + feature names (important for later)
        signature = mlflow.models.infer_signature(
            X_train, model.predict(X_train)
        )
        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            signature=signature,
            registered_model_name=MODEL_NAME,
        )

        # Save feature columns so API knows expected input
        feature_cols = list(X_train.columns)
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/feature_columns.json", "w") as f:
            json.dump(feature_cols, f)
        mlflow.log_artifact("data/processed/feature_columns.json")

        # Save optimal threshold so API uses it at inference time
        with open("models/threshold.json", "w") as f:
            json.dump({"threshold": threshold}, f)
        mlflow.log_artifact("models/threshold.json")

        run_id = run.info.run_id
        log.info(f"MLflow Run ID: {run_id}")
        log.info(f"Experiment: {EXPERIMENT_NAME}")

    return run_id


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs("mlruns", exist_ok=True)

    X, y                              = load_data(DATA_PATH)
    X_train, X_val, y_train, y_val   = split_data(X, y)
    scale_pos_weight                  = get_scale_pos_weight(y_train)
    model, params                     = train_model(X_train, y_train, X_val, y_val, scale_pos_weight)
    metrics, threshold                = evaluate_model(model, X_val, y_val)
    run_id                            = log_to_mlflow(model, params, metrics, X_train, threshold)

    log.info("=" * 50)
    log.info("Training complete!")
    log.info(f"Best ROC-AUC: {metrics['roc_auc']}")
    log.info(f"MLflow run ID: {run_id}")
    log.info("Run `venv/Scripts/mlflow ui` to view results in browser")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
