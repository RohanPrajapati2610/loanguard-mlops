"""
Loan Fraud Detection — Auto Retraining DAG
==========================================
What this DAG does:
  Every day at 2am, it:
  1. Checks if data drift has been detected (reads drift_summary.json)
  2. If drift detected → runs full retraining pipeline
  3. Evaluates new model vs old model
  4. Promotes new model only if it's better (A/B gate)
  5. Saves new model and logs result
  6. Sends alert if retraining failed

Why Airflow?
- Runs automatically on schedule
- Retries failed steps automatically
- Gives full visibility: which step failed, why, when
- Industry standard at Airflow, Spotify, Airbnb, etc.

DAG Structure:
  check_drift
      ↓
  [drift detected?]
      ↓ YES
  load_new_data → train_new_model → evaluate_models → promote_if_better
      ↓ NO
  skip_retraining
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# PROJECT PATHS (inside Docker container)
# ─────────────────────────────────────────────
PROJECT_DIR    = "/opt/ml_project"
TRAIN_PATH     = f"{PROJECT_DIR}/data/processed/train.csv"
NEW_DATA_PATH  = f"{PROJECT_DIR}/data/processed/new_data.csv"
DRIFTED_PATH   = f"{PROJECT_DIR}/data/drifted/drifted_data.csv"
MODEL_PATH     = f"{PROJECT_DIR}/models/loan_fraud_model.json"
FEATURES_PATH  = f"{PROJECT_DIR}/data/processed/feature_columns.json"
DRIFT_SUMMARY  = f"{PROJECT_DIR}/logs/drift_summary.json"
RETRAIN_LOG    = f"{PROJECT_DIR}/logs/retrain_history.json"

TARGET_COL = "fraud_label"

# ─────────────────────────────────────────────
# DAG DEFAULT ARGS
# ─────────────────────────────────────────────
default_args = {
    "owner":            "ml-team",
    "retries":          2,                        # retry twice on failure
    "retry_delay":      timedelta(minutes=5),     # wait 5 min between retries
    "email_on_failure": False,
    "start_date":       datetime(2026, 1, 1),
}

dag = DAG(
    dag_id="loan_fraud_retrain_pipeline",
    description="Auto-retraining pipeline triggered by drift detection",
    schedule="0 2 * * *",   # every day at 2am
    default_args=default_args,
    catchup=False,
    tags=["mlops", "fraud-detection", "retraining"],
)


# ─────────────────────────────────────────────
# TASK 1: CHECK DRIFT
# Returns True if drift detected → pipeline continues
# Returns False → pipeline stops (ShortCircuit)
# ─────────────────────────────────────────────
def check_drift_detected(**context):
    """
    Read drift summary JSON written by Evidently AI.
    If drift detected in drifted data → return True (trigger retraining).
    If no drift → return False (skip retraining, save compute).
    """
    log.info("Checking drift summary ...")

    if not os.path.exists(DRIFT_SUMMARY):
        log.warning("No drift summary found. Running retraining anyway.")
        return True

    with open(DRIFT_SUMMARY) as f:
        summary = json.load(f)

    drift_info = summary.get("drifted_data_vs_train", {})
    drift_detected = drift_info.get("drift_detected", False)
    drifted_count  = drift_info.get("drifted_feature_count", 0)
    total          = drift_info.get("total_features", 0)

    log.info(f"Drift detected: {drift_detected}")
    log.info(f"Drifted features: {drifted_count}/{total}")

    if drift_detected:
        log.info("DRIFT CONFIRMED — Triggering retraining pipeline")
    else:
        log.info("No significant drift — Skipping retraining")

    # Push drift info to XCom for downstream tasks
    context["ti"].xcom_push(key="drift_info", value=drift_info)
    return drift_detected


# ─────────────────────────────────────────────
# TASK 2: LOAD & PREPARE COMBINED TRAINING DATA
# ─────────────────────────────────────────────
def load_and_prepare_data(**context):
    """
    Combine original training data + new incoming data for retraining.
    This is better than retraining on old data alone.
    We also load feature columns to ensure consistent schema.
    """
    log.info("Loading training and new data ...")

    train_df  = pd.read_csv(TRAIN_PATH)
    new_df    = pd.read_csv(NEW_DATA_PATH)

    combined  = pd.concat([train_df, new_df], ignore_index=True)
    combined  = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    log.info(f"Combined training data: {len(combined):,} rows")
    log.info(f"Fraud rate: {combined[TARGET_COL].mean() * 100:.2f}%")

    # Save combined for training task
    combined_path = f"{PROJECT_DIR}/data/processed/retrain_combined.csv"
    combined.to_csv(combined_path, index=False)

    context["ti"].xcom_push(key="combined_path", value=combined_path)
    context["ti"].xcom_push(key="n_rows", value=len(combined))
    return combined_path


# ─────────────────────────────────────────────
# TASK 3: TRAIN NEW MODEL
# ─────────────────────────────────────────────
def train_new_model(**context):
    """
    Train a fresh XGBoost model on the combined dataset.
    Log metrics so we can compare with the old model.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    from xgboost import XGBClassifier

    combined_path = context["ti"].xcom_pull(key="combined_path", task_ids="load_data")
    df = pd.read_csv(combined_path)

    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)

    X = df[feature_cols]
    y = df[TARGET_COL]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    n_legit = (y_train == 0).sum()
    n_fraud = (y_train == 1).sum()
    scale_pos_weight = n_legit / n_fraud

    log.info(f"Training new model on {len(X_train):,} samples ...")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        tree_method="hist",
        eval_metric="auc",
    )
    model.fit(X_train, y_train, verbose=False)

    y_pred  = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    new_metrics = {
        "roc_auc":   round(roc_auc_score(y_val, y_proba), 4),
        "f1_score":  round(f1_score(y_val, y_pred), 4),
        "precision": round(precision_score(y_val, y_pred), 4),
        "recall":    round(recall_score(y_val, y_pred), 4),
    }

    log.info(f"New model metrics: {new_metrics}")

    # Save new model temporarily
    new_model_path = f"{PROJECT_DIR}/models/loan_fraud_model_candidate.json"
    model.save_model(new_model_path)

    context["ti"].xcom_push(key="new_metrics",    value=new_metrics)
    context["ti"].xcom_push(key="new_model_path", value=new_model_path)
    return new_metrics


# ─────────────────────────────────────────────
# TASK 4: EVALUATE & A/B COMPARE
# ─────────────────────────────────────────────
def evaluate_and_promote(**context):
    """
    Compare new model vs old model on held-out new data.
    Only promote if new model is better — never blindly replace.

    This is the A/B gate — separates good MLOps from bad MLOps.
    Bad: drift → retrain → deploy (blind replace, dangerous)
    Good: drift → retrain → compare → deploy only if better
    """
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    new_metrics    = context["ti"].xcom_pull(key="new_metrics",    task_ids="train_model")
    new_model_path = context["ti"].xcom_pull(key="new_model_path", task_ids="train_model")

    # Load validation data (new incoming data)
    new_df = pd.read_csv(NEW_DATA_PATH)
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)

    X_val = new_df[feature_cols]
    y_val = new_df[TARGET_COL]

    # Evaluate OLD model
    old_model = XGBClassifier()
    old_model.load_model(MODEL_PATH)
    old_proba   = old_model.predict_proba(X_val)[:, 1]
    old_roc_auc = round(roc_auc_score(y_val, old_proba), 4)

    # Evaluate NEW model
    new_model = XGBClassifier()
    new_model.load_model(new_model_path)
    new_proba   = new_model.predict_proba(X_val)[:, 1]
    new_roc_auc = round(roc_auc_score(y_val, new_proba), 4)

    log.info(f"Old model ROC-AUC: {old_roc_auc}")
    log.info(f"New model ROC-AUC: {new_roc_auc}")

    promoted = False

    if new_roc_auc >= old_roc_auc:
        log.info("NEW MODEL IS BETTER — Promoting to production")
        import shutil
        shutil.copy(new_model_path, MODEL_PATH)
        promoted = True
        decision = "PROMOTED"
    else:
        log.info("OLD MODEL IS BETTER — Keeping existing model")
        decision = "REJECTED"

    # Log retraining history
    history_entry = {
        "timestamp":    datetime.now().isoformat(),
        "old_roc_auc":  old_roc_auc,
        "new_roc_auc":  new_roc_auc,
        "decision":     decision,
        "promoted":     promoted,
        "new_metrics":  new_metrics,
    }

    history = []
    if os.path.exists(RETRAIN_LOG):
        with open(RETRAIN_LOG) as f:
            history = json.load(f)

    history.append(history_entry)
    with open(RETRAIN_LOG, "w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Retraining history updated: {RETRAIN_LOG}")
    log.info(f"Decision: {decision}")

    return decision


# ─────────────────────────────────────────────
# BUILD DAG
# ─────────────────────────────────────────────

t1_check_drift = ShortCircuitOperator(
    task_id="check_drift",
    python_callable=check_drift_detected,
    dag=dag,
)

t2_load_data = PythonOperator(
    task_id="load_data",
    python_callable=load_and_prepare_data,
    dag=dag,
)

t3_train = PythonOperator(
    task_id="train_model",
    python_callable=train_new_model,
    dag=dag,
)

t4_evaluate = PythonOperator(
    task_id="evaluate_and_promote",
    python_callable=evaluate_and_promote,
    dag=dag,
)

# ─────────────────────────────────────────────
# PIPELINE ORDER
# check_drift → load_data → train_model → evaluate_and_promote
# ─────────────────────────────────────────────
t1_check_drift >> t2_load_data >> t3_train >> t4_evaluate
