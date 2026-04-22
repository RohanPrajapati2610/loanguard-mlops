"""
Evidently AI Drift Report Generator — Loan Fraud Detection
===========================================================
What this does:
1. Loads training data (reference) and new/drifted data (current)
2. Runs Evidently AI to compare distributions
3. Generates a full HTML report showing:
   - Which features drifted
   - By how much
   - Statistical test used
   - Visual distribution comparisons
4. Saves report to logs/drift_report.html
5. Saves a JSON summary for the dashboard and Airflow to read

Why Evidently?
- Industry standard tool for ML monitoring
- Used at Netflix, Booking.com, and many fintechs
- Generates audit-ready reports automatically
"""

import json
import logging
import os
from datetime import datetime

import pandas as pd
from evidently import Dataset, DataDefinition
from evidently.presets import DataDriftPreset
from evidently import Report

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TRAIN_PATH     = "data/processed/train.csv"
NEW_PATH       = "data/processed/new_data.csv"
DRIFTED_PATH   = "data/drifted/drifted_data.csv"
REPORT_DIR     = "logs/drift_reports"
SUMMARY_PATH   = "logs/drift_summary.json"

# Features to monitor (most predictive + most likely to drift)
MONITOR_FEATURES = [
    "loan_amnt", "funded_amnt", "term", "int_rate",
    "installment", "annual_inc", "dti", "delinq_2yrs",
    "inq_last_6mths", "open_acc", "revol_bal", "revol_util",
    "total_acc", "mort_acc",
]


def load_data():
    log.info("Loading datasets ...")
    train_df   = pd.read_csv(TRAIN_PATH)
    new_df     = pd.read_csv(NEW_PATH)
    drifted_df = pd.read_csv(DRIFTED_PATH)

    # Keep only numeric monitor features that exist in all datasets
    available = [
        f for f in MONITOR_FEATURES
        if f in train_df.columns and f in new_df.columns and f in drifted_df.columns
    ]
    log.info(f"Monitoring {len(available)} features: {available}")
    return train_df[available], new_df[available], drifted_df[available], available


def run_drift_report(reference_df, current_df, label, available_features):
    """Run Evidently drift report comparing reference vs current data."""
    log.info(f"Running drift report: {label} ...")

    os.makedirs(REPORT_DIR, exist_ok=True)

    # Define numeric columns explicitly
    column_mapping = DataDefinition(
        numerical_columns=available_features,
    )

    reference_dataset = Dataset.from_pandas(reference_df, data_definition=column_mapping)
    current_dataset   = Dataset.from_pandas(current_df,   data_definition=column_mapping)

    report = Report(metrics=[DataDriftPreset()])
    my_eval = report.run(reference_data=reference_dataset, current_data=current_dataset)

    # Save HTML report
    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path  = os.path.join(REPORT_DIR, f"drift_{label}_{timestamp}.html")
    my_eval.save_html(html_path)
    log.info(f"HTML report saved: {html_path}")

    # Extract summary from JSON
    result_json = json.loads(my_eval.json())
    return result_json, html_path


def extract_drift_summary(result_json, label):
    """Pull key drift metrics from Evidently JSON output."""
    summary = {
        "label": label,
        "timestamp": datetime.now().isoformat(),
        "features": {},
        "drift_detected": False,
        "drifted_feature_count": 0,
        "total_features": 0,
        "share_drifted": 0,
    }

    try:
        metrics = result_json.get("metrics", [])
        THRESHOLD = 0.1  # Wasserstein distance threshold

        for metric in metrics:
            metric_name = metric.get("metric_name", "")
            value       = metric.get("value", None)
            config      = metric.get("config", {})

            # Dataset-level: count of drifted columns
            if "DriftedColumnsCount" in metric_name and isinstance(value, dict):
                summary["drifted_feature_count"] = int(value.get("count", 0))
                summary["share_drifted"]         = round(value.get("share", 0), 4)
                summary["drift_detected"]        = summary["drifted_feature_count"] > 0

            # Per-column drift score
            elif "ValueDrift" in metric_name and isinstance(value, (int, float)):
                col_name      = config.get("column", "unknown")
                drift_score   = round(float(value), 4)
                drift_detected = drift_score > THRESHOLD
                summary["features"][col_name] = {
                    "drift_detected": drift_detected,
                    "drift_score":    drift_score,
                    "stat_test":      config.get("method", "Wasserstein"),
                    "threshold":      config.get("threshold", THRESHOLD),
                }

        summary["total_features"] = len(summary["features"])

    except Exception as e:
        log.warning(f"Could not fully parse Evidently output: {e}")

    return summary


def save_summary(new_summary, drifted_summary):
    """Save drift summaries to JSON for dashboard and Airflow to read."""
    combined = {
        "generated_at": datetime.now().isoformat(),
        "new_data_vs_train":     new_summary,
        "drifted_data_vs_train": drifted_summary,
    }
    with open(SUMMARY_PATH, "w") as f:
        json.dump(combined, f, indent=2)
    log.info(f"Drift summary saved: {SUMMARY_PATH}")


def print_summary(summary):
    log.info(f"\n{'='*50}")
    log.info(f"Drift Report: {summary['label']}")
    log.info(f"Drift Detected: {summary['drift_detected']}")
    log.info(f"Drifted Features: {summary['drifted_feature_count']} / {summary['total_features']}")
    for feat, info in summary["features"].items():
        status = "🔴 DRIFT" if info["drift_detected"] else "🟢 OK"
        log.info(f"  {status} {feat}: score={info['drift_score']} (test: {info['stat_test']})")
    log.info(f"{'='*50}\n")


def main():
    os.makedirs("logs", exist_ok=True)

    train_df, new_df, drifted_df, available = load_data()

    # Report 1: Training vs New (should be stable)
    result_new, html_new         = run_drift_report(train_df, new_df, "new_data", available)
    new_summary                  = extract_drift_summary(result_new, "new_data_vs_train")
    print_summary(new_summary)

    # Report 2: Training vs Drifted (should show HIGH drift)
    result_drifted, html_drifted = run_drift_report(train_df, drifted_df, "drifted_data", available)
    drifted_summary              = extract_drift_summary(result_drifted, "drifted_data_vs_train")
    print_summary(drifted_summary)

    save_summary(new_summary, drifted_summary)

    log.info("Done! Open these reports in your browser:")
    log.info(f"  {html_new}")
    log.info(f"  {html_drifted}")
    log.info(f"  Drift summary JSON: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
