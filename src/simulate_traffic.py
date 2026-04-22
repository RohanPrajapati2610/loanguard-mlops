import pandas as pd
import requests
import time
import json
import random
from datetime import datetime

API_URL = "http://localhost:8000/predict"
DELAY_SECONDS = 0.3  # seconds between each transaction

def load_data():
    new_data = pd.read_csv("data/processed/new_data.csv")
    drifted_data = pd.read_csv("data/processed/new_data.csv").copy()

    # Simulate drift: loan amounts 3x bigger, DTI higher, income lower
    drifted_data["loan_amnt"] = drifted_data["loan_amnt"] * 3
    drifted_data["funded_amnt"] = drifted_data["funded_amnt"] * 3
    drifted_data["dti"] = drifted_data["dti"] * 1.8
    drifted_data["annual_inc"] = drifted_data["annual_inc"] * 0.6
    drifted_data["int_rate"] = drifted_data["int_rate"] * 1.5
    drifted_data["revol_util"] = drifted_data["revol_util"] * 1.4

    return new_data, drifted_data

def send_transaction(row, phase):
    payload = {
        "loan_amnt": float(row.get("loan_amnt", 10000)),
        "funded_amnt": float(row.get("funded_amnt", 10000)),
        "term": int(row.get("term", 36)),
        "int_rate": float(row.get("int_rate", 12.0)),
        "installment": float(row.get("installment", 300)),
        "annual_inc": float(row.get("annual_inc", 60000)),
        "dti": float(row.get("dti", 15.0)),
        "delinq_2yrs": int(row.get("delinq_2yrs", 0)),
        "inq_last_6mths": int(row.get("inq_last_6mths", 1)),
        "open_acc": int(row.get("open_acc", 8)),
        "pub_rec": int(row.get("pub_rec", 0)),
        "revol_bal": float(row.get("revol_bal", 10000)),
        "revol_util": float(row.get("revol_util", 40.0)),
        "total_acc": int(row.get("total_acc", 15)),
        "mort_acc": int(row.get("mort_acc", 1)),
        "pub_rec_bankruptcies": int(row.get("pub_rec_bankruptcies", 0)),
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        result = response.json()
        timestamp = datetime.now().strftime("%H:%M:%S")
        risk = result.get("risk_level", "UNKNOWN")
        prob = result.get("fraud_probability", 0)
        print(f"[{timestamp}] [{phase}] loan=${payload['loan_amnt']:,.0f} | {risk} | prob={prob:.2%}")
    except Exception as e:
        print(f"[ERROR] Could not reach API: {e}")

def run_simulation():
    print("=" * 60)
    print("  LOAN FRAUD DETECTION — TRAFFIC SIMULATOR")
    print("=" * 60)
    print(f"  API: {API_URL}")
    print(f"  Delay: {DELAY_SECONDS}s between transactions")
    print("=" * 60)

    new_data, drifted_data = load_data()

    # Phase 1: Normal traffic (200 transactions)
    print("\n[PHASE 1] Sending normal loan applications...")
    print("  Inspector should remain calm — data looks like training data\n")

    sample = new_data.sample(n=min(50, len(new_data)), random_state=42)
    for i, (_, row) in enumerate(sample.iterrows()):
        send_transaction(row, "NORMAL")
        if i % 20 == 0 and i > 0:
            print(f"  --- {i} normal transactions sent ---")
        time.sleep(DELAY_SECONDS)

    print("\n" + "=" * 60)
    print("[PHASE 2] Switching to DRIFTED data...")
    print("  Loan amounts 3x bigger, DTI higher, income lower")
    print("  Inspector should detect drift and trigger retraining!")
    print("=" * 60 + "\n")

    # Phase 2: Drifted traffic (100 transactions)
    drifted_sample = drifted_data.sample(n=min(30, len(drifted_data)), random_state=99)
    for i, (_, row) in enumerate(drifted_sample.iterrows()):
        send_transaction(row, "DRIFTED")
        if i % 20 == 0 and i > 0:
            print(f"  --- {i} drifted transactions sent ---")
        time.sleep(DELAY_SECONDS)

    print("\n" + "=" * 60)
    print("  SIMULATION COMPLETE")
    print(f"  Normal transactions: 200")
    print(f"  Drifted transactions: 100")
    print(f"  Check Streamlit dashboard for drift detection results")
    print("=" * 60)

if __name__ == "__main__":
    run_simulation()
