"""
Data Preprocessing Pipeline — Loan Fraud Detection
====================================================
What this file does:
1. Loads raw Lending Club data
2. Creates fraud labels from loan_status
3. Drops useless columns
4. Handles missing values
5. Encodes categorical features
6. Splits into train / new_data / drifted_data
7. Saves processed files to data/processed/ and data/drifted/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1: Load Raw Data
# ─────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    log.info(f"Loading data from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Loaded {len(df):,} rows and {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# STEP 2: Create Fraud Label
# ─────────────────────────────────────────────
# In real loan fraud detection:
#   - "Charged Off" = borrower defaulted, bank lost money = FRAUD
#   - "Fully Paid"  = borrower paid back = LEGIT
#   - "Current"     = still paying = LEGIT
# We map these to binary: 1 = fraud/default, 0 = legit
def create_label(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Creating fraud labels ...")

    fraud_statuses = [
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Late (16-30 days)",
        "Does not meet the credit policy. Status:Charged Off",
    ]

    # Keep only rows with clear labels (drop "Current", "In Grace Period" etc.)
    clear_statuses = fraud_statuses + [
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    ]
    df = df[df["loan_status"].isin(clear_statuses)].copy()

    df["fraud_label"] = df["loan_status"].apply(
        lambda x: 1 if x in fraud_statuses else 0
    )

    log.info(f"Label distribution:\n{df['fraud_label'].value_counts()}")
    log.info(f"Fraud rate: {df['fraud_label'].mean() * 100:.2f}%")
    return df


# ─────────────────────────────────────────────
# STEP 3: Drop Useless Columns
# ─────────────────────────────────────────────
# Why drop these?
# - ID columns (id, member_id) — unique per row, no predictive value
# - URL/text columns — too noisy for our model
# - Columns that would cause data leakage (info from AFTER loan was issued)
#   e.g. total_pymnt, recoveries — we wouldn't know these at prediction time
LEAKAGE_COLS = [
    "id", "member_id", "url", "desc", "title", "zip_code",
    "loan_status",          # replaced by fraud_label
    "out_prncp", "out_prncp_inv",
    "total_pymnt", "total_pymnt_inv", "total_rec_prncp",
    "total_rec_int", "total_rec_late_fee", "recoveries",
    "collection_recovery_fee", "last_pymnt_d", "last_pymnt_amnt",
    "next_pymnt_d", "last_credit_pull_d",
]

def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Dropping leakage and useless columns ...")
    cols_to_drop = [c for c in LEAKAGE_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    log.info(f"Columns remaining: {df.shape[1]}")
    return df


# ─────────────────────────────────────────────
# STEP 4: Keep Only Useful Columns
# ─────────────────────────────────────────────
# From 145 columns, we keep the most predictive ones
# These are features known AT THE TIME of loan application
KEEP_COLS = [
    "loan_amnt",        # how much they borrowed
    "funded_amnt",      # how much was actually funded
    "term",             # 36 or 60 months
    "int_rate",         # interest rate
    "installment",      # monthly payment
    "grade",            # LendingClub risk grade (A-G)
    "sub_grade",        # more granular grade
    "emp_length",       # employment length
    "home_ownership",   # rent/own/mortgage
    "annual_inc",       # annual income
    "verification_status",  # income verified?
    "purpose",          # reason for loan
    "dti",              # debt-to-income ratio
    "delinq_2yrs",      # delinquencies in past 2 years
    "inq_last_6mths",   # credit inquiries last 6 months
    "open_acc",         # number of open accounts
    "pub_rec",          # public records (bankruptcies etc.)
    "revol_bal",        # revolving balance
    "revol_util",       # revolving line utilization
    "total_acc",        # total accounts
    "addr_state",       # state
    "mort_acc",         # mortgage accounts
    "pub_rec_bankruptcies",  # bankruptcies
    "fraud_label",      # our target variable
]

def select_features(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Selecting key features ...")
    available = [c for c in KEEP_COLS if c in df.columns]
    df = df[available].copy()
    log.info(f"Selected {len(available)} features")
    return df


# ─────────────────────────────────────────────
# STEP 5: Handle Missing Values
# ─────────────────────────────────────────────
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Handling missing values ...")

    # Numeric columns → fill with median (robust to outliers)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "fraud_label"]
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Categorical columns → fill with "Unknown"
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df[cat_cols] = df[cat_cols].fillna("Unknown")

    log.info(f"Missing values remaining: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────
# STEP 6: Encode Categorical Features
# ─────────────────────────────────────────────
def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Encoding categorical features ...")

    # term: " 36 months" → 36
    if "term" in df.columns:
        df["term"] = df["term"].str.extract(r"(\d+)").astype(float)

    # int_rate: "12.5%" → 12.5
    if "int_rate" in df.columns:
        df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "").astype(float)

    # revol_util: "45.2%" → 45.2
    if "revol_util" in df.columns:
        df["revol_util"] = df["revol_util"].astype(str).str.replace("%", "").astype(float)

    # emp_length: "10+ years" → 10, "< 1 year" → 0
    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].str.extract(r"(\d+)").astype(float)

    # One-hot encode remaining categoricals
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    log.info(f"One-hot encoding: {cat_cols}")
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    log.info(f"Final shape after encoding: {df.shape}")
    return df


# ─────────────────────────────────────────────
# STEP 7: Split into Train / New / Drifted
# ─────────────────────────────────────────────
# This is where we simulate real-world data flow:
# - train_data    → model learns from this (70%)
# - new_data      → simulates future normal transactions (20%)
# - drifted_data  → simulates drift: we artificially shift features (10%)
def split_and_simulate_drift(df: pd.DataFrame):
    log.info("Splitting data ...")

    # Sort by index to simulate time order
    df = df.reset_index(drop=True)
    n = len(df)

    train_end   = int(n * 0.70)
    new_end     = int(n * 0.90)

    train_df   = df.iloc[:train_end].copy()
    new_df     = df.iloc[train_end:new_end].copy()
    drifted_df = df.iloc[new_end:].copy()

    log.info(f"Train:   {len(train_df):,} rows")
    log.info(f"New:     {len(new_df):,} rows")
    log.info(f"Drifted: {len(drifted_df):,} rows")

    # Simulate drift: shift loan amounts and dti upward
    # This mimics: inflation causes borrowers to request larger loans
    # and take on more debt (higher dti)
    log.info("Simulating drift in drifted dataset ...")
    if "loan_amnt" in drifted_df.columns:
        drifted_df["loan_amnt"] = drifted_df["loan_amnt"] * 2.5
    if "dti" in drifted_df.columns:
        drifted_df["dti"] = drifted_df["dti"] * 1.8
    if "annual_inc" in drifted_df.columns:
        drifted_df["annual_inc"] = drifted_df["annual_inc"] * 0.7

    return train_df, new_df, drifted_df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/drifted", exist_ok=True)

    df = load_data("data/raw/loan.csv")
    df = create_label(df)
    df = drop_useless_columns(df)
    df = select_features(df)
    df = handle_missing(df)
    df = encode_categoricals(df)

    train_df, new_df, drifted_df = split_and_simulate_drift(df)

    log.info("Saving processed files ...")
    train_df.to_csv("data/processed/train.csv", index=False)
    new_df.to_csv("data/processed/new_data.csv", index=False)
    drifted_df.to_csv("data/drifted/drifted_data.csv", index=False)

    log.info("Done! Files saved:")
    log.info("  data/processed/train.csv")
    log.info("  data/processed/new_data.csv")
    log.info("  data/drifted/drifted_data.csv")


if __name__ == "__main__":
    main()
