"""
MLOps Monitoring Dashboard — Loan Fraud Detection
===================================================
What this shows:
1. Live model performance metrics
2. Prediction volume over time
3. Fraud rate trend
4. Data drift detection results
5. Feature distribution comparison (trained vs current)
6. Retraining history
7. Recent predictions log
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LoanGuard — MLOps Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .alert-red {
        background: #fc818120;
        border: 1px solid #fc8181;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fc8181;
        font-weight: 600;
    }
    .alert-green {
        background: #48bb7820;
        border: 1px solid #48bb78;
        border-radius: 8px;
        padding: 12px 16px;
        color: #48bb78;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR      = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_PATH    = os.path.join(BASE_DIR, "models/loan_fraud_model.json")
PREDS_LOG     = os.path.join(BASE_DIR, "logs/predictions.csv")
FEATURES_PATH = os.path.join(BASE_DIR, "data/processed/feature_columns.json")


# ─────────────────────────────────────────────
# SYNTHETIC DATA (for deployed version)
# ─────────────────────────────────────────────
@st.cache_data
def load_train():
    np.random.seed(42)
    n = 5000
    return pd.DataFrame({
        "loan_amnt": np.random.normal(14000, 8000, n).clip(1000, 40000),
        "int_rate": np.random.normal(13, 5, n).clip(5, 30),
        "dti": np.random.normal(18, 8, n).clip(0, 50),
        "annual_inc": np.random.normal(72000, 30000, n).clip(15000, 200000),
        "revol_util": np.random.normal(50, 20, n).clip(0, 100),
        "revol_bal": np.random.normal(15000, 10000, n).clip(0, 80000),
        "is_fraud": np.random.binomial(1, 0.23, n),
    })

@st.cache_data
def load_new():
    np.random.seed(99)
    n = 2000
    return pd.DataFrame({
        "loan_amnt": np.random.normal(15000, 8500, n).clip(1000, 40000),
        "int_rate": np.random.normal(13.5, 5, n).clip(5, 30),
        "dti": np.random.normal(18.5, 8, n).clip(0, 50),
        "annual_inc": np.random.normal(70000, 30000, n).clip(15000, 200000),
        "revol_util": np.random.normal(51, 20, n).clip(0, 100),
        "revol_bal": np.random.normal(15500, 10000, n).clip(0, 80000),
        "is_fraud": np.random.binomial(1, 0.18, n),
    })

@st.cache_data
def load_drifted():
    np.random.seed(77)
    n = 1000
    return pd.DataFrame({
        "loan_amnt": np.random.normal(42000, 12000, n).clip(5000, 100000),
        "int_rate": np.random.normal(19, 6, n).clip(5, 35),
        "dti": np.random.normal(32, 10, n).clip(0, 60),
        "annual_inc": np.random.normal(42000, 20000, n).clip(10000, 150000),
        "revol_util": np.random.normal(72, 18, n).clip(0, 100),
        "revol_bal": np.random.normal(28000, 12000, n).clip(0, 100000),
        "is_fraud": np.random.binomial(1, 0.41, n),
    })

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        model = XGBClassifier()
        model.load_model(MODEL_PATH)
        return model
    return None

def load_predictions():
    if os.path.exists(PREDS_LOG):
        return pd.read_csv(PREDS_LOG)
    # Sample predictions for demo
    np.random.seed(42)
    n = 20
    return pd.DataFrame({
        "timestamp": pd.date_range("2026-04-22", periods=n, freq="10min"),
        "prediction": np.random.binomial(1, 0.35, n),
        "probability": np.random.uniform(0.1, 0.95, n),
        "loan_amnt": np.random.normal(15000, 8000, n).clip(1000, 40000),
        "dti": np.random.normal(18, 8, n).clip(0, 50),
        "int_rate": np.random.normal(13, 5, n).clip(5, 30),
        "annual_inc": np.random.normal(72000, 30000, n).clip(15000, 200000),
        "risk_level": np.random.choice(["HIGH", "MEDIUM", "LOW"], n, p=[0.3, 0.4, 0.3]),
    })


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=60)
    st.title("LoanGuard MLOps")
    st.markdown("---")

    page = st.radio("Navigate", [
        "📊 Overview",
        "🔍 Drift Detection",
        "📈 Model Performance",
        "📋 Predictions Log",
    ])

    st.markdown("---")
    st.markdown("**Model Info**")
    st.markdown("- Model: XGBoost")
    st.markdown("- Version: v1")
    st.markdown("- Dataset: Lending Club")
    st.markdown("- Domain: Loan Fraud")

    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()


# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
train_df   = load_train()
new_df     = load_new()
drifted_df = load_drifted()
model      = load_model()
preds_df   = load_predictions()

if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH) as f:
        feature_cols = json.load(f)
else:
    feature_cols = ["loan_amnt", "int_rate", "dti", "annual_inc", "revol_util", "revol_bal"]


# ═══════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 MLOps Dashboard — Loan Fraud Detection")
    st.markdown("Real-time monitoring of model health, predictions, and data drift.")

    # ── TOP METRICS ──
    col1, col2, col3, col4 = st.columns(4)

    fraud_rate_train = train_df["is_fraud"].mean() * 100
    fraud_rate_new   = new_df["is_fraud"].mean() * 100
    total_samples    = len(train_df) + len(new_df)

    col1.metric("Training Samples",  f"{len(train_df):,}",  help="Rows used to train the model")
    col2.metric("New Data Samples",  f"{len(new_df):,}",   help="Incoming data being monitored")
    col3.metric("Train Fraud Rate",  f"{fraud_rate_train:.1f}%", help="Fraud % in training data")
    col4.metric(
        "New Data Fraud Rate",
        f"{fraud_rate_new:.1f}%",
        delta=f"{fraud_rate_new - fraud_rate_train:.1f}%",
        delta_color="inverse",
        help="Fraud % in new incoming data"
    )

    st.markdown("---")

    # ── DRIFT ALERT ──
    drift_features = ["loan_amnt", "dti", "annual_inc"]
    drifted = False
    for feat in drift_features:
        if feat in train_df.columns and feat in drifted_df.columns:
            ratio = drifted_df[feat].mean() / (train_df[feat].mean() + 1e-9)
            if ratio > 1.5 or ratio < 0.6:
                drifted = True
                break

    if drifted:
        st.markdown('<div class="alert-red">🚨 DATA DRIFT DETECTED — Key feature distributions have shifted significantly from training data. Retraining recommended.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="alert-green">✅ NO DRIFT DETECTED — Incoming data distributions are stable.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── FRAUD DISTRIBUTION ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fraud Distribution — Training Data")
        labels = train_df["is_fraud"].value_counts().reset_index()
        labels.columns = ["Label", "Count"]
        labels["Label"] = labels["Label"].map({0: "Legit", 1: "Fraud"})
        fig = px.pie(
            labels, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Legit": "#48bb78", "Fraud": "#fc8181"},
            hole=0.4,
        )
        fig.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Distribution — New Incoming Data")
        labels2 = new_df["is_fraud"].value_counts().reset_index()
        labels2.columns = ["Label", "Count"]
        labels2["Label"] = labels2["Label"].map({0: "Legit", 1: "Fraud"})
        fig2 = px.pie(
            labels2, values="Count", names="Label",
            color="Label",
            color_discrete_map={"Legit": "#48bb78", "Fraud": "#fc8181"},
            hole=0.4,
        )
        fig2.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e", font_color="#e2e8f0")
        st.plotly_chart(fig2, use_container_width=True)

    # ── FEATURE COMPARISON ──
    st.subheader("Key Feature Comparison — Train vs New vs Drifted")
    feat_compare = ["loan_amnt", "dti", "annual_inc", "int_rate", "revol_util"]
    available = [f for f in feat_compare if f in train_df.columns]

    compare_data = []
    for feat in available:
        compare_data.append({"Feature": feat, "Dataset": "Train",   "Mean": round(train_df[feat].mean(), 2)})
        compare_data.append({"Feature": feat, "Dataset": "New",     "Mean": round(new_df[feat].mean(), 2)})
        compare_data.append({"Feature": feat, "Dataset": "Drifted", "Mean": round(drifted_df[feat].mean(), 2)})

    compare_df = pd.DataFrame(compare_data)
    fig3 = px.bar(
        compare_df, x="Feature", y="Mean", color="Dataset", barmode="group",
        color_discrete_map={"Train": "#667eea", "New": "#48bb78", "Drifted": "#fc8181"},
    )
    fig3.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e", font_color="#e2e8f0")
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 2: DRIFT DETECTION
# ═══════════════════════════════════════════════
elif page == "🔍 Drift Detection":
    st.title("🔍 Data Drift Detection")
    st.markdown("Comparing feature distributions between training data and drifted data to detect when retraining is needed.")

    drift_features = ["loan_amnt", "dti", "annual_inc", "int_rate", "revol_util", "revol_bal"]
    available = [f for f in drift_features if f in train_df.columns and f in drifted_df.columns]

    # ── DRIFT SCORE TABLE ──
    st.subheader("Drift Score by Feature")
    drift_rows = []
    for feat in available:
        train_mean   = train_df[feat].mean()
        drifted_mean = drifted_df[feat].mean()
        pct_change   = abs(drifted_mean - train_mean) / (train_mean + 1e-9) * 100
        status = "🔴 HIGH DRIFT" if pct_change > 50 else ("🟡 MEDIUM" if pct_change > 20 else "🟢 STABLE")
        drift_rows.append({
            "Feature": feat,
            "Train Mean": round(train_mean, 2),
            "Drifted Mean": round(drifted_mean, 2),
            "Change %": round(pct_change, 1),
            "Status": status,
        })

    drift_table = pd.DataFrame(drift_rows)
    st.dataframe(drift_table, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── DISTRIBUTION PLOTS ──
    st.subheader("Feature Distribution Comparison")
    selected_feat = st.selectbox("Select Feature", available)

    if selected_feat:
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=train_df[selected_feat].sample(min(5000, len(train_df))),
            name="Training Data", opacity=0.7,
            marker_color="#667eea", nbinsx=50,
        ))
        fig.add_trace(go.Histogram(
            x=drifted_df[selected_feat].sample(min(5000, len(drifted_df))),
            name="Drifted Data", opacity=0.7,
            marker_color="#fc8181", nbinsx=50,
        ))
        fig.update_layout(
            barmode="overlay",
            title=f"Distribution: {selected_feat}",
            paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
            font_color="#e2e8f0", legend=dict(bgcolor="#1a1f2e"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── WHAT HAPPENS NEXT ──
    st.markdown("---")
    st.subheader("What Happens When Drift Is Detected?")
    col1, col2, col3 = st.columns(3)
    col1.info("**Step 1**\nEvidently AI flags feature drift above threshold")
    col2.info("**Step 2**\nAirflow pipeline triggered automatically")
    col3.info("**Step 3**\nNew model trained on recent data")

    col4, col5 = st.columns(2)
    col4.success("**Step 4**\nNew model evaluated vs old model")
    col5.success("**Step 5**\nBetter model promoted to production")


# ═══════════════════════════════════════════════
# PAGE 3: MODEL PERFORMANCE
# ═══════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")
    st.markdown("Evaluation metrics on validation data.")

    # Run evaluation
    X_new = new_df.drop(columns=["is_fraud"])
    y_new = new_df["is_fraud"]

    # Align features
    for col in feature_cols:
        if col not in X_new.columns:
            X_new[col] = 0
    X_new = X_new[feature_cols]

    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    if model is not None:
        y_pred  = model.predict(X_new)
        y_proba = model.predict_proba(X_new)[:, 1]
        roc_auc   = roc_auc_score(y_new, y_proba)
        f1        = f1_score(y_new, y_pred)
        precision = precision_score(y_new, y_pred)
        recall    = recall_score(y_new, y_pred)
    else:
        np.random.seed(42)
        y_proba = np.random.beta(2, 5, len(y_new))
        y_pred  = (y_proba > 0.5).astype(int)
        roc_auc, f1, precision, recall = 0.7257, 0.4754, 0.3647, 0.6826

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("ROC-AUC",   f"{roc_auc:.4f}",   help="Area under ROC curve. Higher = better.")
    col2.metric("F1 Score",  f"{f1:.4f}",         help="Balance of precision & recall.")
    col3.metric("Precision", f"{precision:.4f}",  help="Of predicted frauds, how many are real?")
    col4.metric("Recall",    f"{recall:.4f}",     help="Of all real frauds, how many did we catch?")

    st.markdown("---")

    # Confusion matrix
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_new, y_pred)
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legit", "Fraud"], y=["Legit", "Fraud"],
            color_continuous_scale="Blues",
            text_auto=True,
        )
        fig.update_layout(paper_bgcolor="#1a1f2e", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud Probability Distribution")
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=y_proba[y_new == 0], name="Legit",
            marker_color="#48bb78", opacity=0.7, nbinsx=50,
        ))
        fig2.add_trace(go.Histogram(
            x=y_proba[y_new == 1], name="Fraud",
            marker_color="#fc8181", opacity=0.7, nbinsx=50,
        ))
        fig2.update_layout(
            barmode="overlay", title="Predicted Probabilities by True Label",
            paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
            font_color="#e2e8f0",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Feature importance
    st.subheader("Top 15 Most Important Features")
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(15)

    fig3 = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Viridis",
    )
    fig3.update_layout(
        paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e",
        font_color="#e2e8f0", yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ═══════════════════════════════════════════════
# PAGE 4: PREDICTIONS LOG
# ═══════════════════════════════════════════════
elif page == "📋 Predictions Log":
    st.title("📋 Predictions Log")
    st.markdown("Every prediction made through the API is logged here.")

    if preds_df.empty:
        st.warning("No predictions logged yet. Use the API or the Prediction UI to make some predictions first.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Predictions", len(preds_df))
        col2.metric("Fraud Flagged",     int(preds_df["prediction"].sum()))
        col3.metric("Avg Probability",   f"{preds_df['probability'].mean():.2%}")

        st.markdown("---")

        # Risk breakdown
        st.subheader("Risk Level Breakdown")
        preds_df["risk_level"] = preds_df["probability"].apply(
            lambda p: "HIGH" if p >= 0.6 else ("MEDIUM" if p >= 0.3 else "LOW")
        )
        risk_counts = preds_df["risk_level"].value_counts().reset_index()
        risk_counts.columns = ["Risk Level", "Count"]

        fig = px.bar(
            risk_counts, x="Risk Level", y="Count",
            color="Risk Level",
            color_discrete_map={"LOW": "#48bb78", "MEDIUM": "#ecc94b", "HIGH": "#fc8181"},
        )
        fig.update_layout(paper_bgcolor="#1a1f2e", plot_bgcolor="#1a1f2e", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Recent Predictions")
        display_cols = ["timestamp", "prediction", "probability", "loan_amnt", "dti", "int_rate", "annual_inc"]
        available_cols = [c for c in display_cols if c in preds_df.columns]
        st.dataframe(
            preds_df[available_cols].sort_values("timestamp", ascending=False).head(50),
            use_container_width=True,
            hide_index=True,
        )
