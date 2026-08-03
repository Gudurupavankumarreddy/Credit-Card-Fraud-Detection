"""
Credit Card Fraud Detection — Streamlit Dashboard
Single-file app: Home | Demo Prediction | Batch Prediction | About

Expects these files in the SAME folder as this script:
    - xgb_fraud_model.pkl
    - robust_scaler.pkl
    - threshold.pkl
    - creditcard.csv   (Kaggle creditcard.csv — Time, V1..V28, Amount, Class)

Run with:  streamlit run streamlit_app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "xgb_fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "robust_scaler.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "threshold.pkl")
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Hardcoded from your notebook's model comparison (Step 6)
MODEL_COMPARISON = pd.DataFrame([
    {"Model": "XGBoost",       "Precision (Fraud)": 0.88, "Recall (Fraud)": 0.85, "F1-Score (Fraud)": 0.86, "ROC-AUC": 0.9691},
    {"Model": "Random Forest", "Precision (Fraud)": 0.96, "Recall (Fraud)": 0.76, "F1-Score (Fraud)": 0.85, "ROC-AUC": 0.9580},
    {"Model": "CatBoost",      "Precision (Fraud)": 0.66, "Recall (Fraud)": 0.86, "F1-Score (Fraud)": 0.74, "ROC-AUC": 0.9667},
    {"Model": "AdaBoost",      "Precision (Fraud)": 0.71, "Recall (Fraud)": 0.73, "F1-Score (Fraud)": 0.72, "ROC-AUC": 0.9792},
]).sort_values("F1-Score (Fraud)", ascending=False)

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)

# ------------------------------------------------------------------
# Cached loaders
# ------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    threshold = joblib.load(THRESHOLD_PATH)
    return model, scaler, threshold


@st.cache_resource
def load_explainer(_model):
    # underscore prefix tells st.cache_resource not to hash the model object
    return shap.TreeExplainer(_model)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def prepare_features(df: pd.DataFrame, scaler) -> pd.DataFrame:
    """Drop non-model columns, reorder to match training, scale Amount."""
    df = df.copy()
    for col in ["Time", "Hour", "Class"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df = df[FEATURE_COLS]
    df["Amount"] = scaler.transform(df[["Amount"]])
    return df


def predict(df_features: pd.DataFrame, model, threshold):
    probs = model.predict_proba(df_features)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs, preds


# ------------------------------------------------------------------
# Load everything once
# ------------------------------------------------------------------
try:
    model, scaler, threshold = load_artifacts()
    artifacts_ok = True
except Exception as e:
    artifacts_ok = False
    load_error = str(e)

try:
    raw_df = load_data()
    data_ok = True
except Exception as e:
    data_ok = False
    data_error = str(e)

# ------------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------------
st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Home", "🔍 Demo Prediction", "📁 Batch Prediction", "ℹ️ About"],
)

if not artifacts_ok:
    st.sidebar.error("Model artifacts not found. Check file paths at the top of the script.")
if not data_ok:
    st.sidebar.warning("creditcard.csv not found — Home stats and Demo mode will be limited.")

# ====================================================================
# HOME
# ====================================================================
if page == "🏠 Home":
    st.title("💳 Credit Card Fraud Detection")
    st.caption("XGBoost classifier on the Kaggle European cardholders dataset (Sept 2013)")

    if data_ok:
        n_total = len(raw_df)
        n_fraud = int(raw_df["Class"].sum())
        fraud_pct = 100 * n_fraud / n_total

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Transactions", f"{n_total:,}")
        c2.metric("Fraud Cases", f"{n_fraud:,}")
        c3.metric("Fraud Rate", f"{fraud_pct:.3f}%")
        c4.metric("Features", "28 PCA + Amount")
    else:
        st.info("Upload/point to creditcard.csv to see live dataset stats here.")

    st.divider()

    st.subheader("Model Performance")
    best_row = MODEL_COMPARISON.iloc[0]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Selected Model", best_row["Model"])
    m2.metric("Precision (Fraud)", f"{best_row['Precision (Fraud)']:.2f}")
    m3.metric("Recall (Fraud)", f"{best_row['Recall (Fraud)']:.2f}")
    m4.metric("ROC-AUC", f"{best_row['ROC-AUC']:.3f}")

    st.dataframe(MODEL_COMPARISON.reset_index(drop=True), use_container_width=True)

    if artifacts_ok:
        st.caption(f"Decision threshold (tuned for best F1): **{threshold:.4f}**")

    st.divider()
    st.subheader("How it works")
    st.markdown(
        """
        1. **Demo Prediction** — loads a real sample transaction (legitimate or fraud) straight
           from the dataset, no manual V1–V28 entry needed.
        2. **Batch Prediction** — upload your own CSV of transactions and score all of them at once.
        3. Every prediction is explained with **SHAP** — showing which features pushed the score
           toward fraud or legitimate.
        """
    )

# ====================================================================
# DEMO PREDICTION
# ====================================================================
elif page == "🔍 Demo Prediction":
    st.title("🔍 Demo Prediction")

    if not (artifacts_ok and data_ok):
        st.error("Need both model artifacts and creditcard.csv loaded to run the demo.")
        st.stop()

    st.write("Pick a real transaction from the dataset and see how the model scores it.")

    mode = st.radio(
        "Prediction Mode",
        ["🟢 Sample Legitimate Transaction", "🔴 Sample Fraud Transaction"],
        horizontal=True,
    )

    if "demo_row" not in st.session_state:
        st.session_state.demo_row = None

    if st.button("🎲 Load Sample", type="primary"):
        target_class = 0 if "Legitimate" in mode else 1
        st.session_state.demo_row = raw_df[raw_df["Class"] == target_class].sample(1)

    if st.session_state.demo_row is not None:
        row = st.session_state.demo_row
        with st.expander("View raw transaction data", expanded=False):
            st.dataframe(row, use_container_width=True)

        if st.button("⚡ Predict"):
            X = prepare_features(row, scaler)
            probs, preds = predict(X, model, threshold)
            prob = probs[0]
            pred = preds[0]

            st.divider()
            if pred == 1:
                st.error("🚨 **FRAUD DETECTED**")
            else:
                st.success("✅ **LEGITIMATE TRANSACTION**")

            r1, r2, r3 = st.columns(3)
            r1.metric("Fraud Probability", f"{prob*100:.2f}%")
            r2.metric("Threshold", f"{threshold*100:.2f}%")
            r3.metric("Actual Label", "Fraud" if row["Class"].values[0] == 1 else "Non-Fraud")

            st.subheader("📈 Why this prediction?")
            explainer = load_explainer(model)
            shap_values = explainer(X)

            fig = plt.figure(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

# ====================================================================
# BATCH PREDICTION
# ====================================================================
elif page == "📁 Batch Prediction":
    st.title("📁 Batch Prediction")

    if not artifacts_ok:
        st.error("Model artifacts not found — cannot run batch prediction.")
        st.stop()

    st.write(
        "Upload a CSV with columns `V1..V28` and `Amount` (a `Time`/`Class` column, "
        "if present, will be ignored for scoring)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
            X = prepare_features(batch_df, scaler)
            probs, preds = predict(X, model, threshold)

            result_df = batch_df.copy()
            result_df["Fraud Probability"] = probs
            result_df["Prediction"] = np.where(preds == 1, "Fraud", "Non-Fraud")

            n_fraud = int(preds.sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions Scored", f"{len(result_df):,}")
            c2.metric("Flagged as Fraud", f"{n_fraud:,}")
            c3.metric("Fraud Rate", f"{100*n_fraud/len(result_df):.2f}%")

            st.dataframe(
                result_df[["Fraud Probability", "Prediction"]].join(
                    batch_df.reset_index(drop=True)
                ).head(200),
                use_container_width=True,
            )
            if len(result_df) > 200:
                st.caption(f"Showing first 200 of {len(result_df):,} rows.")

            csv_bytes = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download Predictions",
                data=csv_bytes,
                file_name="fraud_predictions.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ====================================================================
# ABOUT
# ====================================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown(
        """
        **Credit Card Fraud Detection** — end-to-end ML project on the Kaggle
        "creditcard.csv" dataset (European cardholders, September 2013).

        - **284,807** transactions over 2 days, **492** frauds (0.172%) — highly imbalanced.
        - Features are PCA-transformed (`V1`–`V28`) plus `Time` and `Amount`.
        - Compared **Random Forest, XGBoost, CatBoost, AdaBoost**; selected
          **XGBoost** for the best F1-score on the fraud class, with a threshold tuned
          via precision-recall curve.
        - Model explainability via **SHAP** (TreeExplainer) — waterfall plots per prediction.
        - This dashboard: **Streamlit** for the UI, model served via a companion **FastAPI**
          service for programmatic access.

        **Links**
        - GitHub: [github.com/Gudurupavankumarreddy](https://github.com/Gudurupavankumarreddy)
        - Portfolio: [pavan-data-portfolio.netlify.app](https://pavan-data-portfolio.netlify.app)
        """
    )