from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# ----------------------------------------
# Load Saved Artifacts
# ----------------------------------------
model = joblib.load("xgb_fraud_model.pkl")
scaler = joblib.load("robust_scaler.pkl")
threshold = joblib.load("threshold.pkl")

# ----------------------------------------
# Feature order — MUST match training column order (V1..V28, Amount)
# ----------------------------------------
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

# ----------------------------------------
# Create FastAPI App
# ----------------------------------------
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="Predicts whether a credit card transaction is Fraud or Non-Fraud using XGBoost.",
    version="1.0"
)

# ----------------------------------------
# Input Schema
# ----------------------------------------
class Transaction(BaseModel):
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float


# ----------------------------------------
# Home Route
# ----------------------------------------
@app.get("/")
def home():
    return {
        "message": "Credit Card Fraud Detection API is Running",
        "model": "XGBoost",
        "threshold": float(threshold)
    }


# ----------------------------------------
# Prediction Route
# ----------------------------------------
@app.post("/predict")
def predict(data: Transaction):

    # Convert request to DataFrame, then force column order to match training
    df = pd.DataFrame([data.model_dump()])[FEATURE_COLS]

    # Scale Amount
    df["Amount"] = scaler.transform(df[["Amount"]])

    # Predict probability
    probability = model.predict_proba(df)[0][1]

    # Apply threshold
    prediction = int(probability >= threshold)

    return {
        "prediction": "Fraud" if prediction else "Non-Fraud",
        "fraud_probability": round(float(probability), 6),
        "threshold": round(float(threshold), 6),
        "is_fraud": prediction
    }


# ----------------------------------------
# Run directly with: python app.py
# Opens the interactive docs page automatically in your browser.
# (This block only runs on `python app.py` — it does NOT run if you
#  start the server with `uvicorn app:app --reload` instead.)
# ----------------------------------------
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open("http://127.0.0.1:8000/docs")

    threading.Timer(1.5, open_browser).start()
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)