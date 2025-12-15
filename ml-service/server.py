# ============================================================
#  ML-SERVICE FINAL VERSION (Prediction + Anomaly + Trend)
#  FastAPI - Hugging Face Production Ready
# ============================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import requests
import os

app = FastAPI(title="Predictive Maintenance ML Service")

# ============================================================
# CONFIG
# ============================================================

BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL:
    print("⚠️ BACKEND_URL not set. /latest and /trend endpoints disabled.")

REQUEST_TIMEOUT = 5  # seconds

# ============================================================
# LOAD FAILURE PREDICTION MODEL
# ============================================================

MODEL_PATH = "./models/model.pkl"
SCALER_PATH = "./models/scaler.pkl"
FAILURE_LE_PATH = "./models/failure_le.pkl"
TYPE_LE_PATH = "./models/type_le.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
failure_le = joblib.load(FAILURE_LE_PATH)
type_le = joblib.load(TYPE_LE_PATH)

FEATURE_ORDER = [
    "Type", "air_temp", "process_temp", "rpm", "torque", "tool_wear",
    "temp_diff", "torque_rpm_ratio", "power", "temp_stress",
    "wear_per_rpm", "torque_squared", "tool_wear_squared",
    "torque_wear_interaction", "rpm_temp_interaction", "power_wear_ratio",
]

# ============================================================
# REQUEST SCHEMAS
# ============================================================

class PredictRequest(BaseModel):
    Type: str
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float

class AnomalyRequest(BaseModel):
    Type: str
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def make_features_from_input(d: dict) -> pd.DataFrame:
    df = pd.DataFrame([d])

    if d["Type"] not in type_le.classes_:
        raise ValueError(f"Unknown Type '{d['Type']}'")

    df["Type"] = type_le.transform([d["Type"]])[0]

    df["temp_diff"] = df["process_temp"] - df["air_temp"]
    df["torque_rpm_ratio"] = df["torque"] / (df["rpm"] + 1e-5)
    df["power"] = df["torque"] * df["rpm"] / 9.5488
    df["temp_stress"] = df["process_temp"] / (df["air_temp"] + 1e-5)
    df["wear_per_rpm"] = df["tool_wear"] / (df["rpm"] + 1e-5)
    df["torque_squared"] = df["torque"] ** 2
    df["tool_wear_squared"] = df["tool_wear"] ** 2
    df["torque_wear_interaction"] = df["torque"] * df["tool_wear"]
    df["rpm_temp_interaction"] = df["rpm"] * df["temp_diff"]
    df["power_wear_ratio"] = df["power"] / (df["tool_wear"] + 1e-5)

    return df[FEATURE_ORDER]

def get_status(pred_failure: str, anomaly_label: int = 0):
    if pred_failure not in ["No Failure", "NO_FAILURE"]:
        return "CRITICAL"
    if anomaly_label == 1:
        return "WARNING"
    return "NORMAL"

# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(req: PredictRequest):
    try:
        df = make_features_from_input(req.dict())
        X = scaler.transform(df)
        pred_encoded = model.predict(X)[0]

        probabilities = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
        pred_failure = failure_le.inverse_transform([pred_encoded])[0]

        prob_dict = {}
        if probabilities is not None:
            for i, p in enumerate(probabilities):
                label = failure_le.inverse_transform([i])[0]
                prob_dict[label] = float(p)

        return {
            "predicted_failure": pred_failure,
            "confidence": float(probabilities[pred_encoded]) if probabilities is not None else None,
            "probabilities": prob_dict or None,
            "status": get_status(pred_failure),
            "input_features": df.to_dict(orient="records")[0],
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# BACKEND-DEPENDENT ENDPOINTS (SAFE)
# ============================================================

def fetch_backend_json(path: str):
    if not BACKEND_URL:
        raise HTTPException(status_code=503, detail="BACKEND_URL not configured")
    try:
        r = requests.get(f"{BACKEND_URL}{path}", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")

@app.get("/predict/latest/{machine_id}")
def predict_latest(machine_id: int):
    sensor = fetch_backend_json(f"/api/machines/latest/{machine_id}")
    return predict(PredictRequest(**sensor))

# ============================================================
# LOAD ANOMALY MODEL
# ============================================================

try:
    anomaly_model = joblib.load("./models/model_anomaly/anomaly_isoforest.pkl")
    anomaly_scaler = joblib.load("./models/model_anomaly/anomaly_scaler.pkl")
    anomaly_type_le = joblib.load("./models/model_anomaly/anomaly_type_le.pkl")
    anomaly_meta = joblib.load("./models/model_anomaly/model_metadata.pkl")
except Exception:
    anomaly_model = None

ANOMALY_FEATURES = FEATURE_ORDER + [
    "air_temp_rolling_mean","air_temp_rolling_std",
    "process_temp_rolling_mean","process_temp_rolling_std",
    "rpm_rolling_mean","rpm_rolling_std",
    "torque_rolling_mean","torque_rolling_std",
    "tool_wear_rolling_mean","tool_wear_rolling_std"
]

def make_features_for_anomaly(d: dict):
    if anomaly_model is None:
        raise HTTPException(status_code=500, detail="Anomaly model not loaded")

    df = make_features_from_input(d)
    for col in ["air_temp","process_temp","rpm","torque","tool_wear"]:
        df[f"{col}_rolling_mean"] = d[col]
        df[f"{col}_rolling_std"] = 0.0
    return df[ANOMALY_FEATURES]

@app.post("/anomaly")
def anomaly(req: AnomalyRequest):
    df = make_features_for_anomaly(req.dict())
    X = anomaly_scaler.transform(df)

    raw = anomaly_model.predict(X)[0]
    score = float(anomaly_model.decision_function(X)[0])

    is_anomaly = bool(raw == -1)

    return {
        "is_anomaly": is_anomaly,
        "score": score,
        "status": "WARNING" if is_anomaly else "NORMAL",
        "metadata": anomaly_meta,
    }

@app.get("/anomaly/latest/{machine_id}")
def anomaly_latest(machine_id: int):
    sensor = fetch_backend_json(f"/api/machines/latest/{machine_id}")
    return anomaly(AnomalyRequest(**sensor))

@app.get("/trend/{machine_id}")
def trend(machine_id: int):
    return fetch_backend_json(f"/api/machines/trend/{machine_id}")

# ============================================================
# HEALTH & ROOT
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "service": "ml-service"}

@app.get("/")
def root():
    return {"message": "ML API is running"}
    