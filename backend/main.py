"""FastAPI backend for the React UI.

Serves the model already trained by `python -m src.train` (models/final_model.joblib)
and the result files it wrote to results/. Does NOT retrain anything.

Run from the project root:
    pip install -r backend/requirements.txt
    uvicorn backend.main:app --reload --port 8000
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.config import FEATURES, MODEL_DIR, RESULTS_DIR, FIG_DIR

app = FastAPI(title="Diabetes Risk Prediction API")

# Vite's default dev server port. Add your deployed frontend URL here too, once you have one.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        path = MODEL_DIR / "final_model.joblib"
        if not path.exists():
            raise HTTPException(500, "No trained model found. Run `python -m src.train` first.")
        _bundle = joblib.load(path)
    return _bundle


class PredictRequest(BaseModel):
    Pregnancies: float = Field(ge=0, le=20)
    Glucose: float = Field(ge=0, le=300)
    BloodPressure: float = Field(ge=0, le=200)
    SkinThickness: float = Field(ge=0, le=100)
    Insulin: float = Field(ge=0, le=900)
    BMI: float = Field(ge=0, le=80)
    DiabetesPedigreeFunction: float = Field(ge=0, le=3)
    Age: float = Field(ge=1, le=120)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_trained": (MODEL_DIR / "final_model.joblib").exists()}


@app.get("/api/metadata")
def metadata():
    b = get_bundle()
    return {"features": FEATURES, "model_name": b["name"], "threshold": b["threshold"]}


@app.post("/api/predict")
def predict(req: PredictRequest):
    b = get_bundle()
    row = pd.DataFrame([[getattr(req, f) for f in FEATURES]], columns=FEATURES)
    proba = float(b["model"].predict_proba(row)[0, 1])
    label = "diabetic" if proba >= b["threshold"] else "non_diabetic"
    return {"probability": proba, "threshold": b["threshold"], "predicted_class": label,
            "model_name": b["name"]}


def _read_csv(name):
    path = RESULTS_DIR / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found. Run `python -m src.train` first.")
    return json.loads(pd.read_csv(path).to_json(orient="records"))


@app.get("/api/comparison")
def comparison():
    return _read_csv("model_comparison.csv")


@app.get("/api/importance")
def importance():
    return _read_csv("permutation_importance.csv")


@app.get("/api/leakage-demo")
def leakage_demo():
    return _read_csv("preprocessing_ablation.csv") if not (RESULTS_DIR / "leakage_demonstration.csv").exists() \
        else _read_csv("leakage_demonstration.csv")


@app.get("/api/final-metrics")
def final_metrics():
    path = RESULTS_DIR / "final_metrics.json"
    if not path.exists():
        raise HTTPException(404, "final_metrics.json not found. Run `python -m src.train` first.")
    return json.loads(path.read_text())


@app.get("/api/data-quality")
def data_quality():
    path = RESULTS_DIR / "data_quality_report.json"
    if not path.exists():
        raise HTTPException(404, "data_quality_report.json not found. Run `python -m src.train` first.")
    return json.loads(path.read_text())


@app.get("/api/figure/{name}")
def figure(name: str):
    path = FIG_DIR / name
    if ".." in name or "/" in name or not path.exists():
        raise HTTPException(404, "Figure not found.")
    return FileResponse(path)
