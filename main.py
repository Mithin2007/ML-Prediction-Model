import json
import pickle
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / "model" / "student_model.pkl", "rb") as f:
    model = pickle.load(f)

with open(BASE_DIR / "model" / "metrics.json") as f:
    metrics = json.load(f)

app = FastAPI(title="ML Prediction API")

class PredictionInput(BaseModel):
    hours_studied: float = Field(ge=0, le=24)
    previous_score: float = Field(ge=0, le=100)

@app.get("/")
def home():
    return {
        "message": "Trained ML Prediction API is running",
        "model": "Logistic Regression"
    }

@app.get("/model-info")
def model_info():
    return metrics

@app.post("/predict")
def predict(data: PredictionInput):
    features = [[data.hours_studied, data.previous_score]]
    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return {
        "hours_studied": data.hours_studied,
        "previous_score": data.previous_score,
        "prediction": "Pass" if prediction == 1 else "Needs more study",
        "pass_probability": round(probability, 4),
        "model": "Logistic Regression"
    }
