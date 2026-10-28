from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "model.joblib"
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURE_NAMES = bundle["feature_names"]

app = FastAPI(
    title="Breast Cancer Classification API",
    description="Educational MLOps demo: predicts a class from Wisconsin Diagnostic Breast Cancer features.",
    version="1.0.0",
)

class PredictionInput(BaseModel):
    # Keep the same feature names and order as the training pipeline.
    mean_radius: float = Field(..., gt=0, description="Mean radius")
    mean_texture: float = Field(..., gt=0, description="Mean texture")
    mean_perimeter: float = Field(..., gt=0, description="Mean perimeter")
    mean_area: float = Field(..., gt=0, description="Mean area")
    mean_smoothness: float = Field(..., gt=0)
    mean_compactness: float = Field(..., gt=0)
    mean_concavity: float = Field(..., ge=0)
    mean_concave_points: float = Field(..., ge=0)
    mean_symmetry: float = Field(..., gt=0)
    mean_fractal_dimension: float = Field(..., gt=0)
    radius_error: float = Field(..., gt=0)
    texture_error: float = Field(..., gt=0)
    perimeter_error: float = Field(..., gt=0)
    area_error: float = Field(..., gt=0)
    smoothness_error: float = Field(..., gt=0)
    compactness_error: float = Field(..., gt=0)
    concavity_error: float = Field(..., ge=0)
    concave_points_error: float = Field(..., ge=0)
    symmetry_error: float = Field(..., gt=0)
    fractal_dimension_error: float = Field(..., gt=0)
    worst_radius: float = Field(..., gt=0)
    worst_texture: float = Field(..., gt=0)
    worst_perimeter: float = Field(..., gt=0)
    worst_area: float = Field(..., gt=0)
    worst_smoothness: float = Field(..., gt=0)
    worst_compactness: float = Field(..., gt=0)
    worst_concavity: float = Field(..., ge=0)
    worst_concave_points: float = Field(..., ge=0)
    worst_symmetry: float = Field(..., gt=0)
    worst_fractal_dimension: float = Field(..., gt=0)

    def to_model_frame(self) -> pd.DataFrame:
        # API-friendly names are mapped to the exact scikit-learn dataset columns.
        values = self.model_dump()
        mapping = {
            "mean_radius": "mean radius",
            "mean_texture": "mean texture",
            "mean_perimeter": "mean perimeter",
            "mean_area": "mean area",
            "mean_smoothness": "mean smoothness",
            "mean_compactness": "mean compactness",
            "mean_concavity": "mean concavity",
            "mean_concave_points": "mean concave points",
            "mean_symmetry": "mean symmetry",
            "mean_fractal_dimension": "mean fractal dimension",
            "radius_error": "radius error",
            "texture_error": "texture error",
            "perimeter_error": "perimeter error",
            "area_error": "area error",
            "smoothness_error": "smoothness error",
            "compactness_error": "compactness error",
            "concavity_error": "concavity error",
            "concave_points_error": "concave points error",
            "symmetry_error": "symmetry error",
            "fractal_dimension_error": "fractal dimension error",
            "worst_radius": "worst radius",
            "worst_texture": "worst texture",
            "worst_perimeter": "worst perimeter",
            "worst_area": "worst area",
            "worst_smoothness": "worst smoothness",
            "worst_compactness": "worst compactness",
            "worst_concavity": "worst concavity",
            "worst_concave_points": "worst concave points",
            "worst_symmetry": "worst symmetry",
            "worst_fractal_dimension": "worst fractal dimension",
        }
        return pd.DataFrame([[values[k] for k in mapping]], columns=[mapping[k] for k in mapping])

@app.get("/")
def root():
    return {"message": "Breast Cancer Classification API", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/features")
def features():
    return {"count": len(FEATURE_NAMES), "features": FEATURE_NAMES}

@app.post("/predict")
def predict(payload: PredictionInput):
    try:
        frame = payload.to_model_frame()
        prediction = int(model.predict(frame)[0])
        probability = model.predict_proba(frame)[0]
        return {
            "prediction": "malignant" if prediction == 1 else "benign",
            "prediction_code": prediction,
            "confidence": round(float(max(probability)), 4),
            "disclaimer": "Educational demonstration only; not a medical diagnosis."
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
