import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Load best model (we assume RandomForest or LinearRegression is best)
MODEL_PATH = "models/RandomForest.pkl"  # will still work even if LR was best

try:
    model = joblib.load(MODEL_PATH)
    model_loaded = True
except:
    model = None
    model_loaded = False

app = FastAPI()


# Input validation schema
class InputData(BaseModel):
    controls_count: int = Field(..., ge=10, le=200)
    evidence_items: int = Field(..., ge=20, le=500)
    auditor_experience: int = Field(..., ge=1, le=20)
    is_regulatory: int = Field(..., ge=0, le=1)


# Health endpoint
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model_loaded}


# Prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    if not model_loaded:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        features = np.array([[
            data.controls_count,
            data.evidence_items,
            data.auditor_experience,
            data.is_regulatory
        ]])

        prediction = model.predict(features)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))