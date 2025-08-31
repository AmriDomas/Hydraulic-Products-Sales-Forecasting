from fastapi import FastAPI
from pydantic import BaseModel
import sys
import os
from pathlib import Path

# Tambahkan path ke sys.path
current_dir = Path(__file__).parent
app_dir = current_dir.parent
sys.path.append(str(app_dir))

from utils import load_model, build_features

app = FastAPI()
model = load_model()

class InputData(BaseModel):
    variant: str
    date: str
    marketing_spend: float
    discount_percent: float
    competitor_activity: float
    economic_indicator: float
    seasonality_index: float

@app.post("/predict")
def predict_api(data: InputData):
    features = build_features(data.dict())
    pred = model.predict([features])[0]
    return {"prediction": int(pred)}

@app.get("/")
def read_root():
    return {"message": "Hydraulic Products Sales Forecasting API"}