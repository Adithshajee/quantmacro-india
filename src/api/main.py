from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

app = FastAPI(title="BSE Macro-Sector Analyzer API")

model_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "rf_model.pkl")
features_path = os.path.join(os.path.dirname(__file__), "..", "..", "models", "features.pkl")

model = None
features = None

class PredictRequest(BaseModel):
    lag1_return: float
    weighted_sentiment: float
    sector_index: str

class PredictResponse(BaseModel):
    is_green: int
    probability: float

@app.on_event("startup")
def load_assets():
    global model, features
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        features = joblib.load(features_path)

@app.get("/health")
def health_check():
    status = "healthy" if model is not None else "model_missing"
    return {"status": status}

@app.post("/reload")
def reload_model():
    load_assets()
    if model is not None:
        return {"status": "success", "message": "Model reloaded."}
    else:
        raise HTTPException(status_code=500, detail="Model reload failed.")

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if model is None or features is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train the model first.")

    input_data = {
        "lag1_return": [req.lag1_return],
        "weighted_sentiment": [req.weighted_sentiment]
    }
    df = pd.DataFrame(input_data)
    
    # Add dummies for sector
    # All dummies are 0 except the one that matches req.sector_index
    for f in features:
        if f.startswith("sector_index_"):
            if f.replace("sector_index_", "") == req.sector_index:
                df[f] = [1]
            else:
                df[f] = [0]
                
    # Ensure any missing features from `features.pkl` are present
    for f in features:
        if f not in df.columns:
            df[f] = [0]
            
    df = df[features] # order properly
    
    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1] # prob of class 1
    
    return PredictResponse(is_green=int(pred), probability=float(prob))
