from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = FastAPI(title="Hospital Resource Optimiser API")

MODEL_PATH = "model.joblib"

# ------------------------
# Data Models
# ------------------------
class Metrics(BaseModel):
    date: str
    bed_usage: int
    oxygen_usage: int
    staff_on_duty: int

# ------------------------
# Storage (in-memory for demo, replace with DB)
# ------------------------
historical_data = pd.DataFrame(columns=["date", "bed_usage", "oxygen_usage", "staff_on_duty"])

# ------------------------
# API Endpoints
# ------------------------
@app.post("/ingest")
def ingest(metrics: Metrics):
    global historical_data
    new_entry = pd.DataFrame([metrics.dict()])
    historical_data = pd.concat([historical_data, new_entry], ignore_index=True)
    return {"status": "ok", "rows": len(historical_data)}

@app.get("/historical")
def get_historical():
    return historical_data.to_dict(orient="records")

@app.get("/predict")
def predict(days: int = 7):
    if historical_data.empty:
        raise HTTPException(status_code=400, detail="No data available")

    preds = {}
    for col in ["bed_usage", "oxygen_usage", "staff_on_duty"]:
        df = historical_data.reset_index()
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[col].astype(float).values
        model = LinearRegression().fit(X, y)
        future_idx = np.arange(len(df), len(df) + days).reshape(-1, 1)
        pred_vals = model.predict(future_idx)
        preds[col] = pred_vals.tolist()

        # save model (demo: only bed usage)
        if col == "bed_usage":
            joblib.dump(model, MODEL_PATH)

    return {"forecast_days": days, "predictions": preds}
