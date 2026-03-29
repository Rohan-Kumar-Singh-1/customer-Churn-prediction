import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("models/model.keras")
preprocessor = joblib.load("models/preprocessor.pkl")

THRESHOLD = 0.6


class ChurnInput(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running 🚀"}


@app.post("/predict")
def predict(data: ChurnInput):
    df = pd.DataFrame([data.dict()])

    X = preprocessor.transform(df)

    prob = float(model.predict(X)[0][0])
    prediction = int(prob > THRESHOLD)

    # 🔥 ADD THIS
    risk = "High Risk" if prob > 0.5 else "Low Risk"

    return {
        "probability": float(prob),
        "prediction": prediction,
        "label": "Churn" if prediction == 1 else "No Churn",
        "risk_level": risk   # 🔥 ADD THIS TO RESPONSE
    }