import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

app = FastAPI(title="Credit Card Fraud Detection API",
              description="Predict fraud using trained model",
              version="1.0")

class Transaction(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predict fraud (0 = Not Fraud, 1 = Fraud) using the trained model.
    """
    try:
        data = np.array(transaction.features).reshape(1, -1)
        data[:, [0, -1]] = scaler.transform(data[:, [0, -1]])

        proba = model.predict_proba(data)[0][1]
        prediction = int(proba>=0.3)

        return {
            "fraud_probability": float(proba),
            "prediction": prediction
        }
    except Exception as e:
        return {"error": str(e)}
