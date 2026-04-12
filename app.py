from fastapi import FastAPI
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("churn_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # Convert input to array
        features = np.array(list(data.values())).reshape(1, -1)

        # Scale input
        features = scaler.transform(features)

        # Predict
        prediction = model.predict(features)[0]

        return {"churn": int(prediction)}

    except Exception as e:
        return {"error": str(e)}
