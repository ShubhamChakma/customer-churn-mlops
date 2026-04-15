from fastapi import FastAPI
import joblib
import numpy as np
import mlflow
import os

app = FastAPI()

# ✅ Get correct base directory (fixes path issues everywhere)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ Load model + scaler + columns safely
model = joblib.load(os.path.join(BASE_DIR, "model/churn_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model/scaler.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "model/columns.pkl"))
print(columns)

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

@app.post("/predict")
def predict(data: dict):
    try:
        # ✅ Ensure correct feature order using columns.pkl
        features = np.array([data[col] for col in columns]).reshape(1, -1)

        # ✅ Scale input
        features = scaler.transform(features)

        # ✅ MLflow tracking
        with mlflow.start_run():
            prediction = model.predict(features)[0]

            mlflow.log_param("input_data", str(data))
            mlflow.log_metric("prediction", int(prediction))

        return {"churn": int(prediction)}

    except KeyError as e:
        return {"error": f"Missing feature: {str(e)}"}

    except Exception as e:
        return {"error": str(e)}