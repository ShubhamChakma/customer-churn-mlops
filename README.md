# 🏦 Customer Churn Prediction API

This project is a machine learning-based service that predicts the likelihood of a customer leaving a bank (churn). It uses a FastAPI backend to serve predictions from a Scikit-Learn model.

### 🚀 Live Demo
**[API Documentation (Swagger UI)](https://churn-prediction-api-zy13.onrender.com/docs)**

---

### 🛠️ Tech Stack
* **Language:** Python 3.12
* **ML Framework:** Scikit-Learn
* **API Framework:** FastAPI
* **Deployment:** Render (CI/CD connected via GitHub)
* **Web Server:** Uvicorn

### 📊 Model Information
The model was trained on a bank customer dataset. Features included:
- Credit Score, Geography, Gender
- Age, Tenure, Balance
- Number of Products, Member Activity, and Salary

### 🔌 API Usage
You can test the API by sending a POST request to the `/predict` endpoint.

### 🚀 Example Input
To test the API manually via the Swagger UI or Postman, use the following JSON structure:
```json
{
  "CreditScore": 600,
  "Age": 40,
  "Tenure": 3,
  "Balance": 60000,
  "NumOfProducts": 2,
  "HasCrCard": 1,
  "IsActiveMember": 1,
  "EstimatedSalary": 50000,
  "Geography_Germany": 0,
  "Geography_Spain": 0,
  "Gender_Male": 1
}
```
👥 Team Members

Anuradha Paswan -https://github.com/anuradhapaswan1
Shubham Chakma -https://github.com/ShubhamChakma
Kriti Yadav -https://github.com/kriti546

🛠️ Tech Stack

Backend: Python, FastAPI
Frontend: Streamlit (Enterprise Design)
Deployment: Render (API)
ML Model: Scikit-Learn (Random Forest/XGBoost)

