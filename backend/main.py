from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import shap
import os

app = FastAPI(title="Diabetes Prediction API")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Determine path to model file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'diabetes_model.pkl')

model = None
explainer = None

@app.on_event("startup")
def load_assets():
    global model, explainer
    try:
        model = joblib.load(MODEL_PATH)
        explainer = shap.TreeExplainer(model)
        print("Model and SHAP Explainer loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

class PatientData(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.post("/predict")
def predict(data: PatientData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    # Create dataframe matching the original order
    input_data = {
        'Pregnancies': [data.Pregnancies],
        'Glucose': [data.Glucose],
        'BloodPressure': [data.BloodPressure],
        'SkinThickness': [data.SkinThickness],
        'Insulin': [data.Insulin],
        'BMI': [data.BMI],
        'DiabetesPedigreeFunction': [data.DiabetesPedigreeFunction],
        'Age': [data.Age]
    }
    input_df = pd.DataFrame(input_data)
    user_data = input_df.values.copy()

    # Original logic: Replace zeros with NaNs for columns 1 through 5
    # (Glucose, BloodPressure, SkinThickness, Insulin, BMI)
    cols_missing_vals = [1, 2, 3, 4, 5]
    for col in cols_missing_vals:
        if user_data[0, col] == 0:
            user_data[0, col] = np.nan
            
    # Predict
    try:
        prediction_proba = model.predict_proba(user_data)
        diabetic_prob = float(prediction_proba[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction Error: {e}")

    # Explain with SHAP
    try:
        shap_values = explainer.shap_values(user_data)
        
        # Depending on XGBoost version/model type, shap_values might be a list or array
        if isinstance(shap_values, list):
            shap_val_to_plot = shap_values[1][0]
            expected_value = explainer.expected_value[1]
        else:
            shap_val_to_plot = shap_values[0]
            expected_value = explainer.expected_value

        # Convert to native python floats for JSON serialization
        shap_values_dict = {
            feature: float(shap_val) 
            for feature, shap_val in zip(input_df.columns, shap_val_to_plot)
        }
        
    except Exception as e:
        print(f"SHAP Error: {e}")
        shap_values_dict = {}
        expected_value = 0.0

    return {
        "probability": diabetic_prob,
        "is_high_risk": diabetic_prob > 0.5,
        "shap_values": shap_values_dict,
        "base_value": float(expected_value) if isinstance(expected_value, (float, np.floating)) else float(expected_value[0])
    }

@app.get("/health")
def health_check():
    return {"status": "ok"}
