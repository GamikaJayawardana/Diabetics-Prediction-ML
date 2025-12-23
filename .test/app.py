import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Load the saved model and processors
model = joblib.load('diabetes_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('knn_imputer.pkl')

# 2. App Title and Description
st.title("🏥 Diabetes Prediction App")
st.write("Enter the patient details below to predict the likelihood of diabetes.")

# 3. Create Input Fields for User
# We use columns to make the UI look better
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)
    glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)
    bp = st.number_input("Blood Pressure", min_value=0, max_value=140, step=1)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)

with col2:
    insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, step=0.1)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, step=0.001)
    age = st.number_input("Age", min_value=0, max_value=120, step=1)

# 4. Prediction Logic
if st.button("Predict Result"):
    # Create a numpy array with the inputs
    # IMPORTANT: The order must be exactly the same as your training data
    user_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    
    # --- Preprocessing Steps (Must match your training exactly) ---
    
    # Step A: Replace 0s with NaN for the specific columns (Glucose, BP, Skin, Insulin, BMI)
    # Indices: Glucose(1), BP(2), Skin(3), Insulin(4), BMI(5)
    cols_missing_vals = [1, 2, 3, 4, 5]
    for col in cols_missing_vals:
        if user_data[0, col] == 0:
            user_data[0, col] = np.nan
            
    # Step B: Scale the data
    user_data_scaled = scaler.transform(user_data)
    
    # Step C: Impute missing values
    # Note: KNN Imputer usually needs neighbors. For a single prediction, 
    # it might act like a simple filler or require the original dataset reference.
    # However, since we saved the fitted imputer, it will try its best.
    user_data_imputed = imputer.transform(user_data_scaled)
    
    # Step D: Predict
    prediction = model.predict(user_data_imputed)
    probability = model.predict_proba(user_data_imputed)[0][1]

    # 5. Display Result
    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"🔴 Result: **Diabetic**")
        st.write(f"Confidence Level: {probability * 100:.2f}%")
        st.warning("Please consult a doctor for a medical checkup.")
    else:
        st.success(f"🟢 Result: **Non-Diabetic**")
        st.write(f"Confidence Level: {(1-probability) * 100:.2f}%")