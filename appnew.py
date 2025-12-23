import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Set Page Configuration (Must be the first line)
st.set_page_config(
    page_title="Diabetes Risk AI",
    page_icon="🏥",
    layout="wide"
)

# --- 1. Load Models & Assets ---


@st.cache_resource
def load_assets():
    try:
        model = joblib.load('diabetes_model.pkl')
        scaler = joblib.load('scaler.pkl')
        imputer = joblib.load('knn_imputer.pkl')
        return model, scaler, imputer
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None, None


model, scaler, imputer = load_assets()

# --- 2. Helper Functions ---


def get_radar_chart(input_data):
    """
    Creates a Radar Chart comparing User Data vs Average Diabetic vs Average Healthy.
    Note: Averages are approximated from the Pima Indians Diabetes Dataset.
    """
    categories = ['Glucose', 'BloodPressure',
                  'SkinThickness', 'Insulin', 'BMI', 'Age']

    # Average values from the dataset (Approximations)
    avg_healthy = [110, 70, 27, 80, 30, 27]
    avg_diabetic = [142, 75, 33, 100, 35, 37]

    # Extract user values for these specific columns
    user_values = [
        input_data['Glucose'][0],
        input_data['BloodPressure'][0],
        input_data['SkinThickness'][0],
        input_data['Insulin'][0],
        input_data['BMI'][0],
        input_data['Age'][0]
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=avg_healthy, theta=categories, fill='toself', name='Avg Healthy'
    ))
    fig.add_trace(go.Scatterpolar(
        r=avg_diabetic, theta=categories, fill='toself', name='Avg Diabetic'
    ))
    fig.add_trace(go.Scatterpolar(
        r=user_values, theta=categories, fill='toself', name='Your Data',
        line_color='red', opacity=0.8
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 200])),
        showlegend=True,
        title="Your Health Profile vs. Population Averages"
    )
    return fig


# --- 3. Sidebar Inputs ---
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966327.png", width=100)
st.sidebar.title("Patient Data")
st.sidebar.write("Adjust the values below:")


def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 1)
    glucose = st.sidebar.slider('Glucose Level (mg/dL)', 0, 200, 110)
    bp = st.sidebar.slider('Blood Pressure (mm Hg)', 0, 122, 72)
    skin = st.sidebar.slider('Skin Thickness (mm)', 0, 99, 25)
    insulin = st.sidebar.slider('Insulin Level (mu U/ml)', 0, 846, 80)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age (years)', 21, 81, 29)

    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame([data])


input_df = user_input_features()

# --- 4. Main App Layout ---
st.title("🏥 Intelligent Diabetes Prediction System")
st.markdown("""
This application uses **Machine Learning (XGBoost)** to assess the likelihood of diabetes.
It explains *why* a result was predicted and compares your metrics to population averages.
""")

col_main_1, col_main_2 = st.columns([1, 1])

with col_main_1:
    st.subheader("Patient Overview")
    st.dataframe(input_df.T.rename(columns={0: 'Value'}), height=300)

# --- 5. Prediction Logic ---
if st.button("Analyze Risk", type="primary"):

    # 5a. Preprocessing (Must match training exactly)
    # Convert dataframe to numpy array
    user_data = input_df.values

    # Step A: Replace 0s with NaN for specific columns (Glucose, BP, Skin, Insulin, BMI)
    # Column indices in input_df: Glucose(1), BP(2), Skin(3), Insulin(4), BMI(5)
    cols_missing_vals = [1, 2, 3, 4, 5]
    for col in cols_missing_vals:
        if user_data[0, col] == 0:
            user_data[0, col] = np.nan

    # Step B: Scale
    user_data_scaled = scaler.transform(user_data)

    # Step C: Impute
    user_data_imputed = imputer.transform(user_data_scaled)

    # 5b. Prediction
    prediction = model.predict(user_data_imputed)
    prediction_proba = model.predict_proba(user_data_imputed)

    # Get probability of being diabetic (class 1)
    diabetic_prob = prediction_proba[0][1]

    # --- 6. Results Display ---
    with col_main_2:
        st.subheader("Risk Assessment")

        # Gauge Chart / Probability Bar
        if diabetic_prob > 0.5:
            st.error(f"Result: **High Risk of Diabetes**")
            bar_color = "red"
        else:
            st.success(f"Result: **Low Risk of Diabetes**")
            bar_color = "green"

        st.write(f"Probability: **{diabetic_prob * 100:.2f}%**")
        st.progress(diabetic_prob)

        if diabetic_prob > 0.5:
            st.warning(
                "⚠️ Recommendation: Please consult a healthcare professional for a formal diagnosis.")
        else:
            st.info(
                "✅ Recommendation: Maintain a healthy lifestyle and regular checkups.")

    st.divider()

    # --- 7. Advanced Visualization & Explainability ---

    tab1, tab2 = st.tabs(["📊 Comparative Analysis", "🧠 AI Explanation (SHAP)"])

    with tab1:
        # Radar Chart
        st.plotly_chart(get_radar_chart(input_df), use_container_width=True)
        st.caption(
            "This chart compares your values (Red) against the average Healthy person (Blue) and Diabetic person (Orange).")

    with tab2:
        st.write("How did the model reach this conclusion?")

        # Calculate SHAP values
        # Note: We use the *imputed* data for explanation because that's what the model sees
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(user_data_imputed)

            # Plot
            fig, ax = plt.subplots(figsize=(10, 5))
            # Force plot or Waterfall plot
            # Since shap_values for classification might be a list or array depending on XGBoost version:
            if isinstance(shap_values, list):
                # Binary classification usually returns a list of two arrays
                shap_val_to_plot = shap_values[1][0]
                base_value = explainer.expected_value[1]
            else:
                shap_val_to_plot = shap_values[0]
                base_value = explainer.expected_value

            # Create a Waterfall plot
            shap.plots.waterfall(
                shap.Explanation(values=shap_val_to_plot,
                                 base_values=base_value,
                                 data=input_df.iloc[0],
                                 feature_names=input_df.columns)
            )
            st.pyplot(bbox_inches='tight')
            st.caption(
                "Red bars increase the risk, Blue bars decrease the risk.")

        except Exception as e:
            st.warning(f"Could not generate SHAP plot: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("*Disclaimer: This tool is for educational purposes only and not a substitute for professional medical advice.*")
