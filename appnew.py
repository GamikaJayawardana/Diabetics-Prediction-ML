import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🏥",
    layout="wide"
)

# --- 2. Load Models & Assets ---


@st.cache_resource
def load_assets():
    try:
        model = joblib.load('diabetes_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model file: {e}")
        return None


model = load_assets()

# --- 3. Custom CSS for Styling ---
st.markdown("""
<style>
div.stButton > button:first-child {
    background: linear-gradient(to right, #4b6cb7, #182848);
    color: white;
    font-size: 20px;
    font-weight: bold;
    padding: 15px 30px;
    border-radius: 12px;
    border: none;
    box-shadow: 0px 4px 6px rgba(0,0,0,0.2);
    transition: all 0.3s ease;
    width: 100%;
}
div.stButton > button:first-child:hover {
    background: linear-gradient(to right, #182848, #4b6cb7);
    transform: translateY(-2px);
    box-shadow: 0px 6px 8px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# --- 4. Helper Functions ---


def get_radar_chart(input_data, risk_color):
    categories = ['Glucose', 'BloodPressure',
                  'SkinThickness', 'Insulin', 'BMI', 'Age']
    avg_healthy = [110, 70, 27, 80, 30, 27]
    avg_diabetic = [142, 75, 33, 100, 35, 37]

    user_values = [
        input_data['Glucose'][0],
        input_data['BloodPressure'][0],
        input_data['SkinThickness'][0],
        input_data['Insulin'][0],
        input_data['BMI'][0],
        input_data['Age'][0]
    ]

    if risk_color == 'green':
        fill_color = 'rgba(40, 167, 69, 0.4)'
        line_color = '#28a745'
    else:
        fill_color = 'rgba(220, 53, 69, 0.4)'
        line_color = '#dc3545'

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=avg_healthy, theta=categories,
                  fill='toself', name='Avg Healthy', line_color='blue', opacity=0.1))
    fig.add_trace(go.Scatterpolar(r=avg_diabetic, theta=categories,
                  fill='toself', name='Avg Diabetic', line_color='orange', opacity=0.1))
    fig.add_trace(go.Scatterpolar(
        r=user_values, theta=categories, fill='toself', name='Current Patient',
        line=dict(color=line_color, width=3), fillcolor=fill_color
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 200])),
        showlegend=True,
        margin=dict(l=50, r=50, t=30, b=30),
        height=400,
        title=dict(text="Patient Profile vs Averages", x=0.5)
    )
    return fig


# --- 5. Main UI Layout ---
st.title(" Intelligent Diabetes Prediction System")
st.markdown(
    "Enter the patient's clinical data below to assess diabetes risk using **XGBoost AI**.")

st.divider()

# Input Form Area
with st.container():
    st.subheader("Patient Clinical Data")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=1, step=1)
        insulin = st.number_input(
            "Insulin (mu U/ml)", min_value=0, max_value=900, value=80, step=1)

    with c2:
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=300, value=110, step=1)
        bmi = st.number_input(
            "BMI", min_value=0.0, max_value=70.0, value=32.0, step=0.1, format="%.1f")

    with c3:
        bp = st.number_input("Blood Pressure (mm Hg)",
                             min_value=0, max_value=140, value=72, step=1)
        dpf = st.number_input("Diabetes Pedigree Func", min_value=0.000,
                              max_value=3.000, value=0.372, step=0.001, format="%.3f")

    with c4:
        skin = st.number_input("Skin Thickness (mm)",
                               min_value=0, max_value=100, value=25, step=1)
        age = st.number_input("Age (years)", min_value=1,
                              max_value=120, value=29, step=1)

    input_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    input_df = pd.DataFrame([input_data])

st.divider()

# --- 6. Prediction & Results ---
col_results, col_viz = st.columns([1, 1.5])

with col_results:
    st.subheader("Analysis Results")

    if st.button("RUN DIAGNOSIS", use_container_width=True):

        # --- FIX: BYPASS SCALING ---
        # We send the RAW data directly to the model.
        user_data = input_df.values

        # Optional: Replace 0 with NaN for biological metrics
        cols_missing_vals = [1, 2, 3, 4, 5]
        for col in cols_missing_vals:
            if user_data[0, col] == 0:
                user_data[0, col] = np.nan

        # Direct Prediction on RAW Data
        try:
            prediction_proba = model.predict_proba(user_data)
            diabetic_prob = float(prediction_proba[0][1])
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            diabetic_prob = 0.0

        # Determine Status
        if diabetic_prob > 0.5:
            result_color = "red"
            result_header = "🔴 HIGH RISK DETECTED"
            result_msg = "The model indicates a high probability of diabetes."
        else:
            result_color = "green"
            result_header = "🟢 LOW RISK / HEALTHY"
            result_msg = "Great! The model indicates a low probability of diabetes."

        # Save to Session State
        st.session_state['run_analysis'] = True
        st.session_state['user_data'] = user_data  # Save raw data
        st.session_state['diabetic_prob'] = diabetic_prob
        st.session_state['input_df'] = input_df
        st.session_state['result_color'] = result_color
        st.session_state['result_header'] = result_header
        st.session_state['result_msg'] = result_msg

    # Display Results
    if 'run_analysis' in st.session_state and st.session_state['run_analysis']:
        header = st.session_state['result_header']
        msg = st.session_state['result_msg']
        prob = st.session_state['diabetic_prob']
        color = st.session_state['result_color']

        st.markdown(f"<div style='margin-top: 20px;'></div>",
                    unsafe_allow_html=True)

        if color == 'green':
            st.markdown(f"""
            <div style="background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 2px solid #c3e6cb; text-align: center;">
                <h2 style="margin:0;">{header}</h2>
                <p style="margin-top:10px; font-size:18px;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)
            # BALLOONS REMOVED HERE
        else:
            st.markdown(f"""
            <div style="background-color: #f8d7da; color: #721c24; padding: 20px; border-radius: 10px; border: 2px solid #f5c6cb; text-align: center;">
                <h2 style="margin:0;">{header}</h2>
                <p style="margin-top:10px; font-size:18px;">{msg}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"<div style='margin-top: 20px;'></div>",
                    unsafe_allow_html=True)
        st.write(f"**Confidence Score:** {prob*100:.1f}%")
        st.progress(prob)

# --- 7. Visualization Section ---
with col_viz:
    if 'run_analysis' in st.session_state and st.session_state['run_analysis']:

        tab1, tab2 = st.tabs(["📊 Visual Comparison", "🧠 Simple Explanation"])

        with tab1:
            risk_color = st.session_state.get('result_color', 'blue')
            st.plotly_chart(get_radar_chart(
                st.session_state['input_df'], risk_color), use_container_width=True)

        with tab2:
            st.write("### What is driving this result?")
            try:
                # Use RAW data for SHAP
                data_for_shap = st.session_state['user_data']
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(data_for_shap)

                if isinstance(shap_values, list):
                    shap_val_to_plot = shap_values[1][0]
                else:
                    shap_val_to_plot = shap_values[0]

                feature_names = st.session_state['input_df'].columns
                top_indices = np.argsort(np.abs(shap_val_to_plot))[-3:][::-1]

                st.info(
                    "Here are the top factors influencing this specific prediction:")
                for idx in top_indices:
                    impact = shap_val_to_plot[idx]
                    feature = feature_names[idx]
                    val = st.session_state['input_df'].iloc[0][feature]

                    if impact > 0:
                        st.markdown(f"🔴 **{feature}** ({val}): Increases Risk")
                    else:
                        st.markdown(f"🟢 **{feature}** ({val}): Decreases Risk")

                st.write("---")

                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(
                    shap.Explanation(
                        values=shap_val_to_plot,
                        base_values=explainer.expected_value[1] if isinstance(
                            shap_values, list) else explainer.expected_value,
                        data=st.session_state['input_df'].iloc[0],
                        feature_names=feature_names
                    ),
                    max_display=8,
                    show=False
                )
                st.pyplot(fig)

            except Exception as e:
                st.warning(f"Explanation could not be generated: {e}")
  

st.markdown("---")
st.caption("Developed for Intelligent Systems Module")
