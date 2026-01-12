import streamlit as st
import joblib
import pandas as pd

try:
    model = joblib.load('model.pkl')
except:
    st.error("Model missing. Please run train_model.py.")
    st.stop()

st.set_page_config(page_title="Liver Risk Predictor", layout="wide")

st.title("MLRP: Advanced Clinical Prediction")
st.markdown(f"**System Status:** Online | **Model Trained on:** 200,000 Patient Records")

# INPUT
tab1, tab2, tab3 = st.tabs(["👤 Patient Profile", "🫀 Vitals & Obs", "🧪 Lab Results"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 90, 45)
        gender_txt = st.radio("Gender", ["Female", "Male"], horizontal=True)
        gender = 1 if gender_txt == "Male" else 0
    with col2:
        history_txt = st.radio("Family History of Illness?", ["No", "Yes"], horizontal=True)
        family_history = 1 if history_txt == "Yes" else 0
        smoking_txt = st.radio("Smoking Status", ["Non-Smoker", "Smoker"], horizontal=True)
        smoking = 1 if smoking_txt == "Smoker" else 0

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        systolic = st.number_input("Systolic BP", 90, 220, 120)
        diastolic = st.number_input("Diastolic BP", 50, 130, 80)
    with col2:
        heart_rate = st.number_input("Heart Rate", 40, 150, 72)
        spo2 = st.number_input("SpO2 (%)", 70, 100, 98)
    with col3:
        bmi = st.number_input("BMI", 15.0, 50.0, 24.5)
        temp = st.number_input("Temperature (°F)", 95.0, 106.0, 98.6)

with tab3:
    col1, col2, col3 = st.columns(3)
    with col1:
        glucose = st.number_input("Glucose (mg/dL)", 50, 400, 95)
        cholesterol = st.number_input("Cholesterol", 100, 400, 180)
    with col2:
        wbc = st.number_input("WBC Count (k)", 1.0, 30.0, 6.5)
        platelets = st.number_input("Platelets (k)", 10, 800, 250)
    with col3:
        albumin = st.number_input("Albumin (g/dL)", 1.0, 6.0, 4.2)

# PREDICTION
input_data = pd.DataFrame([[
    age, gender, family_history, bmi, systolic, diastolic, 
    heart_rate, spo2, temp, glucose, albumin, 
    cholesterol, wbc, platelets, smoking
]], columns=[
    'Age', 'Gender', 'Family_History', 'BMI', 'Systolic_BP', 'Diastolic_BP', 
    'Heart_Rate', 'SpO2', 'Temperature', 'Glucose', 'Albumin', 
    'Cholesterol', 'WBC', 'Platelets', 'Smoking'
])

# Predict
probability = model.predict_proba(input_data)[0][1]

st.divider()

c1, c2 = st.columns([1, 2])

with c1:
    st.metric(label="Risk Probability", value=f"{probability:.1%}")

with c2:
    if probability < 0.3:
        st.success("✅ **LOW RISK**\n\nPatient is healthy. Routine checkup recommended annually.")
    elif probability < 0.7:
        st.warning("⚠️ **MODERATE RISK**\n\nPatient shows early warning signs. Review Vitals and Labs.")
    else:
        st.error("🚨 **HIGH RISK**\n\nClinical indicators suggest immediate attention is required.")