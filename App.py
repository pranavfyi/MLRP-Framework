import streamlit as st
import numpy as np
import joblib


try:
    model = joblib.load("mlrp_model_v1.pkl")
except:
    st.error("Model file not found. Please ensure 'mlrp_model_v1.pkl' is in the directory.")


st.set_page_config(page_title="MLRP – Metabolic Liver Risk Predictor", layout="wide")


with st.sidebar:
    st.title("About MLRP")
    st.info(
        "**MLRP (Metabolic Liver Risk Predictor)** is an open-source clinical decision support tool "
        "designed to identify risk factors for Metabolic Dysfunction-Associated Steatotic Liver Disease (MASLD)."
    )

    st.subheader("⚠️ Medical Disclaimer")
    st.warning(
        "This tool is for **informational and educational purposes only**. It is not a substitute for "
        "professional medical advice, diagnosis, or treatment. Always seek the advice of your physician "
        "or other qualified health provider with any questions you may have regarding a medical condition."
    )


st.title("🧬 MLRP: Clinical Risk Stratification")
st.markdown("### *Metabolic Liver Risk Predictor – Version 1.0*")

st.write(
    "This framework estimates the probability of MASLD risk using routine clinical and metabolic parameters. "
    "It acts as a primary screening layer to identify patients who may require further clinical investigation."
)

st.markdown("---")


col1, col2 = st.columns(2)

with col1:
    st.header("👤 Patient Demographics")
    age = st.number_input("Age", min_value=18, max_value=100, value=45)
    sex = st.selectbox("Sex", ["Male", "Female"])
    bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=27.0)
    waist = st.number_input("Waist Circumference (cm)", min_value=50.0, max_value=150.0, value=90.0)

with col2:
    st.header("🧪 Metabolic Markers")
    fbs = st.number_input("Fasting Blood Sugar (mg/dL)", value=110.0)
    trig = st.number_input("Triglycerides (mg/dL)", value=180.0)
    hba1c = st.number_input("HbA1c (%)", value=6.1)
    insulin = st.number_input("Insulin (µIU/mL)", value=15.0)

st.header("🩺 Liver Enzymes & Lipid Profile")
c1, c2, c3 = st.columns(3)
with c1:
    ggt = st.number_input("GGT (U/L)", value=55.0)
    chol = st.number_input("Total Cholesterol (mg/dL)", value=200.0)
with c2:
    ast = st.number_input("AST / SGOT (U/L)", value=40.0)
    hdl = st.number_input("HDL (mg/dL)", value=45.0)
with c3:
    alt = st.number_input("ALT / SGPT (U/L)", value=50.0)
    ldl = st.number_input("LDL (mg/dL)", value=120.0)

# Data Preparation
sex_val = 1 if sex == "Male" else 0
input_data = np.array([[
    age, sex_val, bmi, waist, fbs, trig, chol, hdl, ldl,
    ggt, ast, alt, hba1c, insulin
]])


if st.button("Calculate Risk Probability"):
    try:

        prob = model.predict_proba(input_data)[0][1]

        if prob < 0.3:
            risk = "Low Risk"
            color = "#28a745"
        elif prob < 0.6:
            risk = "Moderate Risk"
            color = "#fd7e14"
        else:
            risk = "High Risk"
            color = "#dc3545"

        st.markdown("---")
        st.subheader("Assessment Results")

        m1, m2 = st.columns(2)
        m1.metric(label="Risk Probability", value=f"{prob * 100:.2f}%")

        st.markdown(
            f"<div style='padding:20px; border-radius:10px; background-color:{color}; color:white; text-align:center;'>"
            f"<h2>Category: {risk}</h2>"
            f"</div>",
            unsafe_allow_html=True
        )

        if risk != "Low Risk":
            st.error(
                "**Recommendation:** Based on these parameters, a clinical consultation and further "
                "diagnostic imaging (such as specialized ultrasound) are highly advised."
            )
    except Exception as e:
        st.error(f"Prediction Error: {e}")