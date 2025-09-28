# app_diabetes.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.write("Enter patient data to predict the likelihood of diabetes. 💻📊")

# User input form
with st.form(key='diabetes_form'):
    age = st.number_input("👶 Age", min_value=0, max_value=120, value=30)
    hypertension = st.selectbox("💓 Hypertension (0=No, 1=Yes)", [0,1])
    heart_disease = st.selectbox("❤️ Heart Disease (0=No, 1=Yes)", [0,1])
    bmi = st.number_input("⚖️ BMI", min_value=0.0, max_value=100.0, value=25.0, step=0.1)
    HbA1c_level = st.number_input("🧪 HbA1c Level", min_value=0.0, max_value=20.0, value=5.5, step=0.1)
    blood_glucose_level = st.number_input("🩸 Blood Glucose Level", min_value=0.0, max_value=500.0, value=100.0, step=0.1)

    gender = st.selectbox("🚻 Gender", ["Female", "Male", "Other"])
    smoking_history = st.selectbox("🚬 Smoking History", [
        "No Info", "current", "ever", "former", "never", "not current"
    ])

    submit_button = st.form_submit_button(label='🔍 Predict')

if submit_button:
    # Encode gender
    gender_Female = 1 if gender == "Female" else 0
    gender_Male = 1 if gender == "Male" else 0
    gender_Other = 1 if gender == "Other" else 0

    # Encode smoking_history
    smoking_No_Info = 1 if smoking_history == "No Info" else 0
    smoking_current = 1 if smoking_history == "current" else 0
    smoking_ever = 1 if smoking_history == "ever" else 0
    smoking_former = 1 if smoking_history == "former" else 0
    smoking_never = 1 if smoking_history == "never" else 0
    smoking_not_current = 1 if smoking_history == "not current" else 0

    # Create input dataframe
    input_data = pd.DataFrame([[
        age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level,
        gender_Female, gender_Male, gender_Other,
        smoking_No_Info, smoking_current, smoking_ever, smoking_former, smoking_never, smoking_not_current
    ]], columns=[
        'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level',
        'gender_Female', 'gender_Male', 'gender_Other',
        'smoking_history_No Info', 'smoking_history_current', 'smoking_history_ever',
        'smoking_history_former', 'smoking_history_never', 'smoking_history_not current'
    ])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    prediction_proba = model.predict_proba(input_scaled)[0][1]
    prediction_percent = prediction_proba * 100

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error(f"⚠️ The patient is likely to have diabetes. Probability: {prediction_percent:.2f}%")
        st.info("💡 Recommendation: Consider consulting a healthcare professional for further evaluation.")
    else:
        st.success(f"✅ The patient is unlikely to have diabetes. Probability: {prediction_percent:.2f}%")
        st.info("💡 Recommendation: Maintain a healthy lifestyle and regular check-ups. 🏃‍♂️🥗")
