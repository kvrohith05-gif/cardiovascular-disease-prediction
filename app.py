
import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("heart_disease_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

# Application heading
st.title("❤️ Cardiovascular Disease Prediction")

st.write(
    "Enter the health-related information below "
    "to obtain an educational machine learning prediction."
)

st.info(
    "This application is for academic purposes only. "
    "It is not a medical diagnostic tool."
)

st.subheader("Patient Information")

# Input fields
age = st.number_input(
    "Age",
    min_value=1,
    max_value=120,
    value=45
)

sex = st.selectbox(
    "Gender",
    options=[0, 1],
    format_func=lambda x: "Female" if x == 0 else "Male"
)

cp = st.selectbox(
    "Chest Pain Type",
    options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-Anginal Pain",
        4: "Asymptomatic"
    }[x]
)

trestbps = st.number_input(
    "Resting Blood Pressure",
    min_value=50,
    max_value=300,
    value=120
)

chol = st.number_input(
    "Cholesterol",
    min_value=50,
    max_value=700,
    value=200
)

fbs = st.selectbox(
    "Fasting Blood Sugar > 120 mg/dl",
    options=[0, 1]
)

restecg = st.selectbox(
    "Resting ECG",
    options=[0, 1, 2]
)

thalach = st.number_input(
    "Maximum Heart Rate",
    min_value=40,
    max_value=250,
    value=150
)

exang = st.selectbox(
    "Exercise Induced Angina",
    options=[0, 1]
)

oldpeak = st.number_input(
    "ST Depression (Oldpeak)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1
)

slope = st.selectbox(
    "Slope of ST Segment",
    options=[1, 2, 3]
)

ca = st.selectbox(
    "Number of Major Vessels",
    options=[0, 1, 2, 3]
)

thal = st.selectbox(
    "Thalassemia",
    options=[3, 6, 7]
)

# Prediction button
if st.button("Predict Heart Disease"):

    # Prepare input data
    input_data = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }])

    # Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.warning(
            "The model predicts the heart disease class (1)."
        )
    else:
        st.success(
            "The model predicts the no heart disease class (0)."
        )

    st.caption(
        "This is an experimental model output, not a diagnosis. "
        "A healthcare professional must assess any health concerns."
    )
