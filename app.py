import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Age Prediction System",
    page_icon="🧠",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🧠 Age Prediction System")
st.write("Predict whether a respondent belongs to the Adult or Senior age group")

# =========================
# LOAD MODEL & SCALER
# =========================
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder.pkl", "rb"))

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("NHANES_age_prediction.csv")

# Remove target and ID column
feature_columns = [
    col for col in df.columns
    if col not in ["age_group", "SEQN"]
]

# =========================
# FEATURE LABELS
# =========================
feature_labels = {
    "RIDAGEYR": "Respondent's Age",
    "RIAGENDR": "Respondent's Gender",
    "PAQ605": "Weekly Physical Activity",
    "BMXBMI": "Body Mass Index (BMI)",
    "LBXGLU": "Blood Glucose After Fasting",
    "DIQ010": "Diabetic Status",
    "LBXGLT": "Oral Glucose Tolerance",
    "LBXIN": "Blood Insulin Level"
}

st.subheader("Enter Feature Values")

# =========================
# USER INPUTS
# =========================
user_inputs = []

for col in feature_columns:

    label = feature_labels.get(col, col)

    # Gender Dropdown
    if col == "RIAGENDR":

        gender = st.selectbox(
            "Respondent's Gender",
            ["Male", "Female"]
        )

        value = 1 if gender == "Male" else 2

    # Diabetic Status Dropdown
    elif col == "DIQ010":

        diabetic = st.selectbox(
            "Diabetic Status",
            ["No", "Yes"]
        )

        value = 1 if diabetic == "Yes" else 2

    # Physical Activity Dropdown
    elif col == "PAQ605":

        activity = st.selectbox(
            "Weekly Physical Activity",
            ["No", "Yes"]
        )

        value = 1 if activity == "Yes" else 2

    # Numeric Inputs
    else:
        value = st.number_input(label, value=0.0)

    user_inputs.append(value)

# =========================
# PREDICTION BUTTON
# =========================
if st.button("Predict Age Group"):

    input_array = np.array([user_inputs])

    # Scale Input
    input_array_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_array_scaled)

    # Convert numeric prediction back to label
    prediction_label = label_encoder.inverse_transform(prediction)

    # Display Prediction
    st.success(f"Predicted Age Group: {prediction_label[0]}")

    # Confidence Score
    if hasattr(model, "predict_proba"):

        probs = model.predict_proba(input_array_scaled)

        confidence = np.max(probs) * 100

        st.info(f"Confidence: {confidence:.2f}%")