import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("loan_approval_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Loan Approval Predictor")
st.markdown("Enter applicant details to check loan approval status")

# Input fields
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount (in 1000s)", min_value=0)
Loan_Amount_Term = st.number_input("Loan Term (in days)", min_value=0)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert to numerical
input_data = [
    1 if Gender == "Male" else 0,
    1 if Married == "Yes" else 0,
    int(Dependents.replace("3+", "3")),
    1 if Education == "Graduate" else 0,
    1 if Self_Employed == "Yes" else 0,
    ApplicantIncome,
    CoapplicantIncome,
    LoanAmount,
    Loan_Amount_Term,
    Credit_History,
    {"Urban": 2, "Semiurban": 1, "Rural": 0}[Property_Area]
]

input_array = np.array([input_data])
scaled_input = scaler.transform(input_array)

if st.button("Predict Loan Approval"):
    prediction = model.predict(scaled_input)[0]
    result = "Approved" if prediction == 1 else " Not Approved"
    st.subheader(f"Loan Status: {result}")
