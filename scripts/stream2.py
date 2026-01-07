
import streamlit as st
import pickle
import pandas as pd

# -------------------------------
# Load trained model
# -------------------------------
model = pickle.load(open(r"C:\Users\yashr\OneDrive\Desktop\loan approval\ab_best_model.pkl",   'rb'))

st.title("🏦      Loan Approval Prediction")

st.write("Enter applicant details:")

# -------------------------------
# User Inputs (all features)
# -------------------------------
Gender = st.selectbox("Gender \n  Encode:   Male/Female as 0/1", [0, 1])          # Encode Male/Female as 0/1
Married = st.selectbox("Married \n  Encode: 0=No, 1=Yes", [0, 1])       # 0=No, 1=Yes
Dependents = st.number_input("Dependents", min_value=0)
Education = st.selectbox("Education   Encode:   0=Not Graduate, 1=Graduate", [0, 1])   # 0=Not Graduate, 1=Graduate
Self_Employed = st.selectbox("Self Employed   Encode:   0=Unemployed, 1=Employed", [0, 1])
LoanAmount = st.number_input("Loan Amount", min_value=0.0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [0, 1])
Property_Area = st.selectbox("Property Area", [0, 1, 2])
Income = st.number_input("Applicant Income", min_value=0.0)

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict"):

  input_data = {
    'Gender': Gender,
    'Married': Married,
    'Dependents': Dependents,
    'Education': Education,
    'Self_Employed': Self_Employed,
    'LoanAmount': LoanAmount,
    'Loan_Amount_Term': Loan_Amount_Term,
    'Credit_History': Credit_History,
    'Property_Area': Property_Area,
    'Income': Income }

  input_df = pd.DataFrame([input_data])
  prediction = model.predict(input_df)[0]


  if prediction == 1:
        
        st.success("Loan Approved 🎉✅")
  else:
        st.error("❌ Loan Not Approved")





