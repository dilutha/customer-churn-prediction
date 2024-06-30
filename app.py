import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open('churn_model.pkl', 'rb') as model_file, open('scaler.pkl', 'rb') as scaler_file:
    model = pickle.load(model_file)
    scaler = pickle.load(scaler_file)

# Define the Streamlit app
st.title("Customer Churn Prediction")

st.header("Input customer features:")

# Input features
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", [0, 1])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.number_input("Tenure (months)", min_value=0)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
total_charges = st.number_input("Total Charges", min_value=0.0)

# Convert categorical features to numerical values
gender = 1 if gender == "Male" else 0
partner = 1 if partner == "Yes" else 0
dependents = 1 if dependents == "Yes" else 0
phone_service = 1 if phone_service == "Yes" else 0
multiple_lines = 1 if multiple_lines == "Yes" else 2 if multiple_lines == "No phone service" else 0
internet_service = 1 if internet_service == "Fiber optic" else 2 if internet_service == "No" else 0
online_security = 1 if online_security == "Yes" else 2 if online_security == "No internet service" else 0
online_backup = 1 if online_backup == "Yes" else 2 if online_backup == "No internet service" else 0
device_protection = 1 if device_protection == "Yes" else 2 if device_protection == "No internet service" else 0
tech_support = 1 if tech_support == "Yes" else 2 if tech_support == "No internet service" else 0
streaming_tv = 1 if streaming_tv == "Yes" else 2 if streaming_tv == "No internet service" else 0
streaming_movies = 1 if streaming_movies == "Yes" else 2 if streaming_movies == "No internet service" else 0
contract = 1 if contract == "One year" else 2 if contract == "Two year" else 0
paperless_billing = 1 if paperless_billing == "Yes" else 0
payment_method = ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(payment_method)

# Prepare the feature array
features = np.array([[gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service,
                      online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies,
                      contract, paperless_billing, payment_method, monthly_charges, total_charges]])

# Scale the features
features_scaled = scaler.transform(features)

# Predict button
if st.button("Predict"):
    prediction = model.predict(features_scaled)
    churn_prob = model.predict_proba(features_scaled)[0][1]
    if prediction[0] == 1:
        st.subheader(f"The customer is likely to churn with a probability of {churn_prob:.2f}")
    else:
        st.subheader(f"The customer is not likely to churn with a probability of {1-churn_prob:.2f}")
