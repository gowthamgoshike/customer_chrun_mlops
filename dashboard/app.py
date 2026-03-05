import streamlit as st
import requests

# 1. Page Configuration
st.set_page_config(page_title="Telco Churn Predictor", page_icon="📊", layout="wide")

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Enter the customer's profile below to generate a real-time churn risk assessment.")
st.markdown("---")
st.sidebar.header("Customer Information")

st.metric("Model", "Random Forest")
# 2. UI Layout using Columns
col1, col2, col3 = st.columns(3)

with col1:
    st.header("👤 Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])

with col2:
    st.header("🌐 Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col3:
    st.header("💳 Account & Billing")
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=600.0)

st.markdown("---")

# 3. Prediction Execution
# use_container_width makes the button span the whole screen nicely
if st.button("🚀 Predict Churn Risk", use_container_width=True):
    
    # Package all inputs into the dictionary our API expects
    data = {
        "customerID": "0000-TEST", # Required by schema, but dropped by the API
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": "No", # Simplified for UI
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": "No",  # Simplified for UI
        "DeviceProtection": "No", # Simplified for UI
        "TechSupport": tech_support,
        "StreamingTV": "No", # Simplified for UI
        "StreamingMovies": "No", # Simplified for UI
        "Contract": contract,
        "PaperlessBilling": "Yes", # Simplified for UI
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Show a loading spinner while waiting for the API
    with st.spinner("Analyzing customer profile..."):
        try:
            # Send the request to your Docker container running on port 8000
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json=data
            )
            
            # Check if the API request was successful
            if response.status_code == 200:
                # Extract the prediction from the JSON response
                prediction = response.json()["prediction"]
                
                # Display dynamic, colored alerts based on the result
                if prediction == 1:
                    st.error("⚠️ **High Risk:** This customer is likely to churn. Recommend immediate retention action.")
                else:
                    st.success("✅ **Safe:** This customer is likely to stay.")
            else:
                st.warning(f"API Error: {response.status_code} - Something went wrong on the backend.")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Could not connect to the API. Make sure your Docker container is running!")