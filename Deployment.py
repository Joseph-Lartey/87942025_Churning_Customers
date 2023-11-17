# Importing the Libraries
import pickle
import pandas as pd
import streamlit as st
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from sklearn.metrics import roc_auc_score

# Load the trained model
final_model = "finalmodel.h5"
best_model = keras.models.load_model(final_model)

# Load the scaler
scaler_path = "scaler.pkl"
with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Load AUC score
file_path = r"C:\Users\Joseph Lartey\OneDrive - Ashesi University\Desktop\Joseph\Course materials\Year 2\Sem 2\Intro to AI\codes\auc_score.txt"

with open(file_path, "r") as file:
    auc_score = float(file.read())

# Title of the web app with "by Joseph" in the corner
st.title("Churn Prediction Web App")

# Display AUC score at the top right corner
st.markdown("<h1 style='text-align: right; font-weight: bold; color: #FFFFFF;'>AUC Score: {:.4f}</h1>".format(auc_score), unsafe_allow_html=True)

# Arrange the input fields side by side
col1, col2, col3 = st.columns(3)
# User input fields in the first column
with col1:
    monthly_charges = st.number_input('Monthly Charges', min_value=0.0, max_value=500.0, step=100.0)
    total_charges = st.number_input('Total Charges', min_value=0.0, max_value=8684.8, step=10.0)
    tenure = st.number_input('Tenure', min_value=0.0, max_value=100.0, step=10.0)
    online_security_no = st.selectbox("Online Security", ["No", "Yes"])

# User input fields in the second column
with col2:
    gender = st.selectbox('Gender', ['Female', 'Male'])
    senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
    partner = st.selectbox('Partner', ['No', 'Yes'])
    tech_support_no = st.selectbox("Tech Support", ["No", "Yes"])

# User input fields in the third column
with col3:
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two years"])
    paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
    internet_service_fiber_optic = st.selectbox("Internet Service (Fiber Optic)", ["No", "Yes"])
    payment_method_electronic_check = st.selectbox("Payment Method (Electronic Check)", ["No", "Yes"])

# Button to trigger the prediction
predict_button = st.button("Predict")
clear_button = st.button("Clear")

if predict_button:
    # Convert categorical features to numerical values
    gender = 1 if gender == "Male" else 0
    senior_citizen = 1 if senior_citizen == "Yes" else 0
    partner = 1 if partner == "Yes" else 0
    contract_mapping = {"Month-to-month": 0, "One year": 1, "Two years": 2}
    contract = contract_mapping[contract]
    paperless_billing = 1 if paperless_billing == "Yes" else 0
    internet_service_fiber_optic = 1 if internet_service_fiber_optic == "Yes" else 0
    online_security_no = 1 if online_security_no == "Yes" else 0
    tech_support_no = 1 if tech_support_no == "Yes" else 0
    payment_method_electronic_check = 1 if payment_method_electronic_check == "Yes" else 0

    # Create a DataFrame with the user input
    user_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [0],  # This field was missing in the original code
        'tenure': [tenure],
        'OnlineSecurity': [online_security_no],
        'OnlineBackup': [0],  # Add default values for missing fields
        'DeviceProtection': [0],
        'TechSupport': [tech_support_no],
        'StreamingTV': [0],
        'StreamingMovies': [0],
        'PaperlessBilling': [paperless_billing],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges],
        'Contract_One year': [1 if contract == 1 else 0],
        'Contract_Two year': [1 if contract == 2 else 0],
        'MultipleLines_Yes': [0],
        'InternetService_Fiber optic': [internet_service_fiber_optic],
        'PaymentMethod_Credit card (automatic)': [0],
        'PaymentMethod_Electronic check': [payment_method_electronic_check],
        'PaymentMethod_Mailed check': [0]
    })

    # Scale the numerical features using the loaded scaler
    scaled_data = scaler.transform(user_data)

    # Make predictions using the loaded model
    prediction = best_model.predict(scaled_data)

    # Calculate confidence rate
    confidence_rate = round(max(prediction[0, 0], 1 - prediction[0, 0]) * 100, 2)

    #Displaying the prediction and the confidence
    confidence_rate = round(max(prediction[0, 0], 1 - prediction[0, 0]) * 100, 2)
    st.markdown("<h1 style='text-align: left; font-weight: bold; color: #FFFFFF;'>Prediction: {}</h1>".format("Churn" if prediction[0][0] > 0.5 else "No Churn"), unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left; font-weight: bold; color: #FFFFFF;'>Confidence Level: {}%</h1>".format(confidence_rate), unsafe_allow_html=True)
    
