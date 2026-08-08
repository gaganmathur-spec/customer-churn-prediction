import pandas as pd
import joblib


model = joblib.load("models/customer_churn_model.pkl")
scaler = joblib.load("models/scaler.pkl")


# Create New Customer Data
new_customer = pd.DataFrame({
    "gender": [1],
    "SeniorCitizen": [0],
    "Partner": [1],
    "Dependents": [0],
    "tenure": [24],
    "PhoneService": [1],
    "MultipleLines": [0],
    "InternetService": [1],
    "OnlineSecurity": [1],
    "OnlineBackup": [1],
    "DeviceProtection": [0],
    "TechSupport": [1],
    "StreamingTV": [0],
    "StreamingMovies": [0],
    "Contract": [1],
    "PaperlessBilling": [1],
    "PaymentMethod": [2],
    "MonthlyCharges": [75.50],
    "TotalCharges": [1800.00]
})

# Scale the Data (because i use standardScaler during trained the model)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
new_customer = scaler.fit_transform(new_customer)

# Predict
prediction = model.predict(new_customer)
print("prediction:",prediction)

if prediction[0] == 1:
    print("Customer is likely to Churn.")
else:
    print("Customer is likely to Stay.")