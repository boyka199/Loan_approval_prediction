# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

# Load the trained model, scaler, features, and training data
model = joblib.load("model.pkl")  # Trained Logistic Regression model
scaler = joblib.load("scaler.pkl")  # Scaler used during training
features = joblib.load("features.pkl")  # List of model features
df = joblib.load("training_data.pkl")  # Full training dataframe

# Title of the Streamlit app
st.title("🏦 Loan Approval Predictor")

# Header for input section
st.header("Loan Application Details")

# Input widgets for user data
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", value=5000)
CoapplicantIncome = st.number_input("Coapplicant Income", value=0.0)
LoanAmount = st.number_input("Loan Amount (in thousands)", value=150)
Loan_Amount_Term = st.number_input("Loan Term (in days)", value=360)
Credit_History = st.selectbox("Credit History", [1.0, 0.0])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Check for invalid loan term
if Loan_Amount_Term <= 50:
    st.warning("❗ Loan Term must be greater than 50 days. Please enter a valid loan term.")
elif LoanAmount <= 50:
    st.warning(" Loan Amount should not be less than 50,000")
elif ApplicantIncome <= 5000:
    st.warning(" Income should not be less than 5000")
elif CoapplicantIncome <=3000:
    st.warning("CoapplicantIncome should not be less than 3000")
else:
    # Prepare user input for prediction
    user_input = {
        "ApplicantIncome": ApplicantIncome,
        "CoapplicantIncome": CoapplicantIncome,
        "LoanAmount": LoanAmount,
        "Loan_Amount_Term": Loan_Amount_Term,
        "Credit_History": Credit_History,
        "Dependents": 3 if Dependents == '3+' else int(Dependents),
        "Gender_Male": 1 if Gender == "Male" else 0,
        "Married_Yes": 1 if Married == "Yes" else 0,
        "Education_Not Graduate": 1 if Education == "Not Graduate" else 0,
        "Self_Employed_Yes": 1 if Self_Employed == "Yes" else 0,
        "Property_Area_Semiurban": 1 if Property_Area == "Semiurban" else 0,
        "Property_Area_Urban": 1 if Property_Area == "Urban" else 0,
    }

    user_df = pd.DataFrame([user_input])
    user_df = user_df.reindex(columns=features, fill_value=0)
    user_scaled = scaler.transform(user_df)

    # Button to make prediction
    if st.button("Predict Loan Approval"):
        prediction = model.predict(user_scaled)[0]

        # Evaluate model on original training data for visualization
        X = df.drop(['Loan_ID', 'Loan_Status'], axis=1)
        X = X.reindex(columns=features, fill_value=0)
        X_scaled = scaler.transform(X)
        y_true = df['Loan_Status']
        y_pred = model.predict(X_scaled)

        accuracy = accuracy_score(y_true, y_pred)
        st.write(f"📊 Model Accuracy: {accuracy * 100:.2f}%")

        if prediction == 1:
            st.success("✅ Loan Approved")
        else:
            st.error("❌ Loan Rejected")

        # --- Visualizations ---

        # Scatter Plot
        plt.figure(figsize=(6, 4))
        sns.scatterplot(data=df, x="ApplicantIncome", y="LoanAmount", hue=y_true, style=y_pred)
        plt.title("Applicant Income vs Loan Amount")
        st.pyplot(plt)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        st.pyplot(plt)

        # Actual vs Predicted
        plt.figure(figsize=(6, 4))
        sns.regplot(x=y_true, y=y_pred, ci=None, scatter_kws={'color': 'blue'}, line_kws={"color": "red"})
        plt.title("Actual vs Predicted Loan Status")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        st.pyplot(plt)
