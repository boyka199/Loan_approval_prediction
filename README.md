# 🏦 Loan Approval Prediction Web App

This Streamlit web app uses logistic regression to predict whether a loan application will be approved based on applicant's data.

---

## 🚀 Features

- Upload your own training and test CSV files
- Preprocesses data (missing values, outliers, encoding, scaling)
- Trains a logistic regression model
- Predicts loan status
- Shows model accuracy and visualizations:
  - ApplicantIncome vs LoanAmount (scatter plot)
  - Confusion matrix
  - Actual vs Predicted (regression plot)

---
### ✅ For Predictions we need:

| Column            | Example Values        |
|-------------------|-----------------------|
| Gender            | Male / Female         |
| Married           | Yes / No              |
| Dependents        | 0 / 1 / 2 / 3+         |
| Education         | Graduate / Not Graduate |
| Self_Employed     | Yes / No              |
| ApplicantIncome   | 5000, 2000, etc.      |
| CoapplicantIncome | 0.0, 1500.0, etc.     |
| LoanAmount        | 120, 85, etc.         |
| Loan_Amount_Term  | 360.0, 120.0, etc.    |
| Credit_History    | 1.0 / 0.0             |
| Property_Area     | Urban / Semiurban / Rural |
| Loan_Status       | Y / N                 |
