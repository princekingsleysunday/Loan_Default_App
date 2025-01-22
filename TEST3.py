import streamlit as st
import joblib
import numpy as np
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('PLAS.model')
scaler = joblib.load('scaler.model')

# Page setup
st.set_page_config(
    page_title="Credit Default Prediction App",
    layout="wide",
    page_icon="💳"
)

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state["history"] = []

# Function to add a prediction to history
def add_to_history(features, prediction, probability):
    st.session_state["history"].append({
        "Income": features[0][0],
        "Savings": features[0][1],
        "Debt": features[0][2],
        "Debt-to-Income Ratio": features[0][3],
        "Expenditure": features[0][4],
        "Expenditure-to-Income Ratio": features[0][5],
        "Expenditure-to-Debt Ratio": features[0][6],
        "Gambling Category": features[0][7],
        "Credit Score": features[0][8],
        "Prediction": "Default" if prediction[0] == 1 else "No Default",
        "Probability": f"{probability:.2%}" if prediction[0] == 1 else f"{1 - probability:.2%}"
    })

# Login page
def login_page():
    st.title("🔒 Login to Access")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "password123":  
            st.session_state["logged_in"] = True
            st.success("Logged in successfully!")
        else:
            st.error("Invalid username or password.")

# Function to calculate derived ratios
def calculate_ratios(income, expenditure, debt):
    debt_income_ratio = round(debt / income, 2) if income > 0 else 0
    expenditure_income_ratio = round(expenditure / income, 2) if income > 0 else 0
    expenditure_debt_ratio = round(expenditure / debt, 2) if debt > 0 else 0
    return debt_income_ratio, expenditure_income_ratio, expenditure_debt_ratio

# Function for manual input
def manual_input():
    st.header("Enter Customer Financial Data")
    col1, col2, col3 = st.columns(3)

    with col1:
        income = st.number_input('Monthly Income (in Naira)', min_value=0, step=1000)
        savings = st.number_input('Savings (in Naira)', min_value=0, step=1000)

    with col2:
        debt = st.number_input('Current Debt (in Naira)', min_value=0, step=1000)
        expenditure = st.number_input('Total Expenditure (Last 12 months, in Naira)', min_value=0, step=1000)

    debt_income_ratio, expenditure_income_ratio, expenditure_debt_ratio = calculate_ratios(income, expenditure, debt)

    st.text_input("Debt-to-Income Ratio", value=debt_income_ratio, disabled=True)
    st.text_input("Expenditure-to-Income Ratio", value=expenditure_income_ratio, disabled=True)
    st.text_input("Expenditure-to-Debt Ratio", value=expenditure_debt_ratio, disabled=True)

    with col3:
        gambling_category = st.selectbox('Gambling Category (High=1, Low=2, None=3)', [1, 2, 3])
        credit_score = st.number_input('Credit Score (0-800)', min_value=0, max_value=800)

    features = [income, savings, debt, debt_income_ratio, expenditure, expenditure_income_ratio, expenditure_debt_ratio, gambling_category, credit_score]
    return np.array(features).reshape(1, -1)

# Function for file upload
def upload_statement():
    st.header("Upload Customer Statement")

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "pdf"])

    if uploaded_file is not None:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        # Check if the file is either CSV or PDF
        if file_type not in ['csv', 'pdf']:
            st.error("❗ Invalid file type. Please upload a CSV or PDF file.")
            return None
        
        if file_type == "csv":
            try:
                data = pd.read_csv(uploaded_file)
                st.write("Uploaded Data Preview:")
                st.write(data.head())

                if 'Income' in data.columns and 'Debt' in data.columns and 'Expenditure' in data.columns:
                    income = data['Income'].iloc[0]
                    debt = data['Debt'].iloc[0]
                    expenditure = data['Expenditure'].iloc[0]
                    savings = 0  
                    gambling_category = 1  
                    credit_score = 600  

                    debt_income_ratio, expenditure_income_ratio, expenditure_debt_ratio = calculate_ratios(income, expenditure, debt)

                    features = [income, savings, debt, debt_income_ratio, expenditure, expenditure_income_ratio, expenditure_debt_ratio, gambling_category, credit_score]
                    features = np.array(features).reshape(1, -1)

                    return features  # Returning the features after processing the file
                else:
                    st.warning("The uploaded CSV file doesn't contain the expected columns ('Income', 'Debt', 'Expenditure').")
                    return None
            except Exception as e:
                st.error(f"❗ Error reading CSV file: {e}")
                return None

        elif file_type == "pdf":
            st.info("PDF files are currently not processed for predictions.")
            return None

    else:
        return None

# Function to display prediction history
def display_history():
    if len(st.session_state["history"]) > 0:
        st.header("Prediction History")
        df = pd.DataFrame(st.session_state["history"])
        st.dataframe(df)
        
        # Downloadable CSV
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download History",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
    else:
        st.info("No prediction history yet.")

# Main app logic
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_page()
else:
    st.sidebar.title("Navigation")
    option = st.sidebar.selectbox("Select a Page:", ["📋 Manual Input", "📁 Upload Statement", "📜 Prediction History"])

    if option == "📋 Manual Input":
        input_data_manual = manual_input()
        if st.button("Predict (Manual Input)"):
            if input_data_manual is not None:
                # Scale the data before prediction
                scaled_data = scaler.transform(input_data_manual)
                prediction = model.predict(scaled_data)
                probability = model.predict_proba(scaled_data)[0][1]
                st.header("Prediction Result:")
                if prediction[0] == 1:
                    st.error(f"🔴 Customer is likely to default (Probability: {probability:.2%}).")
                else:
                    st.success(f"🟢 Customer is NOT likely to default (Probability: {1 - probability:.2%}).")
                add_to_history(input_data_manual, prediction, probability)

    elif option == "📁 Upload Statement":
        features = upload_statement()
        if features is not None:
            if st.button("Predict (Upload Statement)"):
                # Scale the data before prediction
                scaled_data = scaler.transform(features)
                prediction = model.predict(scaled_data)
                probability = model.predict_proba(scaled_data)[0][1]
                st.header("Prediction Result:")
                if prediction[0] == 1:
                    st.error(f"🔴 Customer is likely to default (Probability: {probability:.2%}).")
                else:
                    st.success(f"🟢 Customer is NOT likely to default (Probability: {1 - probability:.2%}).")
                add_to_history(features, prediction, probability)

    elif option == "📜 Prediction History":
        display_history()
