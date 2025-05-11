import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
model = joblib.load('model.pkl')

# App title
st.title("Bank Personal Loan Prediction App")
st.markdown("Predict whether a customer will take a personal loan.")

# Sidebar inputs
st.sidebar.header("Customer Input Features")

def user_input_features():
    age = st.sidebar.slider('Age', 18, 70, 30)
    experience = st.sidebar.slider('Experience (Years)', -5, 50, 5)
    income = st.sidebar.slider('Income (in $1000s)', 0, 300, 60)
    family = st.sidebar.selectbox('Family Size', [1, 2, 3, 4])
    ccavg = st.sidebar.slider('Credit Card Avg Spending', 0.0, 10.0, 1.5)
    education = st.sidebar.selectbox('Education Level', [1, 2, 3], format_func=lambda x: {1: "Undergrad", 2: "Graduate", 3: "Advanced/Professional"}[x])
    mortgage = st.sidebar.slider('Mortgage', 0, 700, 0)
    securities_account = st.sidebar.selectbox('Securities Account', [0, 1])
    cd_account = st.sidebar.selectbox('CD Account', [0, 1])
    online = st.sidebar.selectbox('Online Banking', [0, 1])
    creditcard = st.sidebar.selectbox('Credit Card Holder', [0, 1])

    data = {
        'age': age,
        'experience': experience,
        'income': income,
        'family': family,
        'ccavg': ccavg,
        'education': education,
        'mortgage': mortgage,
        'securities_account': securities_account,
        'cd_account': cd_account,
        'online': online,
        'creditcard': creditcard
    }
    return pd.DataFrame(data, index=[0])

# Input from user
input_df = user_input_features()

# Prediction
st.subheader('User Input Features')
st.write(input_df)

prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader('Prediction')
st.write('Loan Approved' if prediction[0] == 1 else 'Loan Not Approved')

st.subheader('Prediction Probability')
st.write(f"Probability of Loan Approval: {prediction_proba[0][1]:.2%}")

# Optional: show dataset
if st.checkbox('Show Raw Data'):
    df = pd.read_csv("Bank_Personal_Loan_Modelling.csv")
    st.write(df.head())
