# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Model and Encoders
# ------------------------------
model = joblib.load("xgb_purchase_model.pkl")

# Load encoders
gender_le = joblib.load("gender_encoder.pkl")
education_le = joblib.load("education_encoder.pkl")
region_le = joblib.load("region_encoder.pkl")
loyalty_le = joblib.load("loyalty_encoder.pkl")
freq_le = joblib.load("purchase_frequency_encoder.pkl")
prod_cat_le = joblib.load("product_category_encoder.pkl")

encoders = {
    'gender': gender_le,
    'education': education_le,
    'region': region_le,
    'loyalty_status': loyalty_le,
    'purchase_frequency': freq_le,
    'product_category': prod_cat_le
}

# ------------------------------
# Load dataset for visualization
# ------------------------------
df = pd.read_csv("customer_data.csv")

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("🛒 Customer Purchase Prediction")
st.markdown("Predict the purchase amount of a customer based on their features.")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
income = st.sidebar.number_input("Income", min_value=0, value=50000)
promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, va_
