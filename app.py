# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="🛍️ Customer Purchase Prediction", layout="wide")

st.title("🧠 Customer Purchase Behavior Prediction")
st.markdown("Predict the **purchase amount** based on customer demographics and shopping preferences.")

# ------------------------------
# Check for Required Files
# ------------------------------
required_files = [
    "xgb_purchase_model.pkl",
    "gender_encoder.pkl",
    "education_encoder.pkl",
    "region_encoder.pkl",
    "loyalty_encoder.pkl",
    "purchase_frequency_encoder.pkl",
    "product_category_encoder.pkl",
]

missing_files = [f for f in required_files if not os.path.exists(f)]
if missing_files:
    st.error(f"⚠️ Missing files: {', '.join(missing_files)}. Please upload them before running the app.")
    st.stop()

# ------------------------------
# Load Model and Encoders
# ------------------------------
model = joblib.load("xgb_purchase_model.pkl")
gender_le = joblib.load("gender_encoder.pkl")
education_le = joblib.load("education_encoder.pkl")
region_le = joblib.load("region_encoder.pkl")
loyalty_le = joblib.load("loyalty_encoder.pkl")
freq_le = joblib.load("purchase_frequency_encoder.pkl")
prod_cat_le = joblib.load("product_category_encoder.pkl")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("🎛️ Input Customer Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
income = st.sidebar.number_input("Income (₹)", min_value=0, value=50000)
promotion_usage_input = st.sidebar.selectbox("Promotion Usage", ["Yes", "No"])
satisfaction_score = st.sidebar.slider("Satisfaction Score", 0, 10, 5)

gender_input = st.sidebar.selectbox("Gender", ["Male", "Female"])
education_input = st.sidebar.selectbox("Education", education_le.classes_)
region_input = st.sidebar.selectbox("Region", region_le.classes_)
loyalty_input = st.sidebar.selectbox("Loyalty Status", loyalty_le.classes_)
freq_input = st.sidebar.selectbox("Purchase Frequency", freq_le.classes_)
prod_cat_input = st.sidebar.selectbox("Product Category", prod_cat_le.classes_)

# Convert Yes/No to numeric
promotion_usage = 1 if promotion_usage_input == "Yes" else 0

# ------------------------------
# Encode Inputs Safely
# ------------------------------
try:
    input_data = {
        "age": [age],
        "gender": [gender_le.transform([gender_input])[0]],
        "income": [income],
        "education": [education_le.transform([education_input])[0]],
        "region": [region_le.transform([region_input])[0]],
        "loyalty_status": [loyalty_le.transform([loyalty_input])[0]],
        "purchase_frequency": [freq_le.transform([freq_input])[0]],
        "promotion_usage": [promotion_usage],
        "satisfaction_score": [satisfaction_score],
        "product_category": [prod_cat_le.transform([prod_cat_input])[0]],
    }

    input_df = pd.DataFrame(input_data)

except ValueError as e:
    st.error(f"⚠️ Invalid input detected: {e}")
    st.stop()

# ------------------------------
# Prediction Section
# ------------------------------
if st.button("🎯 Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 **Predicted Purchase Amount:** ₹{prediction:.2f}")

# ------------------------------
# Visualizations (if dataset available)
# ------------------------------
if os.path.exists("customer_data.csv"):
    df = pd.read_csv("customer_data.csv")

    if not df.empty and "purchase_amount" in df.columns:
        st.subheader("📊 Model Insights & Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            fig1, ax1 = plt.subplots()
            sns.scatterplot(x=df["age"], y=df["purchase_amount"], ax=ax1)
            ax1.set_title("Age vs Purchase Amount")
            st.pyplot(fig1)

        with col2:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(x=df["income"], y=df["purchase_amount"], ax=ax2)
            ax2.set_title("Income vs Purchase Amount")
            st.pyplot(fig2)

        # Feature Importance
        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=input_df.columns)
            fig3, ax3 = plt.subplots()
            fi.sort_values().plot(kind='barh', ax=ax3)
            ax3.set_title("Feature Importance")
            st.pyplot(fig3)
    else:
        st.warning(" Dataset found but missing 'purchase_amount' column.")
else:
    st.info(" Dataset not found. Visualizations skipped.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Developed by Khushi Yadav")
