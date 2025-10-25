import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(page_title="🛍️ Customer Purchase Prediction", layout="wide")

st.title("🧠 Customer Purchase Behavior Prediction")
st.markdown("Predict the **purchase amount** based on customer demographics and preferences.")

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
# User-friendly category mappings
# (These are what the user sees)
# ------------------------------
gender_options = ["Male", "Female"]
region_options = ["North", "South", "East", "West", "Central"]
education_options = ["High School", "Graduate", "Post Graduate", "PhD"]
loyalty_options = ["Bronze", "Silver", "Gold", "Platinum"]
product_category_options = ["Electronics", "Clothing", "Beauty", "Home", "Sports"]

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("🎛️ Input Customer Details")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
income = st.sidebar.number_input("Income (₹)", min_value=0, value=50000)
promotion_usage_input = st.sidebar.selectbox("Promotion Usage", ["Yes", "No"])
satisfaction_score = st.sidebar.slider("Satisfaction Score", 0, 10, 5)

gender_input = st.sidebar.selectbox("Gender", gender_options)
education_input = st.sidebar.selectbox("Education Level", education_options)
region_input = st.sidebar.selectbox("Region", region_options)
loyalty_input = st.sidebar.selectbox("Loyalty Status", loyalty_options)
freq_input = st.sidebar.selectbox("Purchase Frequency", list(freq_le.classes_))
prod_cat_input = st.sidebar.selectbox("Product Category", product_category_options)

promotion_usage = 1 if promotion_usage_input == "Yes" else 0

# ------------------------------
# Helper: safely encode text using the fitted LabelEncoder
# ------------------------------
def safe_encode(encoder, value):
    try:
        return int(encoder.transform([value])[0])
    except:
        st.error(f"⚠️ Invalid input: '{value}'. Must be one of: {', '.join(encoder.classes_)}")
        st.stop()

# ------------------------------
# Prepare input dataframe
# ------------------------------
input_df = pd.DataFrame({
    "age": [age],
    "gender": [safe_encode(gender_le, gender_input)],
    "income": [income],
    "education": [safe_encode(education_le, education_input)],
    "region": [safe_encode(region_le, region_input)],
    "loyalty_status": [safe_encode(loyalty_le, loyalty_input)],
    "purchase_frequency": [safe_encode(freq_le, freq_input)],
    "promotion_usage": [promotion_usage],
    "satisfaction_score": [satisfaction_score],
    "product_category": [safe_encode(prod_cat_le, prod_cat_input)],
})

# ------------------------------
# Prediction
# ------------------------------
if st.button("🎯 Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]

    st.success(f"💰 **Predicted Purchase Amount:** ₹{prediction:.2f}")

    st.markdown("### 🧍 Customer Summary")
    st.info(f"""
    👩‍🦰 Gender: {gender_input}  
    🎓 Education: {education_input}  
    🌍 Region: {region_input}  
    💎 Loyalty: {loyalty_input}  
    🛍️ Product Category: {prod_cat_input}  
    💸 Income: ₹{income}  
    🧾 Promotion Usage: {promotion_usage_input}  
    😊 Satisfaction: {satisfaction_score}/10  
    """)

# ------------------------------
# Optional Visualizations
# ------------------------------
if os.path.exists("customer_data.csv"):
    df = pd.read_csv("customer_data.csv")

    if not df.empty and "purchase_amount" in df.columns:
        st.subheader("📊 Model Insights")

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

        if hasattr(model, "feature_importances_"):
            fi = pd.Series(model.feature_importances_, index=input_df.columns)
            fig3, ax3 = plt.subplots()
            fi.sort_values().plot(kind='barh', ax=ax3)
            ax3.set_title("Feature Importance")
            st.pyplot(fig3)
    else:
        st.warning("⚠️ Dataset found but missing 'purchase_amount' column.")
else:
    st.info("ℹ️ Dataset not found. Visualizations skipped.")

st.markdown("---")
st.markdown("Developed by Khushi Yadav")
