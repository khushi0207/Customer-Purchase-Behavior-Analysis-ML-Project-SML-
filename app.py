# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

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

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

# ------------------------------
# Title and Description
# ------------------------------
st.title("🛒 Customer Purchase Prediction")
st.markdown("""
Predict the purchase amount of a customer based on their features.
""")

# ------------------------------
# Input Section
# ------------------------------
st.sidebar.header("Input Features")

# Numeric features
age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
income = st.sidebar.number_input("Income", min_value=0, value=50000)
promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, value=1)
satisfaction_score = st.sidebar.slider("Satisfaction Score", 0, 10, 5)

# Categorical features
gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
education = st.sidebar.selectbox("Education", ['High School', 'Bachelor', 'Master', 'PhD'])
region = st.sidebar.selectbox("Region", ['North', 'South', 'East', 'West'])
loyalty_status = st.sidebar.selectbox("Loyalty Status", ['Bronze', 'Silver', 'Gold', 'Platinum'])
purchase_frequency = st.sidebar.selectbox("Purchase Frequency", ['Low', 'Medium', 'High'])
product_category = st.sidebar.selectbox("Product Category", ['Books', 'Clothing', 'Food', 'Electronics', 'Home', 'Beauty', 'Health'])

# Prepare input DataFrame
input_df = pd.DataFrame([[age, gender, income, education, region, loyalty_status, purchase_frequency, promotion_usage, satisfaction_score, product_category]],
                        columns=['age', 'gender', 'income', 'education', 'region', 'loyalty_status', 'purchase_frequency', 'promotion_usage', 'satisfaction_score', 'product_category'])

# ------------------------------
# Categorical Encoding
# ------------------------------
encoders = {
    'gender': gender_le,
    'education': education_le,
    'region': region_le,
    'loyalty_status': loyalty_le,
    'purchase_frequency': freq_le,
    'product_category': prod_cat_le
}

# Standardize input and transform
for col, le in encoders.items():
    input_df[col] = input_df[col].str.title()
    if input_df[col][0] not in le.classes_:
        st.error(f" Invalid input for '{col}'. Choose from: {', '.join(le.classes_)}")
        st.stop()
    input_df[col] = le.transform(input_df[col])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Purchase Amount: ₹{prediction:.2f}")

# ------------------------------
# Optional Visualizations
# ------------------------------
st.subheader("Model Insights")

# Load dataset (small sample for visualization)
df = pd.read_csv("customer_data.csv")

# Scatter plots
fig1, ax1 = plt.subplots()
sns.scatterplot(x=df["age"], y=df["purchase_amount"], ax=ax1)
ax1.set_title("Age vs Purchase Amount")
st.pyplot(fig1)

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

# Residuals
y_true = df["purchase_amount"].values
X_df = df[input_df.columns]
y_pred_all = model.predict(X_df)
residuals = y_true - y_pred_all
fig4, ax4 = plt.subplots()
sns.histplot(residuals, bins=30, kde=True, ax=ax4)
ax4.set_title("Residuals Distribution")
st.pyplot(fig4)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Made by Khushi Yadav")
