# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
# Load dataset
# ------------------------------
df = pd.read_csv("customer_data.csv")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
income = st.sidebar.number_input("Income", min_value=0, value=50000)
promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, value=1)
satisfaction_score = st.sidebar.slider("Satisfaction Score", 0, 10, 5)

# Friendly string options
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education", df['education'].unique())
region = st.sidebar.selectbox("Region", df['region'].unique())
loyalty_status = st.sidebar.selectbox("Loyalty Status", df['loyalty_status'].unique())
purchase_frequency = st.sidebar.selectbox("Purchase Frequency", df['purchase_frequency'].unique())
product_category = st.sidebar.selectbox("Product Category", df['product_category'].unique())

# ------------------------------
# Prepare input DataFrame
# ------------------------------
input_df = pd.DataFrame([[age, gender, income, education, region, loyalty_status,
                          purchase_frequency, promotion_usage, satisfaction_score,
                          product_category]],
                        columns=['age', 'gender', 'income', 'education', 'region',
                                 'loyalty_status', 'purchase_frequency', 'promotion_usage',
                                 'satisfaction_score', 'product_category'])

# ------------------------------
# Encode categorical features safely
# ------------------------------
encoders = {
    'gender': gender_le,
    'education': education_le,
    'region': region_le,
    'loyalty_status': loyalty_le,
    'purchase_frequency': freq_le,
    'product_category': prod_cat_le
}

for col, le in encoders.items():
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        st.error(f"Invalid input for '{col}'. Choose from: {', '.join(le.classes_)}")
        st.stop()

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Purchase Amount: ₹{prediction:.2f}")

# ------------------------------
# Visualizations
# ------------------------------
st.subheader("📊 Model Insights")

model_features = ['age', 'gender', 'income', 'education', 'region',
                  'loyalty_status', 'purchase_frequency', 'promotion_usage',
                  'satisfaction_score', 'product_category']

# Encode dataset categorical columns for predictions
X_df = df[model_features].copy()
for col, le in encoders.items():
    X_df[col] = le.transform(X_df[col].astype(str))

y_true = df["purchase_amount"].values
y_pred_all = model.predict(X_df)
residuals = y_true - y_pred_all

# Scatter Plots
fig1, ax1 = plt.subplots()
sns.scatterplot(x=df["age"], y=y_true, ax=ax1)
ax1.set_title("Age vs Purchase Amount")
st.pyplot(fig1)

fig2, ax2 = plt.subplots()
sns.scatterplot(x=df["income"], y=y_true, ax=ax2)
ax2.set_title("Income vs Purchase Amount")
st.pyplot(fig2)

# Predicted vs Actual
fig3, ax3 = plt.subplots()
sns.scatterplot(x=y_true, y=y_pred_all, ax=ax3)
ax3.set_xlabel("Actual Purchase Amount")
ax3.set_ylabel("Predicted Purchase Amount")
ax3.set_title("Predicted vs Actual Purchase Amount")
st.pyplot(fig3)

# Feature Importance
if hasattr(model, "feature_importances_"):
    fi = pd.Series(model.feature_importances_, index=model_features)
    fig4, ax4 = plt.subplots()
    fi.sort_values().plot(kind="barh", ax=ax4)
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)

# Residuals Distribution
fig5, ax5 = plt.subplots()
sns.histplot(residuals, bins=30, kde=True, ax=ax5)
ax5.set_title("Residuals Distribution")
st.pyplot(fig5)

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Made by Khushi Yadav")
