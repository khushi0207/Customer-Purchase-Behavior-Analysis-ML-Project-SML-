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
model = joblib.load("xgb_purchase_model.pkl")   # your trained XGBoost Regressor
# If you used LabelEncoder or scaler
# le = joblib.load("label_encoder.pkl")  

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

# Example: replace these with your actual features
feature1 = st.sidebar.number_input("Feature 1", min_value=0, value=10)
feature2 = st.sidebar.number_input("Feature 2", min_value=0, value=5)
feature3 = st.sidebar.selectbox("Feature 3 (Category)", ['Books', 'Clothing', 'Food', 'Electronics', 'Home', 'Beauty',
       'Health'])

# Prepare input for model
input_df = pd.DataFrame([[feature1, feature2, feature3]], columns=["feature1","feature2","feature3"])

# Encode categorical if needed
# input_df['feature3'] = le.transform(input_df['feature3'])

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Purchase Amount: ₹{prediction:.2f}")

# ------------------------------
# Optional Visualizations
# ------------------------------
st.subheader("📊 Model Insights")

# Load dataset (optional: small sample for visualization)
df = pd.read_csv("customer_data.csv")  # replace with your dataset path

# Scatter plot: Feature1 vs Purchase Amount
fig1, ax1 = plt.subplots()
sns.scatterplot(x=df["feature1"], y=df["purchase_amount"], ax=ax1)
ax1.set_title("Feature1 vs Purchase Amount")
st.pyplot(fig1)

# Scatter plot: Feature2 vs Purchase Amount
fig2, ax2 = plt.subplots()
sns.scatterplot(x=df["feature2"], y=df["purchase_amount"], ax=ax2)
ax2.set_title("Feature2 vs Purchase Amount")
st.pyplot(fig2)

# Feature Importance (from XGBoost)
if hasattr(model, "feature_importances_"):
    fi = pd.Series(model.feature_importances_, index=input_df.columns)
    fig3, ax3 = plt.subplots()
    fi.sort_values().plot(kind='barh', ax=ax3)
    ax3.set_title("Feature Importance")
    st.pyplot(fig3)

# Residuals plot (optional)
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
