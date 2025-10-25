# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load Model and Encoders
# ------------------------------
model = joblib.load("xgb_purchase_model.pkl")  # Trained XGBoost Regressor

# Load LabelEncoders for categorical features
gender_le = joblib.load("gender_encoder.pkl")
education_le = joblib.load("education_encoder.pkl")
region_le = joblib.load("region_encoder.pkl")
loyalty_le = joblib.load("loyalty_encoder.pkl")
freq_le = joblib.load("purchase_frequency_encoder.pkl")
category_le = joblib.load("product_category_encoder.pkl")

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

# ------------------------------
# Title
# ------------------------------
st.title("🛒 Customer Purchase Prediction")
st.markdown("""
Predict the purchase amount of a customer based on their features.
""")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Input Features")

# Numeric
age = st.sidebar.number_input("Age", min_value=0, value=30)
income = st.sidebar.number_input("Income", min_value=0, value=50000)
promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, value=0)
satisfaction_score = st.sidebar.number_input("Satisfaction Score", min_value=0, max_value=10, value=5)

# Categorical
gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
education = st.sidebar.selectbox("Education", ['High School', 'Bachelor', 'Master', 'PhD', 'Other'])
region = st.sidebar.selectbox("Region", ['North', 'South', 'East', 'West', 'Central'])
loyalty_status = st.sidebar.selectbox("Loyalty Status", ['Bronze', 'Silver', 'Gold', 'Platinum'])
purchase_frequency = st.sidebar.selectbox("Purchase Frequency", ['Low', 'Medium', 'High'])
product_category = st.sidebar.selectbox("Product Category", ['Books', 'Clothing', 'Food', 'Electronics', 'Home', 'Beauty', 'Health'])

# ------------------------------
# Prepare Input for Model
# ------------------------------
input_df = pd.DataFrame([[
    age, income, promotion_usage, satisfaction_score,
    gender, education, region, loyalty_status,
    purchase_frequency, product_category
]], columns=[
    "age", "income", "promotion_usage", "satisfaction_score",
    "gender", "education", "region", "loyalty_status",
    "purchase_frequency", "product_category"
])

# Encode categorical features
input_df['gender'] = gender_le.transform(input_df['gender'])
input_df['education'] = education_le.transform(input_df['education'])
input_df['region'] = region_le.transform(input_df['region'])
input_df['loyalty_status'] = loyalty_le.transform(input_df['loyalty_status'])
input_df['purchase_frequency'] = freq_le.transform(input_df['purchase_frequency'])
input_df['product_category'] = category_le.transform(input_df['product_category'])

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Purchase Amount: ₹{prediction:.2f}")

# ------------------------------
# Optional Visualizations
# ------------------------------
st.subheader("📊 Model Insights")

try:
    df = pd.read_csv("customer_data.csv")
    df.columns = df.columns.str.strip()  # remove extra spaces
except FileNotFoundError:
    st.warning("Dataset not found. Visualizations skipped.")
    df = pd.DataFrame()

if not df.empty:
    # Ensure required columns exist
    required_cols = ["age", "income", "promotion_usage", "satisfaction_score", "purchase_amount"]
    if all(col in df.columns for col in required_cols):
        # Scatter: Age vs Purchase
        fig1, ax1 = plt.subplots()
        sns.scatterplot(x=df["age"], y=df["purchase_amount"], ax=ax1)
        ax1.set_title("Age vs Purchase Amount")
        st.pyplot(fig1)

        # Scatter: Income vs Purchase
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
        X_df = df[input_df.columns]
        # Encode df categorical columns
        X_df['gender'] = gender_le.transform(X_df['gender'])
        X_df['education'] = education_le.transform(X_df['education'])
        X_df['region'] = region_le.transform(X_df['region'])
        X_df['loyalty_status'] = loyalty_le.transform(X_df['loyalty_status'])
        X_df['purchase_frequency'] = freq_le.transform(X_df['purchase_frequency'])
        X_df['product_category'] = category_le.transform(X_df['product_category'])

        y_true = df["purchase_amount"].values
        y_pred_all = model.predict(X_df)
        residuals = y_true - y_pred_all
        fig4, ax4 = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax4)
        ax4.set_title("Residuals Distribution")
        st.pyplot(fig4)
    else:
        st.warning("Required columns not found for visualizations.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Made by Khushi Yadav")
