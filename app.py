# app.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ------------------------------
# Load Model and Encoders
# ------------------------------
model = joblib.load("xgb_purchase_model.pkl")

# Load LabelEncoders
gender_le = joblib.load("gender_encoder.pkl")
education_le = joblib.load("education_encoder.pkl")
region_le = joblib.load("region_encoder.pkl")
loyalty_le = joblib.load("loyalty_encoder.pkl")
freq_le = joblib.load("purchase_frequency_encoder.pkl")
prod_cat_le = joblib.load("product_category_encoder.pkl")

# ------------------------------
# Streamlit Page Config
# ------------------------------
st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")
st.title("🛒 Customer Purchase Prediction")
st.markdown("Predict the purchase amount of a customer based on their features.")

# ------------------------------
# Load Dataset (if exists)
# ------------------------------
dataset_path = "customer_data.csv"
if os.path.exists(dataset_path) and os.path.getsize(dataset_path) > 0:
    df = pd.read_csv(dataset_path)
else:
    df = None
    st.warning("Dataset not found or empty. Visualizations will be disabled.")

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
income = st.sidebar.number_input("Income", min_value=0, value=50000)
promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, value=1)
satisfaction_score = st.sidebar.slider("Satisfaction Score", 0, 10, 5)

# User-friendly dropdowns
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education", sorted(list(education_le.classes_)))
region = st.sidebar.selectbox("Region", sorted(list(region_le.classes_)))
loyalty_status = st.sidebar.selectbox("Loyalty Status", sorted(list(loyalty_le.classes_)))
purchase_frequency = st.sidebar.selectbox("Purchase Frequency", sorted(list(freq_le.classes_)))
product_category = st.sidebar.selectbox("Product Category", sorted(list(prod_cat_le.classes_)))

# ------------------------------
# Prepare Input DataFrame
# ------------------------------
input_df = pd.DataFrame([[age, gender, income, education, region, loyalty_status,
                          purchase_frequency, promotion_usage, satisfaction_score,
                          product_category]],
                        columns=['age', 'gender', 'income', 'education', 'region',
                                 'loyalty_status', 'purchase_frequency', 'promotion_usage',
                                 'satisfaction_score', 'product_category'])

# Encode categorical features
encoders = {
    'gender': gender_le,
    'education': education_le,
    'region': region_le,
    'loyalty_status': loyalty_le,
    'purchase_frequency': freq_le,
    'product_category': prod_cat_le
}

for col, le in encoders.items():
    # Handle unseen values gracefully
    try:
        input_df[col] = le.transform(input_df[col])
    except ValueError:
        st.error(f"Invalid input for '{col}'. Please select a valid option.")
        st.stop()

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Purchase Amount: ₹{prediction:.2f}")

# ------------------------------
# Visualizations (only if dataset exists)
# ------------------------------
if df is not None:
    st.subheader("📊 Model Insights")

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
    try:
        X_df = df[input_df.columns]
        for col, le in encoders.items():
            X_df[col] = le.transform(X_df[col].astype(str))
        y_true = df["purchase_amount"].values
        y_pred_all = model.predict(X_df)
        residuals = y_true - y_pred_all
        fig4, ax4 = plt.subplots()
        sns.histplot(residuals, bins=30, kde=True, ax=ax4)
        ax4.set_title("Residuals Distribution")
        st.pyplot(fig4)
    except Exception as e:
        st.warning("Residual plot could not be generated: " + str(e))

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown("Made by Khushi Yadav")
