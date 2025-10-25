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

# Load LabelEncoders for categorical columns
gender_le = joblib.load("gender_encoder.pkl")
education_le = joblib.load("education_encoder.pkl")
region_le = joblib.load("region_encoder.pkl")
loyalty_le = joblib.load("loyalty_encoder.pkl")
freq_le = joblib.load("purchase_frequency_encoder.pkl")
prod_cat_le = joblib.load("product_category_encoder.pkl")

st.set_page_config(page_title="Customer Purchase Prediction", layout="wide")

# ------------------------------
# Load dataset for dropdowns and visualizations
# ------------------------------
df = pd.read_csv("customer_data.csv")

# Map encoded classes to friendly names if needed
gender_options = ["Male", "Female"]
education_options = ["High School", "Bachelor", "Master", "PhD"]
region_options = ["North", "South", "East", "West"]
loyalty_options = ["Bronze", "Silver", "Gold", "Platinum"]
freq_options = ["Low", "Medium", "High"]
product_options = ["Books", "Clothing", "Food", "Electronics", "Home", "Beauty", "Health"]

# ------------------------------
# Sidebar Inputs
# ------------------------------
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=0, max_value=100, value=30)
income = st.sidebar.number_input("Income", min_value=0, value=50000)
promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, value=1)
satisfaction_score = st.sidebar.slider("Satisfaction Score", 0, 10, 5)

gender_input = st.sidebar.selectbox("Gender", gender_options)
education_input = st.sidebar.selectbox("Education", education_options)
region_input = st.sidebar.selectbox("Region", region_options)
loyalty_input = st.sidebar.selectbox("Loyalty Status", loyalty_options)
freq_input = st.sidebar.selectbox("Purchase Frequency", freq_options)
product_input = st.sidebar.selectbox("Product Category", product_options)

# ------------------------------
# Encode categorical inputs
# ------------------------------
input_df = pd.DataFrame({
    'age': [age],
    'gender': [gender_le.transform([gender_input])[0]],
    'income': [income],
    'education': [education_le.transform([education_input])[0]],
    'region': [region_le.transform([region_input])[0]],
    'loyalty_status': [loyalty_le.transform([loyalty_input])[0]],
    'purchase_frequency': [freq_le.transform([freq_input])[0]],
    'promotion_usage': [promotion_usage],
    'satisfaction_score': [satisfaction_score],
    'product_category': [prod_cat_le.transform([product_input])[0]]
})

# ------------------------------
# Prediction
# ------------------------------
if st.button("Predict Purchase Amount"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Purchase Amount: ₹{prediction:.2f}")

# ------------------------------
# Visualizations
# ------------------------------
st.subheader("Model Insights")

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

# Residuals Distribution
y_true = df["purchase_amount"].values
X_df = df[input_df.columns].copy()
# Encode categorical columns in X_df for prediction
for col, le in zip(['gender','education','region','loyalty_status','purchase_frequency','product_category'],
                   [gender_le, education_le, region_le, loyalty_le, freq_le, prod_cat_le]):
    X_df[col] = le.transform(X_df[col])
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
