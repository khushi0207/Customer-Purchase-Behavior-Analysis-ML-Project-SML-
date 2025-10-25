import streamlit as st
import pandas as pd
import pickle
import numpy as np

# ------------------------------------------------------------
# Load Model & Encoders
# ------------------------------------------------------------
model = pickle.load(open("xgb_purchase_model.pkl.pkl", "rb"))
gender_le = pickle.load(open("gender_encoder.pkl", "rb"))
education_le = pickle.load(open("education_encoder.pkl", "rb"))
region_le = pickle.load(open("region_encoder.pkl", "rb"))
loyalty_le = pickle.load(open("loyalty_encoder.pkl", "rb"))
freq_le = pickle.load(open("frequency_encoder.pkl", "rb"))
prod_cat_le = pickle.load(open("product_category_encoder.pkl", "rb"))

st.title("🛒 Customer Purchase Behavior Prediction App")
st.write("Predict customer purchase behavior based on demographic and behavioral factors.")

# ------------------------------------------------------------
# User-Friendly Dropdowns with Internal Mappings
# ------------------------------------------------------------

gender_options = {"Male": gender_le.classes_[0], "Female": gender_le.classes_[1]}
education_options = {label: label for label in education_le.classes_}
region_options = {label: label for label in region_le.classes_}
loyalty_options = {label: label for label in loyalty_le.classes_}
freq_options = {label: label for label in freq_le.classes_}
prod_cat_options = {label: label for label in prod_cat_le.classes_}

# Sidebar Inputs
st.sidebar.header("Enter Customer Details")

gender_input = st.sidebar.selectbox("Gender", list(gender_options.keys()))
age_input = st.sidebar.number_input("Age", min_value=10, max_value=100, value=25)
income_input = st.sidebar.number_input("Annual Income", min_value=1000, value=30000)
education_input = st.sidebar.selectbox("Education Level", list(education_options.keys()))
region_input = st.sidebar.selectbox("Region", list(region_options.keys()))
loyalty_input = st.sidebar.selectbox("Loyalty Status", list(loyalty_options.keys()))
freq_input = st.sidebar.selectbox("Purchase Frequency", list(freq_options.keys()))
promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, value=1)
product_input = st.sidebar.selectbox("Product Category", list(prod_cat_options.keys()))

# ------------------------------------------------------------
# Transform Friendly Inputs to Encoded Values
# ------------------------------------------------------------
try:
    gender_val = gender_le.transform([gender_options[gender_input]])[0]
    education_val = education_le.transform([education_options[education_input]])[0]
    region_val = region_le.transform([region_options[region_input]])[0]
    loyalty_val = loyalty_le.transform([loyalty_options[loyalty_input]])[0]
    freq_val = freq_le.transform([freq_options[freq_input]])[0]
    prod_cat_val = prod_cat_le.transform([prod_cat_options[product_input]])[0]
except Exception as e:
    st.error(f"⚠️ Error while transforming inputs: {e}")
    st.stop()

# ------------------------------------------------------------
# Prepare Input DataFrame
# ------------------------------------------------------------
input_data = pd.DataFrame({
    'gender': [gender_val],
    'age': [age_input],
    'income': [income_input],
    'education': [education_val],
    'region': [region_val],
    'loyalty_status': [loyalty_val],
    'purchase_frequency': [freq_val],
    'promotion_usage': [promotion_usage],
    'product_category': [prod_cat_val]
})

st.subheader("📋 Input Summary")
st.write(input_data)

# ------------------------------------------------------------
# Make Prediction
# ------------------------------------------------------------
if st.button("Predict Purchase Behavior"):
    prediction = model.predict(input_data)[0]
    st.success(f"🧠 Predicted Purchase Behavior: **{prediction}**")
