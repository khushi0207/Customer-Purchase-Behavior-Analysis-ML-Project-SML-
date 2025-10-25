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


#promotion_usage = st.sidebar.number_input("Promotion Usage", min_value=0, value=1)

promotion_usage = st.sidebar.number_input( "Promotion Usage",min_value=int(df.promotion_usage.min()),max_value=int(df.promotion_usage.max()),value=int(df.promotion_usage.median()))
age = st.sidebar.number_input("Age", int(df.age.min()), int(df.age.max()), int(df.age.mean()))
income = st.sidebar.number_input("Income", int(df.income.min()), int(df.income.max()), int(df.income.mean()))
satisfaction_score = st.sidebar.slider("Satisfaction Score", int(df.satisfaction_score.min()), int(df.satisfaction_score.max()), int(df.satisfaction_score.mean()))


# Dropdowns using friendly names
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
education = st.sidebar.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
region = st.sidebar.selectbox("Region", ["North", "South", "East", "West"])
loyalty_status = st.sidebar.selectbox("Loyalty Status", ["Bronze", "Silver", "Gold", "Platinum"])
purchase_frequency = st.sidebar.selectbox("Purchase Frequency", ["Low", "Medium", "High"])
product_category = st.sidebar.selectbox("Product Category", ["Books", "Clothing", "Food", "Electronics", "Home", "Beauty", "Health"])

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
for col, le in encoders.items():
    # Replace any unseen category with first class
    input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    input_df[col] = le.transform(input_df[col])

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
model_features = ['age', 'gender', 'income', 'education', 'region',
                  'loyalty_status', 'purchase_frequency', 'promotion_usage',
                  'satisfaction_score', 'product_category']

X_df = df[model_features].copy()

# Encode categorical safely for full dataset
for col, le in encoders.items():
    X_df[col] = X_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
    X_df[col] = le.transform(X_df[col])

y_true = df["purchase_amount"].values
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
