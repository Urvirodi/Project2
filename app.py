import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------------------
# Configuration
# ---------------------------
MODEL_PATH = Path("model") / "fraud_pipeline.pkl"
DATA_PATH = Path("data") / "clean_data.csv"

# ---------------------------
# Cached Loaders
# ---------------------------
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Could not load model from {MODEL_PATH}: {e}")
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv(DATA_PATH)
        return df
    except Exception as e:
        st.error(f"❌ Could not load data from {DATA_PATH}: {e}")
        return None

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("📂 Navigation")
option = st.sidebar.radio("Go to:", ["🔍 Fraud Detection", "📊 Fraud Analytics Dashboard"])

# Load resources
model = load_model()
df = load_data()

if model is None or df is None:
    st.stop()  # Halt if loading fails

st.sidebar.success("✅ Model and data loaded successfully!")

# ---------------------------
# Fraud Detection UI
# ---------------------------
if option == "🔍 Fraud Detection":
    st.title("💳 Fraud Detection System")

    col1, col2 = st.columns(2)

    with col1:
        transaction_amount = st.number_input("Transaction Amount", min_value=0.01, step=10.0, value=100.0)
        payment_method = st.selectbox("Payment Method", ["Credit Card", "Debit Card", "Net Banking", "UPI", "Wallet"])
        transaction_type = st.selectbox("Transaction Type", ["Online", "In-Store"])
        browser_type = st.selectbox("Browser Type", ["Chrome", "Firefox", "Safari", "Edge", "Other"])
        customer_gender = st.selectbox("Customer Gender", ["Male", "Female", "Other"])

    with col2:
        device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])
        customer_location = st.selectbox("Customer Location", ["Urban", "Semi-Urban", "Rural"])
        account_type = st.selectbox("Account Type", ["Savings", "Current"])
        merchant_category = st.selectbox("Merchant Category", ["Retail", "Electronics", "Travel", "Food", "Other"])
        is_international = st.selectbox("Is International Transaction?", ["No", "Yes"])

    if st.button("🚀 Predict Fraud Status"):
        # Input validation
        if transaction_amount <= 0:
            st.error("❌ Transaction amount must be greater than 0.")
            st.stop()
        
        # Prepare input data (pipeline handles encoding)
        input_data = pd.DataFrame([{
            'transaction_amount': transaction_amount,
            'payment_method': payment_method,
            'transaction_type': transaction_type,
            'browser_type': browser_type,
            'customer_gender': customer_gender,
            'device_type': device_type,
            'customer_location': customer_location,
            'account_type': account_type,
            'merchant_category': merchant_category,
            'is_international': 1 if is_international == "Yes" else 0
        }])

        with st.spinner("🔄 Predicting..."):
            try:
                prediction = model.predict(input_data)[0]
                prob = model.predict_proba(input_data)[0][1]

                if prediction == 1:
                    st.error(f"🚨 Fraud Detected! (Probability: {prob:.2%})")
                else:
                    st.success(f"✅ Genuine Transaction (Fraud Probability: {prob:.2%})")

            except Exception as e:
                st.error(f"❌ Prediction failed: {e}. Check input data or model compatibility.")

# ---------------------------
# Dashboard
# ---------------------------
elif option == "📊 Fraud Analytics Dashboard":
    st.title("📈 Fraud Analytics Dashboard")
    
    if df is not None:
        # Summary stats
        st.subheader("📊 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Transactions", len(df))
        with col2:
            fraud_rate = df['is_fraud'].mean() * 100
            st.metric("Fraud Rate (%)", f"{fraud_rate:.2f}")
        with col3:
            avg_amount = df['transaction_amount'].mean()
            st.metric("Avg Transaction Amount", f"${avg_amount:.2f}")

        # Filters
        st.subheader("🔍 Filters")
        filter_location = st.multiselect("Filter by Customer Location", df['customer_location'].unique(), default=df['customer_location'].unique())
        filtered_df = df[df['customer_location'].isin(filter_location)]

        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(filtered_df.head(10))

        # Visualizations
        st.subheader("📈 Visualizations")
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_df, x="is_fraud", palette="coolwarm", ax=ax)
        ax.set_title("Fraud vs Genuine Transactions")
        st.pyplot(fig)

        # Correlation heatmap (only numeric columns)
        numeric_df = filtered_df.select_dtypes(include='number')
        if not numeric_df.empty:
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            sns.heatmap(numeric_df.corr(), cmap="Blues", annot=True, ax=ax2)
            ax2.set_title("Correlation Heatmap")
            st.pyplot(fig2)
        else:
            st.warning("⚠️ No numeric columns available for heatmap.")
    else:
        st.error("❌ Data not available for dashboard.")
