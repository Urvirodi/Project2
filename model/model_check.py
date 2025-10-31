import joblib
import pandas as pd

# Load saved model pipeline
model_path = "model/fraud_pipeline.pkl"

try:
    model = joblib.load(model_path)
    print("✅ Model pipeline loaded successfully!")
except Exception as e:
    print("❌ Error loading model pipeline:", e)
    quit()

# Test sample (same features used in training)
sample_data = pd.DataFrame([{
    "transaction_amount": 5000,
    "payment_method": "Credit Card",
    "transaction_type": "Online",
    "browser_type": "Chrome",
    "customer_gender": "Male",
    "device_type": "Mobile",
    "customer_location": "Urban",
    "account_type": "Savings",
    "merchant_category": "Retail",
    "is_international": 0
}])

print("\n🧪 Sample Input:")
print(sample_data)

try:
    prediction = model.predict(sample_data)[0]
    prob = model.predict_proba(sample_data)[0][1]

    print("\n✅ Prediction Successful!")
    print(f"Prediction: {'Fraud' if prediction == 1 else 'Not Fraud'}")
    print(f"Probability of Fraud: {prob:.2%}")

except Exception as e:
    print("❌ Prediction error:", e)
