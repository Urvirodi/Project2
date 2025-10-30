import joblib
import pandas as pd
from pathlib import Path

print("--- Checking Fraud Detection Model from Model directory ---", flush=True)

# Define correct model path
model_path = Path("model") / "fraud_pipeline.pkl"

try:
    # Load the trained fraud detection pipeline
    model = joblib.load(model_path)
    print(f"✅ model loaded successfully from: {model_path}", flush=True)
    print(f"model type: {type(model)}", flush=True)

    # Create a small sample transaction for testing
    sample_data = pd.DataFrame({
        'transaction_amount': [15000],
        'payment_method': ['Credit Card'],
        'transaction_type': ['Online'],
        'browser_type': ['Chrome'],
        'customer_gender': ['Female'],
        'device_type': ['Mobile'],
        'customer_location': ['Urban'],
        'account_type': ['Savings'],
        'merchant_category': ['Electronics'],
        'is_international': [0]
    })

    # Run prediction
    prediction = model.predict(sample_data)[0]
    prob = model.predict_proba(sample_data)[0][1]

    print(f"✅ Prediction completed successfully!", flush=True)
    print(f"Predicted Class: {'Fraud' if prediction == 1 else 'Genuine'}", flush=True)
    print(f"Fraud Probability: {prob:.2%}", flush=True)

except FileNotFoundError:
    print(f"❌ model file not found at: {model_path}", flush=True)
    models_dir = Path("model")
    if models_dir.exists():
        print("Available files in model directory:", flush=True)
        for file in models_dir.iterdir():
            print(f"  - {file.name}", flush=True)
    else:
        print("❌ Model directory does not exist!", flush=True)

except Exception as e:
    print(f"❌ Error loading or testing model: {e}", flush=True)

print("Model check completed successfully ✅", flush=True)