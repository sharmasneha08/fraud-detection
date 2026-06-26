import joblib
import numpy as np
import pandas as pd

def load_model(model_path="models/fraud_model.pkl"):
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model

def predict_transaction(model, transaction_data):
    """
    transaction_data: dict or DataFrame of a single transaction
    """
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
    else:
        df = transaction_data

    prediction = model.predict(df)
    probability = model.predict_proba(df)[:, 1]

    result = {
        "prediction": "FRAUD" if prediction[0] == 1 else "NORMAL",
        "fraud_probability": round(float(probability[0]) * 100, 2)
    }

    print(f"Prediction : {result['prediction']}")
    print(f"Fraud Probability: {result['fraud_probability']}%")
    return result
