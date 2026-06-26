from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("models/fraud_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Fraud Detection API is running!",
        "endpoints": {
            "/predict": "POST - Send transaction data to get fraud prediction"
        }
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

        return jsonify({
            "prediction": "FRAUD" if prediction == 1 else "NORMAL",
            "fraud_probability": round(float(probability) * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
