from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle

app = Flask(__name__)
CORS(app)

# Load your trained pipeline (preprocessor + model)
with open("../model/bank_cc_defaults_model.pkl", "rb") as f:
    model = pickle.load(f)

# Prediction threshold
THRESHOLD = 0.5  # probability > 0.5 -> Not Approved

# Features in exact order used during training
FEATURES_ORDER = [
    'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2',
    'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 'BILL_AMT1', 'BILL_AMT2',
    'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1',
    'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
    'avg_bill_amount', 'avg_payment_amount'
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Compute derived features
        data["avg_bill_amount"] = (
            float(data["BILL_AMT1"]) + float(data["BILL_AMT2"]) +
            float(data["BILL_AMT3"]) + float(data["BILL_AMT4"]) +
            float(data["BILL_AMT5"]) + float(data["BILL_AMT6"])
        ) / 6.0

        data["avg_payment_amount"] = (
            float(data["PAY_AMT1"]) + float(data["PAY_AMT2"]) +
            float(data["PAY_AMT3"]) + float(data["PAY_AMT4"]) +
            float(data["PAY_AMT5"]) + float(data["PAY_AMT6"])
        ) / 6.0

        # Convert to DataFrame and reorder columns exactly
        df = pd.DataFrame([data])
        df = df[FEATURES_ORDER]

        # Predict probability and apply threshold
        probability = model.predict_proba(df)[0][1]
        default_prediction = int(probability > THRESHOLD)

        # Print log to server
        print("\n--- New Prediction Request ---")
        print("Input Data:", data)
        print("Predicted Probability:", round(float(probability), 3))
        print("Prediction (0=Approved, 1=Not Approved):", default_prediction)
        print("Threshold Used:", THRESHOLD)
        print("-----------------------------\n")

        # Return to frontend
        response = {
            "default_prediction": default_prediction,
            "default_probability": round(float(probability), 3),
            "threshold": THRESHOLD
        }
        return jsonify(response), 200

    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
