from flask import Flask, render_template, request
import pandas as pd
import pickle
import logging

try:
    logging.basicConfig(
        filename="app.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    app = Flask(__name__)

    model = pickle.load(open('../model/credit_default_pipeline.pkl', 'rb'))
    threshold = pickle.load(open('../model/threshold.pkl', 'rb'))

    preprocessor = model.named_steps["preprocessor"]
    gb_model = model.named_steps["model"]

    logging.info("Model, threshold, and SHAP explainer loaded successfully")


    @app.route('/')
    def home():
        return render_template('index.html')


    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = {
                'LIMIT_BAL': float(request.form['LIMIT_BAL']),
                'SEX': str(request.form['SEX']),
                'EDUCATION': str(request.form['EDUCATION']),
                'MARRIAGE': str(request.form['MARRIAGE']),
                'AGE': float(request.form['AGE']),
                'PAY_1': int(request.form['PAY_1']),
                'PAY_2': int(request.form['PAY_2']),
                'PAY_3': int(request.form['PAY_3']),
                'PAY_4': int(request.form['PAY_4']),
                'PAY_5': int(request.form['PAY_5']),
                'PAY_6': int(request.form['PAY_6']),
                'BILL_AMT1': float(request.form['BILL_AMT1']),
                'BILL_AMT2': float(request.form['BILL_AMT2']),
                'BILL_AMT3': float(request.form['BILL_AMT3']),
                'BILL_AMT4': float(request.form['BILL_AMT4']),
                'BILL_AMT5': float(request.form['BILL_AMT5']),
                'BILL_AMT6': float(request.form['BILL_AMT6']),
                'PAY_AMT1': float(request.form['PAY_AMT1']),
                'PAY_AMT2': float(request.form['PAY_AMT2']),
                'PAY_AMT3': float(request.form['PAY_AMT3']),
                'PAY_AMT4': float(request.form['PAY_AMT4']),
                'PAY_AMT5': float(request.form['PAY_AMT5']),
                'PAY_AMT6': float(request.form['PAY_AMT6']),
            }

            df = pd.DataFrame([data])

            # Prediction
            prob = model.predict_proba(df)[:, 1][0]
            decision = "Default Likely" if prob > threshold else "No Default"

            prediction_text = (
                f"<b>Probability of Default:</b> {prob:.2%}<br>"
                f"<b>Decision:</b> {decision}"
            )

            logging.info(f"Prediction successful: {prob:.4f}")

            logging.info(f"prediction_text: {prediction_text}")

            return render_template(
                "index.html",
                prediction_text=prediction_text,
                shap_image=True
            )

        except Exception as e:
            logging.error("Prediction failed:" + str(e), exc_info=True)
            return render_template('index.html', prediction_text=str(e))

except Exception as e:
    logging.error("Prediction failed:" + str(e), exc_info=True)

if __name__ == '__main__':
    app.run(debug=True)
