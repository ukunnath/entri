# Credit Card Default Prediction

## 📋 Project Overview

This project predicts the probability of credit card default for a customer using historical demographic, billing, and payment behavior data.

**Key Features:**
- Exploratory Data Analysis & Feature Engineering
- Model Comparison Study
- Production-Ready Model Development
- One-Hot Encoding via ColumnTransformer
- Threshold Tuning for Optimal Performance
- Flask Web Application for Inference
- Comprehensive Logging & Error Handling

---

## 🎯 Problem Statement

**Given customer details such as:**
- Credit limit
- Demographics (age, gender, education, marital status)
- Past payment history
- Bill amounts
- Payment amounts

**Predict:**
> Will the customer default on their credit card payment?

---

## 🗂️ Project Structure

```
credit-default-prediction/com/entri/minipro
│
├── model/
│   ├── credit_default_pipeline.pkl   # Trained pipeline (preprocessing + model)
│   └── threshold.pkl                 # Optimized classification threshold
│
├── ui/
│   ├── app.py                        # Flask inference app
│   ├── templates/
│   │   └── index.html                # HTML UI
│   └── static/
│       └── shap.png                  # SHAP explanation plot
│
├── training/
│   └── train_model.py                # Model training script
│
├── data/
│   └── credit_default.csv            # Dataset
│
├── requirements.txt
└── README.md
```

---

## 📊 Dataset Description

**Dataset:** Default of Credit Card Clients

**Target Variable:**
- `default.payment.next.month`
  - `1` → Default
  - `0` → No Default

### Input Features

| Feature | Description |
|---------|-------------|
| `LIMIT_BAL` | Credit limit |
| `SEX` | Gender |
| `EDUCATION` | Education level |
| `MARRIAGE` | Marital status |
| `AGE` | Age |
| `PAY_1` to `PAY_6` | Repayment status (last 6 months) |
| `BILL_AMT1` to `BILL_AMT6` | Bill amount (last 6 months) |
| `PAY_AMT1` to `PAY_AMT6` | Payment amount (last 6 months) |

---

## ⚙️ Model Pipeline

The entire workflow is encapsulated in a single **sklearn Pipeline**:

### Preprocessing
- **OneHotEncoding** for categorical variables
- **StandardScaler** for numeric variables
- `handle_unknown='ignore'` to safely handle new categories

### Model
- **GradientBoostingClassifier**

**Why Gradient Boosting?**
- ✅ Handles non-linear relationships
- ✅ Robust to feature interactions
- ✅ Performs well on structured/tabular data

---

## 🎯 Threshold Tuning

Instead of using the default 0.5 threshold, we:

1. Compute predicted probabilities
2. Evaluate multiple thresholds
3. Choose threshold that maximizes business metric (ROC-AUC / F1 / Recall)

**Saved as:** `model/threshold.pkl`

---

## 🚀 Flask Inference App

### Key Characteristics
- Uses same pipeline as training
- No manual encoding needed
- Safe numeric conversion
- Proper error handling
- Logging enabled

### Prediction Flow
1. User submits form
2. Data converted to DataFrame
3. Pipeline handles preprocessing
4. Model predicts probability
5. Threshold applied
6. SHAP explanation generated
7. Result rendered on UI

---

## 🌐 UI Features

- Clean HTML form
- POST request to Flask backend
- **Displays:**
  - Probability of default
  - Approval / Rejection decision

---

## 🧪 Running Locally

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Linux / Mac
venv\Scripts\activate      # Windows
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Train Model
```bash
python training/train_model.py
```

### 4️⃣ Start Flask App
```bash
cd ui
python app.py
```

### 5️⃣ Open Browser
```
http://127.0.0.1:5000
```

---

## 📦 Requirements

```txt
flask
pandas
numpy
scikit-learn
shap
matplotlib
joblib
```

---

## 📝 Logging

Logs are written to: `ui/app.log`

**Includes:**
- ✅ Successful predictions
- ⚠️ Input validation errors
- ⚠️ SHAP generation issues
- ❌ Unexpected failures

---

## 🛡️ Production-Safe Design Decisions

| Feature | Implementation |
|---------|----------------|
| ✅ Single pipeline | No mismatch between train & inference |
| ✅ No manual encoding | Flask uses the same ColumnTransformer |
| ✅ Non-GUI Matplotlib backend | Uses `Agg` for server environments |
| ✅ SHAP explainer | Recreated at runtime for consistency |
| ✅ Threshold storage | Stored externally for easy updates |
| ✅ Feature order | Preserved throughout pipeline |

---

## 📈 Model Performance

*(Add your model metrics here after training)*

Example:
- **Accuracy:** 82%
- **Precision:** 68%
- **Recall:** 73%
- **ROC-AUC:** 0.78

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is licensed under the MIT License.

---

## 👤 Author

**ukunnath**
- GitHub: [@ukunnath](https://github.com/ukunnath)

---

## 🙏 Acknowledgments

- Dataset: [UCI Machine Learning Repository - Default of Credit Card Clients](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- SHAP for model explainability
- Scikit-learn for ML pipeline