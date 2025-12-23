import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_curve


try:
    df = pd.read_csv("../../../../resource/source/cleaned_credit_defaulter.csv")

    target = 'default.payment.next.month'

    X = df.drop(columns=[target])
    y = df[target]

    X = X.rename(columns={'PAY_0': 'PAY_1'})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cat_cols = ['SEX', 'MARRIAGE', 'EDUCATION']

    num_cols = [
        'AGE', 'LIMIT_BAL',
        'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
            ('num', StandardScaler(), num_cols)
        ]
    )

    gb_model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('model', gb_model)
        ]
    )

    pipeline.fit(X_train, y_train)

    # =====================================================
    # Threshold tuning
    # =====================================================
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    best_threshold = thresholds[np.argmax(tpr - fpr)]

    with open("credit_default_pipeline.pkl", "wb") as f:
        pickle.dump(pipeline, f)

    with open("threshold.pkl", "wb") as f:
        pickle.dump(best_threshold, f)

    print("Model Build completed successfully")

except Exception as e:
    print(e)