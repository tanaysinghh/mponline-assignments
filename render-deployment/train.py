"""Train a Logistic Regression model on the UCI Adult Census Income dataset.

Produces three artifacts consumed by app.py:
    model.pkl           fitted LogisticRegression
    scaler.pkl          StandardScaler fitted on the numerical columns
    feature_columns.pkl ordered list of columns after one-hot encoding
"""

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/pooja2512/Adult-Census-Income/master/adult.csv"
LOCAL_CSV = "adult.csv"

NUMERICAL_FEATURES = [
    "age",
    "fnlwgt",
    "education.num",
    "capital.gain",
    "capital.loss",
    "hours.per.week",
]

CATEGORICAL_FEATURES = [
    "workclass",
    "education",
    "marital.status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native.country",
]

TARGET = "income"


def load_data():
    if os.path.exists(LOCAL_CSV):
        print(f"Loading dataset from local file: {LOCAL_CSV}")
        return pd.read_csv(LOCAL_CSV)

    print(f"Downloading dataset from: {DATA_URL}")
    df = pd.read_csv(DATA_URL)
    df.to_csv(LOCAL_CSV, index=False)
    print(f"Cached dataset to: {LOCAL_CSV}")
    return df


def clean_data(df):
    df = df.replace("?", np.nan)

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print("\nMissing values before imputation:")
        print(missing.to_string())

    for col in CATEGORICAL_FEATURES:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    print(f"\nMissing values after imputation: {int(df.isnull().sum().sum())}")
    return df


def main():
    df = load_data()
    print(f"Dataset shape: {df.shape}")

    expected = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_cols = [c for c in expected if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing expected columns: {missing_cols}")

    df = df[expected]
    df = clean_data(df)

    df[TARGET] = df[TARGET].str.strip().map({"<=50K": 0, ">50K": 1})
    if df[TARGET].isnull().any():
        raise ValueError("Unexpected values found in the income column")

    df_encoded = pd.get_dummies(
        df, columns=CATEGORICAL_FEATURES, drop_first=True, dtype=int
    )
    print(f"Shape after one-hot encoding: {df_encoded.shape}")

    X = df_encoded.drop(TARGET, axis=1)
    y = df_encoded[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape}   Test: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERICAL_FEATURES] = scaler.fit_transform(X_train[NUMERICAL_FEATURES])
    X_test_scaled[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nTest accuracy: {accuracy:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")

    print("\nSaved artifacts:")
    print(f"  model.pkl            ({len(X.columns)} input features)")
    print("  scaler.pkl")
    print("  feature_columns.pkl")


if __name__ == "__main__":
    main()
