"""Flask service that serves the Adult Census Income Logistic Regression model.

Routes:
    GET  /         HTML form
    POST /         HTML form submission, re-renders with the prediction
    POST /predict  JSON API
    GET  /health   health check for Render
"""

import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from train import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
FEATURE_COLUMNS_PATH = "feature_columns.pkl"

LABELS = {0: "<=50K", 1: ">50K"}

NUMERIC_RANGES = {
    "age": (17, 90),
    "fnlwgt": (1, 2_000_000),
    "education.num": (1, 16),
    "capital.gain": (0, 100_000),
    "capital.loss": (0, 5_000),
    "hours.per.week": (1, 99),
}

CATEGORIES = {
    "workclass": [
        "Private",
        "Self-emp-not-inc",
        "Self-emp-inc",
        "Federal-gov",
        "Local-gov",
        "State-gov",
        "Without-pay",
        "Never-worked",
    ],
    "education": [
        "Preschool",
        "1st-4th",
        "5th-6th",
        "7th-8th",
        "9th",
        "10th",
        "11th",
        "12th",
        "HS-grad",
        "Some-college",
        "Assoc-voc",
        "Assoc-acdm",
        "Bachelors",
        "Masters",
        "Prof-school",
        "Doctorate",
    ],
    "marital.status": [
        "Married-civ-spouse",
        "Never-married",
        "Divorced",
        "Separated",
        "Widowed",
        "Married-spouse-absent",
        "Married-AF-spouse",
    ],
    "occupation": [
        "Prof-specialty",
        "Exec-managerial",
        "Adm-clerical",
        "Sales",
        "Craft-repair",
        "Tech-support",
        "Other-service",
        "Machine-op-inspct",
        "Transport-moving",
        "Handlers-cleaners",
        "Farming-fishing",
        "Protective-serv",
        "Priv-house-serv",
        "Armed-Forces",
    ],
    "relationship": [
        "Husband",
        "Wife",
        "Own-child",
        "Not-in-family",
        "Other-relative",
        "Unmarried",
    ],
    "race": [
        "White",
        "Black",
        "Asian-Pac-Islander",
        "Amer-Indian-Eskimo",
        "Other",
    ],
    "sex": ["Male", "Female"],
    "native.country": [
        "United-States",
        "Mexico",
        "Philippines",
        "Germany",
        "Canada",
        "Puerto-Rico",
        "El-Salvador",
        "India",
        "Cuba",
        "England",
        "Jamaica",
        "South",
        "China",
        "Italy",
        "Dominican-Republic",
        "Vietnam",
        "Guatemala",
        "Japan",
        "Poland",
        "Columbia",
        "Taiwan",
        "Haiti",
        "Iran",
        "Portugal",
        "Nicaragua",
        "Peru",
        "France",
        "Greece",
        "Ecuador",
        "Ireland",
        "Hong",
        "Cambodia",
        "Trinadad&Tobago",
        "Laos",
        "Thailand",
        "Yugoslavia",
        "Outlying-US(Guam-USVI-etc)",
        "Hungary",
        "Honduras",
        "Scotland",
        "Holand-Netherlands",
    ],
}

DEFAULTS = {
    "age": "39",
    "workclass": "Private",
    "fnlwgt": "77516",
    "education": "HS-grad",
    "education.num": "9",
    "marital.status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital.gain": "0",
    "capital.loss": "0",
    "hours.per.week": "40",
    "native.country": "United-States",
}

app = Flask(__name__)


def load_artifacts():
    for path in (MODEL_PATH, SCALER_PATH, FEATURE_COLUMNS_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing artifact '{path}'. Run 'python train.py' before starting the app."
            )
    return (
        joblib.load(MODEL_PATH),
        joblib.load(SCALER_PATH),
        joblib.load(FEATURE_COLUMNS_PATH),
    )


model, scaler, feature_columns = load_artifacts()


class InvalidInput(Exception):
    pass


def normalise_keys(payload):
    aliases = {key.replace(".", "_"): key for key in DEFAULTS}
    cleaned = {}
    for key, value in payload.items():
        cleaned[aliases.get(key, key)] = value
    return cleaned


def parse_numeric(field, raw):
    if raw is None or str(raw).strip() == "":
        raise InvalidInput(f"Missing value for '{field}'")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise InvalidInput(f"'{field}' must be a number, got '{raw}'")

    low, high = NUMERIC_RANGES[field]
    if not low <= value <= high:
        raise InvalidInput(f"'{field}' must be between {low} and {high}, got {value:g}")
    return value


def parse_categorical(field, raw):
    if raw is None or str(raw).strip() == "":
        raise InvalidInput(f"Missing value for '{field}'")
    value = str(raw).strip()
    if value not in CATEGORIES[field]:
        raise InvalidInput(
            f"'{field}' must be one of {CATEGORIES[field]}, got '{value}'"
        )
    return value


def build_features(payload):
    payload = normalise_keys(payload)

    row = {}
    for field in NUMERICAL_FEATURES:
        row[field] = parse_numeric(field, payload.get(field))
    for field in CATEGORICAL_FEATURES:
        row[field] = parse_categorical(field, payload.get(field))

    frame = pd.DataFrame([row])
    frame = pd.get_dummies(frame, columns=CATEGORICAL_FEATURES, dtype=int)
    frame = frame.reindex(columns=feature_columns, fill_value=0)
    frame[NUMERICAL_FEATURES] = scaler.transform(frame[NUMERICAL_FEATURES])
    return frame


def predict(payload):
    features = build_features(payload)
    probability = float(model.predict_proba(features)[0][1])
    label = LABELS[int(probability >= 0.5)]
    confidence = probability if label == ">50K" else 1.0 - probability
    return {
        "prediction": label,
        "probability": round(probability, 4),
        "confidence": round(confidence, 4),
    }


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", form=DEFAULTS, result=None, error=None)

    submitted = {key: request.form.get(key, "") for key in DEFAULTS}
    try:
        result = predict(submitted)
    except InvalidInput as exc:
        return (
            render_template(
                "index.html", form=submitted, result=None, error=str(exc)
            ),
            400,
        )

    return render_template("index.html", form=submitted, result=result, error=None)


@app.route("/predict", methods=["POST"])
def predict_api():
    payload = request.get_json(silent=True)
    if payload is None:
        payload = request.form.to_dict() or None
    if not isinstance(payload, dict) or not payload:
        return (
            jsonify(
                {
                    "error": "Request body must be a non-empty JSON object with Content-Type: application/json"
                }
            ),
            400,
        )

    try:
        return jsonify(predict(payload))
    except InvalidInput as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.errorhandler(404)
def not_found(_error):
    if request.path.startswith("/predict"):
        return jsonify({"error": "Not found"}), 404
    return render_template("index.html", form=DEFAULTS, result=None, error=None), 404


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
