
from pathlib import Path
import json
import sys

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "diabetes.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

TARGET = "Outcome"


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        print("Place diabetes.csv inside the data folder.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    required_columns = FEATURES + [TARGET]
    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("ERROR: Required columns are missing:", missing_columns)
        print("CSV columns found:", df.columns.tolist())
        sys.exit(1)

    df = df[required_columns].copy()

    # Convert invalid text values to missing values.
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=[TARGET])
    df = df[df[TARGET].isin([0, 1])]
    df[TARGET] = df[TARGET].astype(int)

    # In this dataset, zero often represents missing measurements
    # for these particular medical features.
    zero_is_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    for column in zero_is_missing:
        df.loc[df[column] == 0, column] = float("nan")

    if df[TARGET].nunique() != 2:
        print("ERROR: Outcome must contain both labels 0 and 1.")
        sys.exit(1)

    X = df[FEATURES]
    y = df[TARGET]

    if y.value_counts().min() < 2:
        print("ERROR: Not enough records in each class.")
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training diabetes model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "Pima Indians Diabetes dataset",
        "rows_used": int(len(df)),
        "features": FEATURES,
        "target": TARGET,
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_test, predictions)), 4
        ),
        "precision": round(
            float(precision_score(y_test, predictions, zero_division=0)), 4
        ),
        "recall": round(
            float(recall_score(y_test, predictions, zero_division=0)), 4
        ),
        "f1_score": round(
            float(f1_score(y_test, predictions, zero_division=0)), 4
        ),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            zero_division=0,
            output_dict=True
        ),
        "warning": (
            "Educational model only. Not clinically validated and "
            "not suitable for diagnosis or treatment decisions."
        )
    }

    model_path = MODEL_DIR / "diabetes_model.joblib"
    report_path = REPORT_DIR / "diabetes_metrics.json"

    joblib.dump({
        "pipeline": pipeline,
        "features": FEATURES,
        "target": TARGET
    }, model_path)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nDiabetes model training completed.")
    print("Model saved to:", model_path)
    print("Metrics saved to:", report_path)

    print("\nEvaluation metrics:")
    for name in [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]:
        print(f"{name}: {metrics[name]}")

    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])

    print("\nReminder: Educational dataset results only; not a diagnosis.")


if __name__ == "__main__":
    main()
