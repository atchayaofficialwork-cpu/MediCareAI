
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
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal"
]

def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found: {DATA_PATH}")
        print("Place your heart.csv file inside the data folder.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    # Support the common processed Cleveland heart CSV format.
    # If your CSV already has named columns, they are used as-is.
    if not set(FEATURES + ["target"]).issubset(df.columns):
        if df.shape[1] == 14:
            df.columns = FEATURES + ["target"]
        else:
            print("ERROR: heart.csv must contain 13 features and 1 target column.")
            print("Expected 14 columns for the standard Cleveland dataset.")
            print("Found columns:", df.columns.tolist())
            print("Shape:", df.shape)
            sys.exit(1)

    df = df[FEATURES + ["target"]].copy()

    # Convert '?' and other non-numeric values to missing values.
    for col in FEATURES + ["target"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.replace([float("inf"), float("-inf")], float("nan"))
    df = df.dropna(subset=["target"])

    # Cleveland target: 0 = no disease; 1,2,3,4 = disease present.
    df["target"] = (df["target"] > 0).astype(int)
    df = df.dropna(subset=["target"])

    if df["target"].nunique() != 2:
        print("ERROR: Target must contain both classes 0 and 1.")
        sys.exit(1)

    X = df[FEATURES]
    y = df["target"]

    if y.value_counts().min() < 2:
        print("ERROR: Not enough examples in each target class.")
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

    print("Training heart disease model...")
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "dataset": "Cleveland heart disease dataset",
        "rows_used": int(len(df)),
        "features": FEATURES,
        "target_definition": "0 = dataset target negative; 1 = target positive",
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
            y_test, predictions, zero_division=0, output_dict=True
        ),
        "warning": (
            "Educational prototype only. Not clinically validated and "
            "not suitable for diagnosis or treatment decisions."
        )
    }

    model_path = MODEL_DIR / "heart_model.joblib"
    report_path = REPORT_DIR / "heart_metrics.json"

    joblib.dump({
        "pipeline": pipeline,
        "features": FEATURES,
        "target_definition": "0 = target negative, 1 = target positive"
    }, model_path)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    print("\nModel training completed.")
    print("Model saved to:", model_path)
    print("Metrics saved to:", report_path)
    print("\nEvaluation metrics:")
    for name in [
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1_score", "roc_auc"
    ]:
        print(f"{name}: {metrics[name]}")

    print("\nConfusion matrix:")
    print(metrics["confusion_matrix"])
    print("\nReminder: These metrics are for an educational dataset,")
    print("not evidence of clinical accuracy.")

if __name__ == "__main__":
    main()
