"""
train_models.py
----------------
Trains and evaluates 5 classification models on the Breast Cancer Wisconsin
(Diagnostic) dataset, saves the trained models + scaler, and writes out:

  - ../test_data.csv        (held-out test split, used by the Streamlit app)
  - metrics.csv / metrics.json (comparison table source data)
  - *.pkl                   (trained models + the fitted StandardScaler)

Dataset source: sklearn.datasets.load_breast_cancer(), which is scikit-learn's
built-in copy of the UCI ML Repository "Breast Cancer Wisconsin (Diagnostic)"
dataset — 569 instances, 30 numeric features, binary target
(0 = malignant, 1 = benign).
"""

from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def main():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    test_df = X_test.copy()
    test_df["target"] = y_test.values
    test_df.to_csv(PROJECT_ROOT / "test_data.csv", index=False)

    models = {
        "logistic_regression": LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "naive_bayes": GaussianNB(),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    }
    display_names = {
        "logistic_regression": "Logistic Regression",
        "decision_tree": "Decision Tree",
        "knn": "kNN",
        "naive_bayes": "Naive Bayes",
        "random_forest": "Random Forest (Ensemble)",
    }

    results = {}
    for key, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        proba = model.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, preds),
            "AUC": roc_auc_score(y_test, proba),
            "Precision": precision_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1": f1_score(y_test, preds),
            "MCC": matthews_corrcoef(y_test, preds),
        }
        results[display_names[key]] = metrics
        joblib.dump(model, BASE_DIR / f"{key}.pkl")

    joblib.dump(scaler, BASE_DIR / "scaler.pkl")

    results_df = pd.DataFrame(results).T
    results_df.index.name = "ML Model Name"
    results_df = results_df.round(4)
    results_df.to_csv(BASE_DIR / "metrics.csv")
    with open(BASE_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(results_df.to_string())


if __name__ == "__main__":
    main()
