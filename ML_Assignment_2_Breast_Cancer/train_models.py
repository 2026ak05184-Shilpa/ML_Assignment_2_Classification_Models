"""Train five classifiers and save models + a held-out test CSV.

Run from the project root:
    python train_models.py

The dataset file is data/dataset.csv. The target column is diagnosis:
M = malignant (positive class), B = benign (negative class).
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)
target_col = "diagnosis"
X = df.drop(columns=[target_col])
y = (df[target_col].astype(str).str.upper() == "M").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": (Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=5000, random_state=42))
    ]), "logistic_regression.joblib"),
    "Decision Tree": (DecisionTreeClassifier(random_state=42, max_depth=5), "decision_tree.joblib"),
    "K-Nearest Neighbors": (Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]), "knn.joblib"),
    "Gaussian Naive Bayes": (GaussianNB(), "naive_bayes.joblib"),
    "Support Vector Machine": (Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ]), "svm.joblib")
}

metric_rows, model_files = [], {}
for name, (model, filename) in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metric_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    })
    joblib.dump(model, MODEL_DIR / filename)
    model_files[name] = filename

test_output = X_test.copy()
test_output[target_col] = ["M" if value == 1 else "B" for value in y_test]
test_output.to_csv(ROOT / "test_data.csv", index=False)
pd.DataFrame(metric_rows).to_csv(ROOT / "metrics_summary.csv", index=False)

metadata = {
    "target_column": target_col,
    "positive_class": "M",
    "negative_class": "B",
    "feature_columns": list(X.columns),
    "model_files": model_files
}
(ROOT / "model_metadata.json").write_text(json.dumps(metadata, indent=2))
print("Training complete. Models, metrics_summary.csv and test_data.csv are ready.")
print(pd.DataFrame(metric_rows).round(4).to_string(index=False))
