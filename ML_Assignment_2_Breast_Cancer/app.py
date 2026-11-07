from pathlib import Path
import json
import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent
META = json.loads((ROOT / "model_metadata.json").read_text())

st.set_page_config(page_title="Breast Cancer Classifier Comparison", page_icon="🧪", layout="wide")
st.title("Breast Cancer Classification – Model Evaluation")
st.write(
    "Compare five trained machine-learning classifiers on a labelled test CSV. "
    "The application loads saved models; it does not retrain them on uploaded data."
)
st.info("Dataset: UCI Breast Cancer Wisconsin (Diagnostic). M = malignant (positive class); B = benign.")

model_names = list(META["model_files"].keys())
selected_model = st.selectbox("Choose a classification model", model_names)

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])
if uploaded is None:
    st.warning("Upload test_data.csv from the project folder to view evaluation results.")
    st.caption("The CSV must contain all 30 feature columns and the diagnosis target column.")
    st.stop()

try:
    test_df = pd.read_csv(uploaded)
    target = META["target_column"]
    required = META["feature_columns"] + [target]
    missing = [col for col in required if col not in test_df.columns]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        st.stop()

    test_df = test_df[required].dropna()
    if test_df.empty:
        st.error("The uploaded CSV has no complete rows after removing missing values.")
        st.stop()

    X_test = test_df[META["feature_columns"]]
    y_raw = test_df[target].astype(str).str.upper().str.strip()
    valid_labels = {"M", "B"}
    if not set(y_raw.unique()).issubset(valid_labels):
        st.error("The diagnosis column must contain only M (malignant) or B (benign).")
        st.stop()
    y_test = (y_raw == "M").astype(int)

    model_path = ROOT / "models" / META["model_files"][selected_model]
    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, probabilities) if y_test.nunique() == 2 else float("nan"),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions)
    }

    st.subheader(f"Evaluation results: {selected_model}")
    cols = st.columns(3)
    for col, (label, value) in zip(cols * 2, metrics.items()):
        col.metric(label, f"{value:.4f}" if pd.notna(value) else "N/A")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, predictions, labels=[0, 1])
        st.dataframe(pd.DataFrame(
            cm, index=["Actual B", "Actual M"], columns=["Predicted B", "Predicted M"]
        ), use_container_width=True)
    with right:
        st.subheader("Classification Report")
        report = classification_report(
            y_test, predictions, labels=[0, 1],
            target_names=["Benign (B)", "Malignant (M)"],
            output_dict=True, zero_division=0
        )
        st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

    with st.expander("Preview uploaded test data"):
        st.dataframe(test_df.head(10), use_container_width=True)
    st.caption(f"Evaluated {len(test_df)} rows. No model training was performed during upload.")
except Exception as exc:
    st.error(f"Could not evaluate the uploaded CSV: {exc}")
