# Breast Cancer Classification – ML Assignment 2

## a. Problem statement
Build and compare five supervised classification models to classify breast tumour observations as malignant (M) or benign (B), and provide an interactive Streamlit application for evaluating saved models on labelled test data.

## b. Dataset description
- Dataset: Breast Cancer Wisconsin (Diagnostic)
- Source: UCI Machine Learning Repository
- Instances: 569
- Input features: 30 numeric features
- Target: diagnosis (M = malignant, B = benign)
- Task: Binary classification
- Data source page: https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic
- The dataset features are computed from digitized fine needle aspirate images and describe cell-nuclei characteristics.

## c. GitHub Repository Link
Replace this line with your own repository URL after uploading:
`[PASTE YOUR GITHUB REPOSITORY URL HERE]`

## d. Models used and evaluation metrics
The five models are Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes and Support Vector Machine.

Run `python train_models.py` to generate the results. Copy the actual values from `metrics_summary.csv` into the table below.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

### Observations on model performance
Write observations using the actual metric values after running the experiment. Compare:
- Accuracy: overall proportion of correct predictions.
- AUC: ability to distinguish malignant and benign cases across thresholds.
- Precision: proportion of predicted malignant cases that are actually malignant.
- Recall: proportion of actual malignant cases correctly identified.
- F1: harmonic mean of precision and recall.
- MCC: balanced correlation measure using all confusion-matrix categories.

| Model | Observation |
|---|---|
| Logistic Regression | Complete after reviewing the measured metrics. |
| Decision Tree | Complete after reviewing the measured metrics. |
| KNN | Complete after reviewing the measured metrics. |
| Naive Bayes | Complete after reviewing the measured metrics. |
| Support Vector Machine | Complete after reviewing the measured metrics. |

**Overall winner:** Select the model based on the metric priorities and actual test results. For this medical classification example, discuss malignant-class recall and AUC alongside the other metrics rather than relying on accuracy alone.

## Streamlit app
The app supports CSV upload, model selection, six evaluation metrics, confusion matrix and classification report. It loads saved models and does not retrain on the uploaded test data.

After deployment, paste the live app URL here:
`[PASTE YOUR STREAMLIT APP URL HERE]`

## How to run
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python train_models.py`
3. Launch the app: `streamlit run app.py`
4. Upload `test_data.csv` in the app.

## Reproducibility
The train/test split uses `test_size=0.20`, `random_state=42`, and stratification by target. StandardScaler is included in the Logistic Regression, KNN and SVM pipelines to ensure preprocessing is applied consistently at training and inference time.
