# Breast Cancer Classification Using Machine Learning

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
https://github.com/2026ak05184-Shilpa/ML_Assignment_2_Classification_Models/tree/main/ML_Assignment_2_Breast_Cancer

## d. Models used and evaluation metrics
The five machine learning models were evaluated using the same test dataset. 
The following six performance metrics were calculated: Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| K-Nearest Neighbors | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations about model performance

1. **Logistic Regression:** Achieved high accuracy and the highest AUC score of 0.9960, indicating excellent classification performance.

2. **Decision Tree:** Achieved an accuracy of 0.9211. It provides interpretable decision rules but performed comparatively lower than the other models.

3. **K-Nearest Neighbors:** Achieved an accuracy of 0.9561 and demonstrated good classification performance.

4. **Gaussian Naive Bayes:** Achieved perfect precision of 1.0000 but had a relatively lower recall of 0.8333.

5. **Support Vector Machine:** Achieved the highest accuracy of 0.9737, highest F1 score of 0.9630, and highest MCC of 0.9442.

### Overall Best Model

Support Vector Machine (SVM) performed best overall based on accuracy, F1 score, precision, recall, and MCC. Logistic Regression achieved the highest AUC score. Therefore, SVM is selected as the best-performing model for this dataset based on the overall evaluation results.
