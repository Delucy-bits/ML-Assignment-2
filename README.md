# Breast Cancer Classification — Model Comparison App

## a. Problem Statement

Breast cancer diagnosis relies on correctly distinguishing malignant tumors
from benign ones based on measurements taken from digitized images of a
fine needle aspirate (FNA) of a breast mass. This project frames diagnosis
as a **binary classification problem**: given 30 numeric features describing
cell nuclei characteristics, predict whether a tumor is **malignant** or
**benign**. Six classification models are trained on the same dataset,
evaluated with six standard metrics, and compared through an interactive
Streamlit app.

## b. Dataset Description

- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source:** UCI Machine Learning Repository (accessed via `sklearn.datasets.load_breast_cancer`,
  scikit-learn's built-in copy of the UCI dataset — also mirrored on Kaggle as
  "Breast Cancer Wisconsin (Diagnostic) Data Set")
- **Instances:** 569 (212 malignant, 357 benign)
- **Features:** 30 numeric features (mean, standard error, and "worst"/largest
  value of 10 real-valued measurements per cell nucleus — e.g. radius, texture,
  perimeter, area, smoothness, compactness, concavity, concave points,
  symmetry, fractal dimension)
- **Target:** binary — `0 = malignant`, `1 = benign`
- **Train/test split:** 80/20, stratified by class, `random_state=42`
- **Preprocessing:** features standardized with `StandardScaler` (fit on the
  training split only, then applied to the test split)

## c. GitHub Repository Link

https://github.com/Delucy-bits/ML-Assignment-2.git/tree/master

## d. Models Used

All 5 models below were trained on the same 80/20 train/test split of the
dataset described above, and evaluated on the same held-out test set
(114 instances).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| kNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Random Forest (Ensemble) | 0.9474 | 0.9937 | 0.9583 | 0.9583 | 0.9583 | 0.8869 |

### Observations

| ML Model Name                            | Observation about model performance |
|------------------------------------------|---|
| Logistic Regression                      | Best model on every single metric here. The features are standardized and this dataset is known to be close to linearly separable in this space, which plays to a linear model's strengths — low variance, no overfitting, and coefficients that stay interpretable. Only 2 misclassifications out of 114 test cases. |
| Decision Tree                            | Weakest model across the board. A single unpruned tree fits the training data almost perfectly but doesn't generalize as well — classic high-variance/overfitting behavior. It also produced the most false negatives (malignant cases predicted benign), which is the costliest error type in this domain. |
| kNN                                      | Second-strongest performer. Distance-based classification works well once features are scaled, since malignant and benign cases form fairly separable clusters in the standardized feature space. |
| Naive Bayes                              | Middling accuracy/F1 but a notably strong AUC (0.987) — its predicted probabilities rank cases well even though its "features are conditionally independent" assumption doesn't really hold here (several features like mean radius, perimeter, and area are directly derived from each other). |
| Random Forest (Ensemble)                 | Solid, well-balanced performance and a clear improvement over the single Decision Tree on every metric — a good illustration of how averaging many trees reduces variance. Its very high AUC (0.994) shows the ranking of predictions is excellent even where hard classification isn't perfect. |
| **Overall Winner for the used dataset?** | **Logistic Regression** — highest score on all 6 metrics for this dataset and split. |

> **Note:** These results come from one fixed 80/20 split (`random_state=42`).
> Rankings between Logistic Regression, kNN, and Random Forest can shift a
> little with a different split or with cross-validation — worth trying if
> you want to stress-test this conclusion.

## Project Structure

```
breast-cancer-classifier/
├── app.py                  # Streamlit app
├── requirements.txt
├── README.md
├── test_data.csv           # held-out test split (features + true labels)
└── model/
    ├── train_models.py     # training/evaluation script
    ├── train_models.ipynb  # notebook version (for lab execution/screenshot)
    ├── metrics.csv / .json # saved comparison metrics
    ├── scaler.pkl
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    └── random_forest.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model/train_models.py   # retrains models, regenerates test_data.csv
streamlit run app.py
```

## How to Use the App

1. Upload `test_data.csv` (included in this repo) using the file uploader.
2. Choose a model from the dropdown.
3. View accuracy, AUC, precision, recall, F1, and MCC, plus the confusion
   matrix and classification report for that model on the uploaded data.
