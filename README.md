# Online Shoppers Purchasing Intention — Classification Models

**BITS Pilani WILP — M.Tech (AIML) — Machine Learning Assignment 2**
Student ID: 2026AK05036

---

## a. Problem statement

E-commerce websites receive a large volume of browsing sessions, but only a
small fraction end in a purchase. The goal of this project is to predict, from
a visitor's session behaviour and page analytics, whether that session will
end in a transaction (`Revenue` = True) or not.

This is a **binary classification** problem. Five classification models are
trained on the same dataset, compared using six evaluation metrics, and
exposed through an interactive Streamlit web application that scores an
uploaded test file using the pre-trained models.

---

## b. Dataset description

| Item | Value |
|---|---|
| Name | Online Shoppers Purchasing Intention Dataset |
| Source | UCI Machine Learning Repository (dataset id 468) |
| Link | https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset |
| Instances | 12,330 sessions |
| Input features | 17 |
| Target column | `Revenue` (binary: True = purchase, False = no purchase) |
| Missing values | 0 |
| Class balance | approximately 84.5% False / 15.5% True |
| Assignment minimum (500 instances) | PASS |
| Assignment minimum (12 features) | PASS |

**Feature groups**

- *Page-visit counts and durations* — `Administrative`, `Administrative_Duration`,
  `Informational`, `Informational_Duration`, `ProductRelated`,
  `ProductRelated_Duration`
- *Google Analytics metrics* — `BounceRates`, `ExitRates`, `PageValues`
- *Temporal / campaign* — `SpecialDay`, `Month`, `Weekend`
- *Technical and visitor profile* — `OperatingSystems`, `Browser`, `Region`,
  `TrafficType`, `VisitorType`

The dataset is notably **imbalanced**, which is important when interpreting
the results: a model can reach high accuracy simply by predicting "no
purchase" most of the time, so Recall, F1 and MCC are more informative than
accuracy alone.

---

## c. GitHub repository link

https://github.com/2026ak05036/ML-Assignment-2-2026AK05036

**Repository contents**

```
├── app.py                 Streamlit application
├── train_models.py        Trains and saves all five models
├── requirements.txt       Dependencies
├── README.md              This file
├── test_data.csv          Held-out 20% test split (features + target)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metrics.json
```

---

## Machine learning methodology

**Preprocessing.** All preprocessing is contained inside a scikit-learn
`Pipeline`, so that each saved `.joblib` file holds the preprocessing *and*
the classifier together:

- `Month` and `VisitorType` → one-hot encoded (`handle_unknown='ignore'`)
- all remaining numeric columns → `StandardScaler`
- `Revenue` and `Weekend` booleans → converted to 0/1

Because preprocessing sits inside the pipeline, `fit()` learns the scaler and
encoder from the **training split only**. There is no data leakage from the
test set, and the Streamlit app cannot accidentally re-fit anything.

**Split.** 80% train / 20% test, `random_state=42`, stratified on the target
to preserve the class imbalance in both halves.

**Models.** Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian
Naive Bayes, and Support Vector Machine — all five trained on the same
dataset and the same split. `SVC` uses `probability=True` so that predicted
probabilities are available for the AUC calculation.

**Metrics.** Accuracy, AUC, Precision, Recall, F1 and MCC, all computed on
the held-out test split.

---

## d. Models used — comparison table

All five models were trained on the same dataset and the same 80/20
stratified split. Metrics are measured on the held-out test set of 2,466
sessions.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8812 | **0.8877** | **0.7432** | 0.3560 | 0.4814 | 0.4603 |
| Decision Tree | 0.8569 | 0.7282 | 0.5377 | 0.5419 | 0.5398 | 0.4550 |
| KNN | 0.8674 | 0.7829 | 0.6170 | 0.3796 | 0.4700 | 0.4145 |
| Naive Bayes | 0.6736 | 0.7939 | 0.2941 | **0.7906** | 0.4287 | 0.3249 |
| Support Vector Machine | **0.8861** | 0.8726 | 0.6996 | 0.4634 | **0.5575** | **0.5090** |

Bold marks the best score in each column.

**Reference point.** 84.51% of the test sessions do not end in a purchase,
so a model that always predicted "no purchase" would score 0.8451 accuracy
while being completely useless. Accuracy must therefore be read against that
baseline, and F1 and MCC are the more meaningful measures here.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This model has the highest AUC (0.8877) and precision (0.7432). It predicts purchases carefully, but its recall is only 0.3560, so it misses many actual purchases. |
| Decision Tree | Precision (0.5377) and recall (0.5419) are nearly equal. Its F1 score is 0.5398, but its AUC (0.7282) is the lowest among the five models. |
| KNN | Accuracy is 0.8674, but recall is only 0.3796. Since most sessions are non-purchases, nearby examples may lead the model to predict the majority class more often. |
| Naive Bayes | This model has the highest recall (0.7906), so it identifies many actual purchases. However, its precision is only 0.2941, meaning it also incorrectly predicts many purchases. |
| Support Vector Machine | SVM achieves the highest accuracy (0.8861), F1 score (0.5575), and MCC (0.5090). It provides the strongest overall balance of results among these five models. |
| **Overall winner for your dataset?** | **Support Vector Machine (SVM)**. I selected SVM because it has the best F1 and MCC scores, which are useful for this imbalanced dataset. Logistic Regression has a higher AUC, but SVM performs better overall at the current classification threshold. |

---

## Installation and running locally

```bash
git clone https://github.com/2026ak05036/ML-Assignment-2-2026AK05036.git
cd ML-Assignment-2-2026AK05036
pip install -r requirements.txt

# Train the five models (creates model/ and test_data.csv)
python train_models.py

# Launch the web application
streamlit run app.py
```

The training dataset itself is not committed to the repository. Download it
from the UCI link above and save it in the project folder as
`online_shoppers_intention.csv` before running `train_models.py`.

---

## Streamlit application features

| Feature | Description |
|---|---|
| Dataset upload | Upload a test CSV containing the input features and the `Revenue` target column |
| Model selection | Dropdown to choose any of the five trained models |
| Evaluation metrics | Displays Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model |
| Confusion matrix | Rendered as a labelled heatmap |
| Classification report | Per-class precision, recall and F1 |
| All-model comparison | Every one of the five models scored on the same uploaded file |
| Validation | Clear error messages for unreadable files, a missing target column, or mismatched feature columns |

The app **loads the saved models and does not retrain** on uploaded data.

**Live app:** https://ml-assignment-2-2026ak05036.streamlit.app/

---

## Reproducibility

`random_state=42` is fixed for the train/test split, the Decision Tree, the
Logistic Regression solver and the SVM, so repeated runs produce identical
results.
