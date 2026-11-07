"""
ML Assignment 2 - Streamlit application
Online Shoppers Purchasing Intention - classification model demonstrator

The app LOADS pre-trained pipelines from model/ and scores an uploaded test
CSV. It never trains anything.

Run locally:  streamlit run app.py
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

MODEL_DIR = "model"
TARGET = "Revenue"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="Online Shoppers Intention Classifier",
                   layout="wide")


@st.cache_resource
def load_model(filename):
    """Load one saved pipeline. Cached so it is read from disk only once."""
    return joblib.load(os.path.join(MODEL_DIR, filename))


def positive_scores(pipeline, X):
    """Score for the positive class, used for AUC.

    SVC is saved without probability=True (deprecated in scikit-learn 1.9
    and much slower), so its decision_function values are used instead.
    AUC depends only on the ranking of these scores.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def score_model(pipeline, X, y_true):
    """Run a saved pipeline on the data and return the six metrics."""
    y_pred = pipeline.predict(X)
    y_score = positive_scores(pipeline, X)
    return y_pred, {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_score),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def prepare(frame):
    """Split an uploaded dataframe into features and target.

    Booleans are converted to 0/1 exactly as they were during training.
    """
    data = frame.copy()
    for column in [TARGET, "Weekend"]:
        if column in data.columns and data[column].dtype == bool:
            data[column] = data[column].astype(int)

    y = data[TARGET]
    X = data.drop(columns=[TARGET])

    # The target may arrive as True/False text rather than real booleans.
    if y.dtype == object:
        y = y.map({"True": 1, "False": 0, True: 1, False: 0}).astype(int)
    return X, y


st.title("Online Shoppers Purchasing Intention - Classifier")
st.caption(
    "Upload the test CSV, choose a model, and view its evaluation metrics. "
    "All five models were trained beforehand and are only loaded here."
)

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Model selection")
choice = st.sidebar.selectbox("Choose a model", list(MODEL_FILES.keys()))
st.sidebar.markdown("---")
st.sidebar.info(
    "Required CSV format: all input feature columns plus the "
    f"`{TARGET}` target column."
)

missing = [name for name, file in MODEL_FILES.items()
           if not os.path.exists(os.path.join(MODEL_DIR, file))]
if missing:
    st.error(
        "These model files are missing from the `model/` folder: "
        + ", ".join(missing)
        + ". Run `python train_models.py` first."
    )
    st.stop()

# ----------------------------------------------------------------- upload
uploaded = st.file_uploader("Upload test data (CSV)", type="csv")

if uploaded is None:
    st.info("Waiting for a CSV file. You can use the `test_data.csv` "
            "from the repository.")
    st.stop()

# Read the file, and fail with a clear message rather than a traceback.
try:
    raw = pd.read_csv(uploaded)
except Exception as error:
    st.error(f"Could not read that file as CSV: {error}")
    st.stop()

if raw.empty:
    st.error("The uploaded file has no rows.")
    st.stop()

if TARGET not in raw.columns:
    st.error(
        f"The uploaded CSV must contain the target column `{TARGET}`. "
        f"Columns found: {list(raw.columns)}"
    )
    st.stop()

st.success(f"Loaded {raw.shape[0]} rows and {raw.shape[1]} columns.")
with st.expander("Preview uploaded data"):
    st.dataframe(raw.head())

X, y_true = prepare(raw)

if y_true.nunique() < 2:
    st.error(
        "The target column contains only one class, so AUC and MCC cannot "
        "be calculated. Please upload a test set containing both classes."
    )
    st.stop()

# ------------------------------------------------------- selected model
st.header(f"Results - {choice}")

try:
    pipeline = load_model(MODEL_FILES[choice])
    y_pred, scores = score_model(pipeline, X, y_true)
except Exception as error:
    st.error(
        f"Could not score this file with the saved model: {error}\n\n"
        "This usually means the uploaded CSV does not have the same feature "
        "columns as the data the models were trained on."
    )
    st.stop()

st.subheader("Evaluation metrics")
columns = st.columns(6)
for column, (name, value) in zip(columns, scores.items()):
    column.metric(name, f"{value:.4f}")

left, right = st.columns(2)

with left:
    st.subheader("Confusion matrix")
    matrix = confusion_matrix(y_true, y_pred)

    figure, axis = plt.subplots(figsize=(4, 3.2))
    axis.imshow(matrix, cmap="Blues")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_xticks([0, 1], ["No purchase", "Purchase"])
    axis.set_yticks([0, 1], ["No purchase", "Purchase"])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, matrix[i, j], ha="center", va="center",
                      color="black", fontsize=12)
    figure.tight_layout()
    st.pyplot(figure)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred,
        target_names=["No purchase", "Purchase"],
        zero_division=0,
    )
    st.code(report)

# -------------------------------------------------- all-model comparison
st.header("Comparison of all five models on this file")
st.caption("Every saved model scored on the same uploaded data.")

rows = {}
for name, filename in MODEL_FILES.items():
    try:
        model = load_model(filename)
        _, model_scores = score_model(model, X, y_true)
        rows[name] = model_scores
    except Exception as error:
        st.warning(f"{name} could not be scored: {error}")

if rows:
    comparison = pd.DataFrame(rows).T.round(4)
    st.dataframe(comparison, width="stretch")

    best = comparison["F1"].idxmax()
    st.success(f"Highest F1 score on this file: **{best}** "
               f"({comparison.loc[best, 'F1']:.4f})")

# --------------------------------------------- metrics from training run
metrics_path = os.path.join(MODEL_DIR, "metrics.json")
if os.path.exists(metrics_path):
    with st.expander("Metrics recorded during the original training run"):
        with open(metrics_path) as handle:
            saved = json.load(handle)
        st.dataframe(pd.DataFrame(saved).T.round(4), width="stretch")
