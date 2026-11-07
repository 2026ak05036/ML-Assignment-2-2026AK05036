"""
ML Assignment 2 - Online Shoppers Purchasing Intention
Trains five classification models and saves them as complete scikit-learn
Pipelines (preprocessing + classifier together).

Run:  python train_models.py

Produces:
    model/logistic_regression.joblib
    model/decision_tree.joblib
    model/knn.joblib
    model/naive_bayes.joblib
    model/svm.joblib
    model/metrics.json
    test_data.csv
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

DATA_FILE = "online_shoppers_intention.csv"
TARGET = "Revenue"
MODEL_DIR = "model"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Columns that hold text categories and must be one-hot encoded.
CATEGORICAL_COLUMNS = ["Month", "VisitorType"]

# The 17 input features the UCI Online Shoppers dataset should contain.
# Used to validate the CSV before training starts.
EXPECTED_FEATURES = [
    "Administrative", "Administrative_Duration",
    "Informational", "Informational_Duration",
    "ProductRelated", "ProductRelated_Duration",
    "BounceRates", "ExitRates", "PageValues", "SpecialDay",
    "Month", "OperatingSystems", "Browser", "Region",
    "TrafficType", "VisitorType", "Weekend",
]


def load_data():
    """Read the dataset and validate it before any training happens.

    Fails early, with a message that says what is actually wrong, rather
    than letting a bad file cause a confusing error deep inside sklearn.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"'{DATA_FILE}' not found in {os.getcwd()}.\n"
            "Download it first:  python download_data.py"
        )

    frame = pd.read_csv(DATA_FILE)
    print("File loaded            :", DATA_FILE)
    print("Dataset shape          :", frame.shape)

    # --- check 1: every expected column is present -----------------------
    found = set(frame.columns)
    expected = set(EXPECTED_FEATURES) | {TARGET}

    missing_columns = sorted(expected - found)
    extra_columns = sorted(found - expected)

    if missing_columns:
        raise ValueError(
            "The dataset is missing these required columns:\n"
            f"  {missing_columns}\n"
            f"Columns found in the file: {sorted(found)}\n"
            "This usually means the wrong CSV was downloaded."
        )
    if extra_columns:
        print(f"Note: ignoring unexpected extra column(s): {extra_columns}")
        frame = frame.drop(columns=extra_columns)

    print(f"Expected columns       : all {len(expected)} present  [PASS]")

    # --- check 2: size requirements from the assignment ------------------
    n_rows = frame.shape[0]
    n_features = frame.shape[1] - 1
    rows_ok = n_rows >= 500
    features_ok = n_features >= 12

    print(f"Requirement >=500 rows : {n_rows:6}   [{'PASS' if rows_ok else 'FAIL'}]")
    print(f"Requirement >=12 feats : {n_features:6}   [{'PASS' if features_ok else 'FAIL'}]")

    if not rows_ok or not features_ok:
        raise ValueError(
            f"Dataset does not meet the assignment minimums "
            f"(needs >=500 rows and >=12 features, got {n_rows} and {n_features})."
        )

    # --- check 3: missing values -----------------------------------------
    nulls = frame.isnull().sum()
    total_nulls = int(nulls.sum())
    print(f"Missing values         : {total_nulls}")
    if total_nulls:
        print("Columns containing missing values:")
        print(nulls[nulls > 0].to_string())
        raise ValueError(
            "This dataset is expected to have no missing values. Found "
            f"{total_nulls}. Re-download the file, or add an imputation "
            "step before training."
        )

    print(f"Duplicate rows         : {int(frame.duplicated().sum())}")

    # --- check 4: convert booleans, then verify data types ---------------
    # Revenue and Weekend arrive as True/False; make them plain 0/1 so that
    # every model and metric function handles them consistently.
    for column in [TARGET, "Weekend"]:
        if frame[column].dtype == bool:
            frame[column] = frame[column].astype(int)
        elif frame[column].dtype == object:
            frame[column] = frame[column].map(
                {"True": 1, "False": 0, True: 1, False: 0}
            )

    wrong_type = []
    for column in EXPECTED_FEATURES:
        is_text = column in CATEGORICAL_COLUMNS
        is_numeric = pd.api.types.is_numeric_dtype(frame[column])
        if is_text and is_numeric:
            wrong_type.append(f"{column} (expected text categories, got numeric)")
        if not is_text and not is_numeric:
            wrong_type.append(f"{column} (expected numeric, got {frame[column].dtype})")

    if wrong_type:
        raise ValueError(
            "These columns have unexpected data types:\n  "
            + "\n  ".join(wrong_type)
        )
    print("Column data types      : as expected  [PASS]")

    # --- check 5: the target must be usable for binary classification ----
    if frame[TARGET].isnull().any():
        raise ValueError(
            f"The target column '{TARGET}' contains values that could not be "
            "interpreted as True/False."
        )

    classes = sorted(frame[TARGET].unique())
    if len(classes) != 2:
        raise ValueError(
            f"Expected a binary target, but '{TARGET}' has {len(classes)} "
            f"distinct values: {classes}"
        )
    print(f"Target '{TARGET}'       : binary {classes}  [PASS]")

    print("\nTarget class balance:")
    counts = frame[TARGET].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label} : {count:6}  ({100 * count / n_rows:.2f}%)")

    minority = counts.min() / n_rows
    if minority < 0.05:
        print(f"Warning: the minority class is only {100 * minority:.2f}% of "
              "the data. Accuracy will be misleading; rely on F1 and MCC.")

    return frame


def build_preprocessor(X):
    """One ColumnTransformer reused by all five models.

    Text categories  -> one-hot encoding
    Everything else  -> standard scaling

    handle_unknown='ignore' means an unseen Month value in an uploaded CSV
    produces all-zero dummies instead of crashing the app.
    """
    categorical = [c for c in CATEGORICAL_COLUMNS if c in X.columns]
    numeric = [c for c in X.columns if c not in categorical]

    print(f"\nCategorical columns ({len(categorical)}): {categorical}")
    print(f"Numeric columns     ({len(numeric)}): {numeric}")

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", StandardScaler(), numeric),
        ]
    )


def get_models():
    """The five classifiers required by the assignment."""
    return {
        "logistic_regression": ("Logistic Regression",
                                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        "decision_tree": ("Decision Tree",
                          DecisionTreeClassifier(random_state=RANDOM_STATE)),
        "knn": ("KNN",
                KNeighborsClassifier()),
        "naive_bayes": ("Naive Bayes",
                        GaussianNB()),
        # Plain SVC. We deliberately do NOT pass probability=True:
        #   * that option is deprecated in scikit-learn 1.9
        #   * it makes training roughly five times slower
        # AUC only depends on how the scores RANK the samples, and
        # decision_function gives a valid ranking, so AUC is unaffected.
        "svm": ("Support Vector Machine",
                SVC(random_state=RANDOM_STATE)),
    }


def positive_scores(pipeline, X):
    """Score for the positive class, used for AUC.

    Four of the models expose predict_proba. SVC does not (see get_models),
    so its decision_function values are used instead.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def evaluate(y_true, y_pred, y_score):
    """The six metrics required by the assignment."""
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": float(roc_auc_score(y_true, y_score)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
    }


def main():
    print("=" * 62)
    print("STEP 1 - LOAD AND VALIDATE DATASET")
    print("=" * 62)
    frame = load_data()

    X = frame.drop(columns=[TARGET])
    y = frame[TARGET]

    print("\n" + "=" * 62)
    print("STEP 2 - TRAIN / TEST SPLIT")
    print("=" * 62)
    # stratify keeps the same class balance in both halves, which matters
    # because only ~15% of sessions end in a purchase.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Train rows:", X_train.shape[0])
    print("Test rows :", X_test.shape[0])

    # Save the untouched test rows. The pipeline does the transforming later,
    # so this file holds ordinary readable values plus the target column.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.values
    test_frame.to_csv("test_data.csv", index=False)
    print("Saved test_data.csv ->", test_frame.shape)

    print("\n" + "=" * 62)
    print("STEP 3 - TRAIN FIVE MODELS")
    print("=" * 62)
    os.makedirs(MODEL_DIR, exist_ok=True)

    preprocessor = build_preprocessor(X)
    results = {}

    for key, (label, classifier) in get_models().items():
        print(f"\nTraining {label} ...")
        start = time.time()

        # Preprocessing lives INSIDE the pipeline. fit() therefore learns the
        # scaler/encoder from training data only - the test set cannot leak in.
        pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("classifier", classifier),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_score = positive_scores(pipeline, X_test)

        scores = evaluate(y_test, y_pred, y_score)
        results[label] = scores

        path = os.path.join(MODEL_DIR, f"{key}.joblib")
        joblib.dump(pipeline, path)

        elapsed = time.time() - start
        print(f"  saved {path}  ({elapsed:.1f}s)")
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    print("\n" + "=" * 62)
    print("STEP 4 - COMPARISON TABLE")
    print("=" * 62)
    table = pd.DataFrame(results).T.round(4)
    print(table.to_string())

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as handle:
        json.dump(results, handle, indent=2)
    print("\nSaved model/metrics.json")

    print("\n" + "=" * 62)
    print("STEP 5 - RELOAD VERIFICATION")
    print("=" * 62)
    # Prove every saved file can be loaded and used again. If this fails,
    # the Streamlit app would fail too.
    for key in get_models():
        path = os.path.join(MODEL_DIR, f"{key}.joblib")
        reloaded = joblib.load(path)
        sample = reloaded.predict(X_test.head(5))
        print(f"  {key:22} reload OK  sample predictions={list(sample)}")

    print("\nTRAINING COMPLETE")


if __name__ == "__main__":
    main()
