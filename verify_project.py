"""
Pre-submission self-check.
Run AFTER train_models.py:   python verify_project.py

Confirms every file the assignment asks for exists, the models reload, and
the app's scoring path works. Useful output for the BITS Lab screenshot.
"""

import json
import os
import re

import joblib
import pandas as pd

REQUIRED_FILES = [
    "app.py",
    "train_models.py",
    "requirements.txt",
    "README.md",
    "test_data.csv",
]

REQUIRED_MODELS = [
    "model/logistic_regression.joblib",
    "model/decision_tree.joblib",
    "model/knn.joblib",
    "model/naive_bayes.joblib",
    "model/svm.joblib",
    "model/metrics.json",
]

TARGET = "Revenue"
checks = []


def record(label, passed, detail=""):
    checks.append(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))


print("=" * 62)
print("REQUIRED FILES")
print("=" * 62)
for path in REQUIRED_FILES + REQUIRED_MODELS:
    record(path, os.path.exists(path))

print()
print("=" * 62)
print("TEST DATA")
print("=" * 62)
if os.path.exists("test_data.csv"):
    test = pd.read_csv("test_data.csv")
    record("test_data.csv has rows", len(test) > 0, f"rows={len(test)}")
    record("test_data.csv has target column", TARGET in test.columns)
    record("test_data.csv has >=12 input features",
           test.shape[1] - 1 >= 12, f"features={test.shape[1] - 1}")
    record("target has both classes", test[TARGET].nunique() == 2)

print()
print("=" * 62)
print("MODEL RELOAD AND SCORING")
print("=" * 62)
if os.path.exists("test_data.csv"):
    data = pd.read_csv("test_data.csv")
    for column in [TARGET, "Weekend"]:
        if column in data.columns and data[column].dtype == bool:
            data[column] = data[column].astype(int)
    X = data.drop(columns=[TARGET])

    for path in REQUIRED_MODELS:
        if path.endswith(".joblib") and os.path.exists(path):
            try:
                model = joblib.load(path)
                model.predict(X.head(10))

                # Four models expose predict_proba. SVC is trained without
                # probability=True (deprecated in scikit-learn 1.9 and much
                # slower), so it provides decision_function instead. Either
                # one gives the positive-class score that AUC needs.
                if hasattr(model, "predict_proba"):
                    model.predict_proba(X.head(10))
                    method = "predict_proba"
                elif hasattr(model, "decision_function"):
                    model.decision_function(X.head(10))
                    method = "decision_function"
                else:
                    raise AttributeError(
                        "model provides neither predict_proba nor "
                        "decision_function, so AUC cannot be calculated"
                    )

                record(f"{path} reload + predict + {method}", True)
            except Exception as error:
                record(f"{path} reload", False, str(error))

print()
print("=" * 62)
print("APP DOES NOT RETRAIN")
print("=" * 62)
if os.path.exists("app.py"):
    source = open("app.py").read()
    hits = re.findall(r"\.fit\(|fit_transform\(", source)
    record("no .fit() or .fit_transform() in app.py", not hits, f"found={hits}")

print()
print("=" * 62)
print("README PLACEHOLDERS")
print("=" * 62)
if os.path.exists("README.md"):
    readme = open("README.md").read()
    record("metric table filled in (no REPLACE marker)",
           "REPLACE THE VALUES" not in readme)
    record("observations written (no WRITE THESE marker)",
           "WRITE THESE YOURSELF" not in readme)
    record("Streamlit URL added",
           "PASTE YOUR STREAMLIT CLOUD URL" not in readme)

if os.path.exists("requirements.txt"):
    reqs = open("requirements.txt").read()
    record("requirements.txt scikit-learn version pinned",
           "REPLACE_WITH_YOUR_LAB_VERSION" not in reqs)

print()
print("=" * 62)
print(f"RESULT: {sum(checks)}/{len(checks)} checks passed")
print("=" * 62)
if all(checks):
    print("ALL CHECKS PASSED - project is ready to submit.")
else:
    print("Some checks failed. Fix them before submitting.")
