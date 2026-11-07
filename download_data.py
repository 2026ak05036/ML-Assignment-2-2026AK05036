"""
Downloads the UCI Online Shoppers Purchasing Intention dataset and checks it
against the assignment's minimum size requirements.

Run:  python download_data.py
"""

import os
import pandas as pd

CSV_FILE = "online_shoppers_intention.csv"
TARGET = "Revenue"


def main():
    if os.path.exists(CSV_FILE):
        print(f"'{CSV_FILE}' already present - skipping download.")
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError:
            print("Install the helper first:  pip install --user ucimlrepo")
            print("Or download manually from:")
            print("  https://archive.ics.uci.edu/dataset/468/"
                  "online+shoppers+purchasing+intention+dataset")
            return

        print("Downloading UCI dataset id=468 ...")
        dataset = fetch_ucirepo(id=468)
        frame = dataset.data.features.join(dataset.data.targets)
        frame.to_csv(CSV_FILE, index=False)
        print(f"Saved '{CSV_FILE}'.")

    frame = pd.read_csv(CSV_FILE)
    n_rows = frame.shape[0]
    n_features = frame.shape[1] - 1

    print("\nshape           :", frame.shape)
    print("input features  :", n_features)
    print("missing values  :", int(frame.isnull().sum().sum()))
    print("duplicate rows  :", int(frame.duplicated().sum()))
    print("\ntarget distribution:")
    print(frame[TARGET].value_counts().to_string())

    print(f"\n>=500 instances : {n_rows:6}  [{'PASS' if n_rows >= 500 else 'FAIL'}]")
    print(f">=12 features   : {n_features:6}  [{'PASS' if n_features >= 12 else 'FAIL'}]")


if __name__ == "__main__":
    main()
