"""
Phase 1 — Verify all three dataset downloads.
Reads the first 5 rows of each required CSV and confirms expected columns exist.
Run: python notebooks/00_verify_downloads.py
"""

import pandas as pd
import os
import sys

# ── Dataset verification specs ──────────────────────────────────────────────
DATASETS = [
    {
        "name": "Lending Club",
        "path": "data/raw/lending_club/accepted_2007_to_2018Q4.csv",
        "required_cols": ["loan_amnt", "int_rate", "grade", "loan_status", "dti", "annual_inc", "fico_range_low"],
    },
    {
        "name": "IEEE-CIS Fraud (transactions)",
        "path": "data/raw/ieee-fraud-detection/train_transaction.csv",
        "required_cols": ["TransactionID", "isFraud", "TransactionAmt", "ProductCD"],
    },
    {
        "name": "IEEE-CIS Fraud (identity)",
        "path": "data/raw/ieee-fraud-detection/train_identity.csv",
        "required_cols": ["TransactionID"],
    },
    {
        "name": "Home Credit (application)",
        "path": "data/raw/home-credit-default-risk/application_train.csv",
        "required_cols": ["SK_ID_CURR", "TARGET", "AMT_CREDIT", "AMT_INCOME_TOTAL", "AMT_ANNUITY"],
    },
    {
        "name": "Home Credit (bureau)",
        "path": "data/raw/home-credit-default-risk/bureau.csv",
        "required_cols": ["SK_ID_CURR", "SK_ID_BUREAU"],
    },
    {
        "name": "Home Credit (previous applications)",
        "path": "data/raw/home-credit-default-risk/previous_application.csv",
        "required_cols": ["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS"],
    },
]


def verify_dataset(spec: dict) -> bool:
    """Read first 5 rows and confirm required columns are present."""
    path = spec["path"]
    name = spec["name"]

    if not os.path.exists(path):
        print(f"  [MISSING] {name} — file not found at: {path}")
        return False

    try:
        df = pd.read_csv(path, nrows=5)
        missing = [c for c in spec["required_cols"] if c not in df.columns]
        if missing:
            print(f"  [FAIL]    {name} — missing columns: {missing}")
            return False
        size_mb = os.path.getsize(path) / (1024 ** 2)
        print(f"  [OK]      {name} — {len(df.columns)} cols, {size_mb:.1f} MB on disk")
        return True
    except Exception as e:
        print(f"  [ERROR]   {name} — {e}")
        return False


if __name__ == "__main__":
    print("\n=== Phase 1: Dataset Verification ===\n")
    results = [verify_dataset(ds) for ds in DATASETS]

    passed = sum(results)
    total = len(results)
    print(f"\n{passed}/{total} datasets verified successfully.")

    if passed < total:
        print("\nDownload missing datasets with:")
        print("  kaggle datasets download -d wordsforthewise/lending-club -p data/raw/lending_club/ --unzip")
        print("  kaggle competitions download -c ieee-fraud-detection -p data/raw/ieee-fraud-detection/")
        print("  kaggle competitions download -c home-credit-default-risk -p data/raw/home-credit-default-risk/")
        sys.exit(1)
    else:
        print("\nAll datasets ready. Proceed to Phase 2.")
