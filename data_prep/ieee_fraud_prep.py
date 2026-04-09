"""
Phase 2 — IEEE-CIS Fraud Detection data preparation.
Produces: data/processed/fraud_clean.parquet
          data/features/fraud_*.npy  (X_train, X_test, y_train, y_test)
          models/artifacts/fraud_feature_names.pkl
          models/metrics/fraud_class_weight.json
Run: python data_prep/ieee_fraud_prep.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ── Paths ────────────────────────────────────────────────────────────────────
TRANS_PATH = "data/raw/ieee-fraud-detection/train_transaction.csv"
IDENT_PATH = "data/raw/ieee-fraud-detection/train_identity.csv"
PROCESSED_PATH = "data/processed/fraud_clean.parquet"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
METRICS_DIR = "models/metrics"

# Columns to cast as pandas Categorical (LightGBM handles natively)
CAT_COLS = ["ProductCD", "card4", "card6", "P_emaildomain", "DeviceType"]

# Missing-value threshold — drop columns with more than this fraction missing
MISSING_THRESHOLD = 0.80


def load_and_merge() -> pd.DataFrame:
    """Merge transaction and identity tables on TransactionID (left join)."""
    print("Loading transaction table...")
    trans = pd.read_csv(TRANS_PATH)
    print(f"  Transactions: {len(trans):,} rows, {trans.shape[1]} cols")

    print("Loading identity table...")
    ident = pd.read_csv(IDENT_PATH)
    print(f"  Identity: {len(ident):,} rows, {ident.shape[1]} cols")

    # Left join — identity NaNs are expected for unmatched rows
    df = pd.merge(trans, ident, on="TransactionID", how="left")
    print(f"  Merged: {len(df):,} rows, {df.shape[1]} cols")
    return df


def drop_high_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns where more than MISSING_THRESHOLD fraction of values are NaN."""
    missing_frac = df.isnull().mean()
    cols_to_drop = missing_frac[missing_frac > MISSING_THRESHOLD].index.tolist()
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped {len(cols_to_drop)} high-missingness columns → {df.shape[1]} remaining")
    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast known categorical columns to pandas Categorical dtype for LightGBM."""
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def compute_class_weight(y: pd.Series) -> float:
    """Compute scale_pos_weight = neg_count / pos_count for imbalanced binary classification."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos
    print(f"  Class distribution — neg: {neg:,}, pos: {pos:,}, scale_pos_weight: {spw:.2f}")
    return spw


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # ── Load & merge ─────────────────────────────────────────────────────────
    df = load_and_merge()

    # ── Drop high-missingness columns ────────────────────────────────────────
    df = drop_high_missingness(df)

    # ── Cast categoricals ────────────────────────────────────────────────────
    df = cast_categoricals(df)

    # ── Median-impute remaining numeric columns ──────────────────────────────
    target_col = "isFraud"
    drop_cols = ["TransactionID", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Separate numeric and categorical for imputation
    num_cols = [c for c in feature_cols if df[c].dtype != "category"]
    cat_cols_present = [c for c in feature_cols if df[c].dtype == "category"]

    imputer = SimpleImputer(strategy="median")
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # ── Save cleaned parquet ─────────────────────────────────────────────────
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"  Saved: {PROCESSED_PATH}")

    # ── Compute and save class weight ────────────────────────────────────────
    spw = compute_class_weight(df[target_col])
    with open(f"{METRICS_DIR}/fraud_class_weight.json", "w") as f:
        json.dump({"scale_pos_weight": spw}, f, indent=2)

    # ── Prepare feature matrix ───────────────────────────────────────────────
    # Encode categoricals as integer codes for numpy storage
    for col in cat_cols_present:
        df[col] = df[col].cat.codes

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values

    # ── Train/test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Save numpy arrays ────────────────────────────────────────────────────
    for name, arr in [
        ("X_train", X_train), ("X_test", X_test),
        ("y_train", y_train), ("y_test", y_test),
    ]:
        np.save(f"{FEATURES_DIR}/fraud_{name}.npy", arr)
    print(f"  Saved 4 numpy arrays to {FEATURES_DIR}/")

    # ── Save feature names ───────────────────────────────────────────────────
    joblib.dump(feature_cols, f"{ARTIFACTS_DIR}/fraud_feature_names.pkl")
    joblib.dump(imputer, f"{ARTIFACTS_DIR}/fraud_imputer.pkl")
    print(f"  Saved feature names and imputer to {ARTIFACTS_DIR}/")
    print("\nIEEE-CIS Fraud prep complete.")


if __name__ == "__main__":
    main()
