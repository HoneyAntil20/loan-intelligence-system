"""
Phase 2 — Home Credit Default Risk data preparation.
Produces: data/processed/credit_clean.parquet
          data/features/credit_*.npy  (X_train, X_test, y_train, y_test)
          models/artifacts/credit_encoder.pkl
          models/artifacts/credit_imputer.pkl
          models/artifacts/credit_feature_names.pkl
          models/metrics/credit_class_weight.json
Run: python data_prep/home_credit_prep.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# ── Paths ────────────────────────────────────────────────────────────────────
APP_PATH = "data/raw/home-credit-default-risk/application_train.csv"
BUREAU_PATH = "data/raw/home-credit-default-risk/bureau.csv"
PREV_PATH = "data/raw/home-credit-default-risk/previous_application.csv"
PROCESSED_PATH = "data/processed/credit_clean.parquet"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
METRICS_DIR = "models/metrics"

# Binary columns: Y/N → 1/0
BINARY_COLS = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY"]

# Categorical columns to cast for LightGBM native handling
CAT_COLS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "OCCUPATION_TYPE",
]


def load_and_merge() -> pd.DataFrame:
    """Load application_train and left-join aggregated bureau + previous_application."""
    print("Loading application_train.csv...")
    app = pd.read_csv(APP_PATH)
    print(f"  Application: {len(app):,} rows, {app.shape[1]} cols")

    # ── Bureau aggregation ────────────────────────────────────────────────────
    print("Loading and aggregating bureau.csv...")
    bureau = pd.read_csv(BUREAU_PATH)
    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        bureau_loan_count=("SK_ID_BUREAU", "count"),
        bureau_max_overdue=("AMT_CREDIT_MAX_OVERDUE", "max"),
        bureau_total_overdue=("AMT_CREDIT_SUM_OVERDUE", "sum"),
    ).reset_index()

    # ── Previous application aggregation ─────────────────────────────────────
    print("Loading and aggregating previous_application.csv...")
    prev = pd.read_csv(PREV_PATH)
    prev_agg = prev.groupby("SK_ID_CURR").agg(
        prev_app_count=("SK_ID_PREV", "count"),
        prev_approved=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        prev_refused=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        prev_avg_credit=("AMT_CREDIT", "mean"),
    ).reset_index()

    # ── Left-join both aggregations onto application ──────────────────────────
    df = app.merge(bureau_agg, on="SK_ID_CURR", how="left")
    df = df.merge(prev_agg, on="SK_ID_CURR", how="left")
    print(f"  Merged: {len(df):,} rows, {df.shape[1]} cols")
    return df


def encode_binary(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Y/N binary columns to 1/0 integers."""
    for col in BINARY_COLS:
        if col in df.columns:
            df[col] = df[col].map({"Y": 1, "N": 0})
    return df


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical columns to pandas Categorical for LightGBM."""
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def engineer_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ratio features — strongest default predictors per domain knowledge.
    credit_income_ratio: how large the credit is relative to income.
    annuity_income_ratio: monthly payment burden relative to income.
    """
    df["credit_income_ratio"] = df["AMT_CREDIT"] / (df["AMT_INCOME_TOTAL"] + 1)
    df["annuity_income_ratio"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] + 1)
    return df


def compute_class_weight(y: pd.Series) -> float:
    """scale_pos_weight = neg / pos for imbalanced binary target."""
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    spw = neg / pos
    print(f"  Default rate: {pos/len(y):.2%}  |  scale_pos_weight: {spw:.2f}")
    return spw


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # ── Load & merge ─────────────────────────────────────────────────────────
    df = load_and_merge()
    df = encode_binary(df)
    df = cast_categoricals(df)
    df = engineer_ratio_features(df)

    # ── Save cleaned parquet ─────────────────────────────────────────────────
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"  Saved: {PROCESSED_PATH}")

    # ── Compute and save class weight ────────────────────────────────────────
    target_col = "TARGET"
    spw = compute_class_weight(df[target_col])
    with open(f"{METRICS_DIR}/credit_class_weight.json", "w") as f:
        json.dump({"scale_pos_weight": spw}, f, indent=2)

    # ── Prepare feature matrix ───────────────────────────────────────────────
    drop_cols = ["SK_ID_CURR", target_col]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Encode categoricals as integer codes for numpy storage
    cat_cols_present = [c for c in feature_cols if df[c].dtype.name == "category"]
    for col in cat_cols_present:
        df[col] = df[col].cat.codes

    X = df[feature_cols].copy()

    # Median-impute all remaining NaNs
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X).astype(np.float32)
    y = df[target_col].values

    # ── Train/test split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Save numpy arrays ────────────────────────────────────────────────────
    for name, arr in [
        ("X_train", X_train), ("X_test", X_test),
        ("y_train", y_train), ("y_test", y_test),
    ]:
        np.save(f"{FEATURES_DIR}/credit_{name}.npy", arr)
    print(f"  Saved 4 numpy arrays to {FEATURES_DIR}/")

    # ── Save artifacts ───────────────────────────────────────────────────────
    joblib.dump(imputer, f"{ARTIFACTS_DIR}/credit_imputer.pkl")
    joblib.dump(feature_cols, f"{ARTIFACTS_DIR}/credit_feature_names.pkl")
    print(f"  Saved imputer and feature names to {ARTIFACTS_DIR}/")
    print("\nHome Credit prep complete.")


if __name__ == "__main__":
    main()
