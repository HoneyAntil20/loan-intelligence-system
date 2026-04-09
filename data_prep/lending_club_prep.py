"""
Phase 2 — Lending Club data preparation for the uplift model.
Produces: data/processed/lending_clean.parquet
          data/features/lending_*.npy  (X_train, X_test, y_train, y_test, T_train, T_test)
          models/artifacts/lending_encoder.pkl
          models/artifacts/lending_imputer.pkl
          models/artifacts/lending_feature_names.pkl
Run: python data_prep/lending_club_prep.py
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ── Paths ────────────────────────────────────────────────────────────────────
RAW_PATH = "data/raw/lending_club/accepted_2007_to_2018Q4.csv"
PROCESSED_PATH = "data/processed/lending_clean.parquet"
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"

# Terminal loan statuses — exclude 'Current' (outcome unknown)
TERMINAL_STATUSES = {"Fully Paid", "Charged Off", "Default", "Late (31-120 days)"}

# Categorical columns to ordinal-encode
CAT_COLS = ["grade", "home_ownership", "purpose"]

# Numeric feature columns used for modelling
NUMERIC_COLS = [
    "loan_amnt", "int_rate", "installment", "annual_inc",
    "dti", "fico_range_low", "open_acc", "pub_rec",
    "revol_bal", "revol_util", "total_acc", "emp_length_num",
    # Engineered features added below
    "rate_sensitivity", "debt_burden", "income_adequacy",
]


def load_and_filter(path: str) -> pd.DataFrame:
    """Load CSV and keep only terminal-status loans."""
    print("Loading Lending Club CSV (this may take a minute)...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  Raw rows: {len(df):,}")
    df = df[df["loan_status"].isin(TERMINAL_STATUSES)].copy()
    print(f"  After terminal-status filter: {len(df):,}")
    return df


def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip % from int_rate, extract numeric digits from emp_length."""
    # int_rate: '13.56%' → 13.56
    df["int_rate"] = df["int_rate"].astype(str).str.replace("%", "").str.strip().astype(float)

    # emp_length: '10+ years' → 10, '< 1 year' → 0, 'n/a' → NaN
    df["emp_length_num"] = (
        df["emp_length"]
        .astype(str)
        .str.extract(r"(\d+)")[0]
        .astype(float)
    )
    return df


def define_treatment_outcome(df: pd.DataFrame) -> pd.DataFrame:
    """
    T = 1 if borrower received a below-median interest rate for their grade.
    y = 1 if loan_status == 'Fully Paid' (positive outcome).
    """
    # Compute median rate per grade
    median_rate_by_grade = df.groupby("grade")["int_rate"].transform("median")
    df["T"] = (df["int_rate"] < median_rate_by_grade).astype(int)
    df["y"] = (df["loan_status"] == "Fully Paid").astype(int)
    print(f"  Treatment rate: {df['T'].mean():.2%}  |  Outcome rate: {df['y'].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create uplift-specific interaction features."""
    # Rate sensitivity: higher FICO + lower rate = more sensitive to rate changes
    df["rate_sensitivity"] = (850 - df["fico_range_low"]) * df["int_rate"] / 100

    # Debt burden: scaled product of DTI and loan amount
    df["debt_burden"] = df["dti"] * df["loan_amnt"] / 10_000

    # Income adequacy: how large the loan is relative to income
    df["income_adequacy"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    return df


def encode_and_impute(df: pd.DataFrame):
    """Ordinal-encode categoricals, median-impute numerics. Return arrays + artifacts."""
    # Ordinal encode categorical columns
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df[CAT_COLS] = encoder.fit_transform(df[CAT_COLS].astype(str))

    # Combine all feature columns
    feature_cols = CAT_COLS + NUMERIC_COLS
    X = df[feature_cols].copy()

    # Median impute remaining NaNs
    imputer = SimpleImputer(strategy="median")
    X_arr = imputer.fit_transform(X)

    return X_arr, feature_cols, encoder, imputer


def main():
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # ── Load & clean ─────────────────────────────────────────────────────────
    df = load_and_filter(RAW_PATH)
    df = clean_strings(df)
    df = define_treatment_outcome(df)
    df = engineer_features(df)

    # ── Save cleaned parquet ─────────────────────────────────────────────────
    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"  Saved: {PROCESSED_PATH}")

    # ── Encode & impute ──────────────────────────────────────────────────────
    X, feature_cols, encoder, imputer = encode_and_impute(df)
    T = df["T"].values
    y = df["y"].values

    # ── Stratified train/test split on treatment T ───────────────────────────
    X_train, X_test, y_train, y_test, T_train, T_test = train_test_split(
        X, y, T, test_size=0.2, random_state=42, stratify=T
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Save numpy arrays ────────────────────────────────────────────────────
    for name, arr in [
        ("X_train", X_train), ("X_test", X_test),
        ("y_train", y_train), ("y_test", y_test),
        ("T_train", T_train), ("T_test", T_test),
    ]:
        np.save(f"{FEATURES_DIR}/lending_{name}.npy", arr)
    print(f"  Saved 6 numpy arrays to {FEATURES_DIR}/")

    # ── Save artifacts ───────────────────────────────────────────────────────
    joblib.dump(encoder, f"{ARTIFACTS_DIR}/lending_encoder.pkl")
    joblib.dump(imputer, f"{ARTIFACTS_DIR}/lending_imputer.pkl")
    joblib.dump(feature_cols, f"{ARTIFACTS_DIR}/lending_feature_names.pkl")
    print(f"  Saved encoder, imputer, feature names to {ARTIFACTS_DIR}/")
    print("\nLending Club prep complete.")


if __name__ == "__main__":
    main()
