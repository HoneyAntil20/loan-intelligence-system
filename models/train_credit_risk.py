"""
Phase 3 — Train LightGBM credit risk model on Home Credit data.
Produces: models/artifacts/credit_lgbm.pkl
          models/artifacts/credit_shap_explainer.pkl
          models/metrics/credit_metrics.json
Run: python models/train_credit_risk.py
"""

import os
import json
import joblib
import numpy as np
import shap
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
METRICS_DIR = "models/metrics"


def gini_coefficient(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Gini = 2 * AUC - 1. Standard credit risk model evaluation metric."""
    auc = roc_auc_score(y_true, y_prob)
    return 2 * auc - 1


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """KS statistic: max separation between default and non-default score distributions."""
    pos_scores = y_prob[y_true == 1]
    neg_scores = y_prob[y_true == 0]
    ks_stat, _ = ks_2samp(pos_scores, neg_scores)
    return float(ks_stat)


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Load feature arrays ──────────────────────────────────────────────────
    print("Loading Home Credit feature arrays...")
    X_train = np.load(f"{FEATURES_DIR}/credit_X_train.npy")
    X_test = np.load(f"{FEATURES_DIR}/credit_X_test.npy")
    y_train = np.load(f"{FEATURES_DIR}/credit_y_train.npy")
    y_test = np.load(f"{FEATURES_DIR}/credit_y_test.npy")
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Load class weight ────────────────────────────────────────────────────
    with open(f"{METRICS_DIR}/credit_class_weight.json") as f:
        spw = json.load(f)["scale_pos_weight"]
    print(f"  scale_pos_weight: {spw:.2f}")

    # ── Create validation split for early stopping ───────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # ── Train LightGBM credit risk model ─────────────────────────────────────
    print("Training LightGBM credit risk model...")
    model = LGBMClassifier(
        n_estimators=300,
        scale_pos_weight=spw,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=50,   # Prevents overfitting on rare default cases
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[early_stopping(50, verbose=False), log_evaluation(50)],
    )
    print("  Training complete.")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_test)[:, 1]
    gini = gini_coefficient(y_test, y_prob)
    ks = ks_statistic(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    print(f"  AUC:  {auc:.4f}  (target > 0.80)")
    print(f"  Gini: {gini:.4f}  (target > 0.60)")
    print(f"  KS:   {ks:.4f}")

    # ── Pre-compute SHAP explainer ────────────────────────────────────────────
    # Saved once — avoids re-fitting per inference request (saves ~2s per call)
    print("Pre-computing SHAP TreeExplainer...")
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, f"{ARTIFACTS_DIR}/credit_shap_explainer.pkl")
    print(f"  Saved: {ARTIFACTS_DIR}/credit_shap_explainer.pkl")

    # ── Save model artifact ──────────────────────────────────────────────────
    joblib.dump(model, f"{ARTIFACTS_DIR}/credit_lgbm.pkl")
    print(f"  Saved: {ARTIFACTS_DIR}/credit_lgbm.pkl")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {"roc_auc": float(auc), "gini": float(gini), "ks_statistic": float(ks)}
    with open(f"{METRICS_DIR}/credit_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {METRICS_DIR}/credit_metrics.json")
    print("\nCredit risk model training complete.")


if __name__ == "__main__":
    main()
