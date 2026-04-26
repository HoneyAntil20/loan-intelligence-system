"""
Phase 3 — Train LightGBM fraud classifier on IEEE-CIS data.
Produces: models/artifacts/fraud_lgbm.pkl
          models/metrics/fraud_metrics.json
          models/metrics/fraud_threshold.json
Run: python models/train_fraud.py
"""

import os
import json
import joblib
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
METRICS_DIR = "models/metrics"


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float = 0.80) -> float:
    """
    Find decision threshold at target recall using precision-recall curve.
    Default 0.5 is wrong for imbalanced data — we tune for acceptable FPR.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Find threshold closest to target recall
    idx = np.argmin(np.abs(recall - target_recall))
    threshold = float(thresholds[min(idx, len(thresholds) - 1)])
    print(f"  Threshold at recall={target_recall:.2f}: {threshold:.4f}  "
          f"(precision={precision[idx]:.4f})")
    return threshold


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Load feature arrays ──────────────────────────────────────────────────
    print("Loading IEEE-CIS feature arrays...")
    X_train = np.load(f"{FEATURES_DIR}/fraud_X_train.npy")
    X_test = np.load(f"{FEATURES_DIR}/fraud_X_test.npy")
    y_train = np.load(f"{FEATURES_DIR}/fraud_y_train.npy")
    y_test = np.load(f"{FEATURES_DIR}/fraud_y_test.npy")
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Load class weight ────────────────────────────────────────────────────
    with open(f"{METRICS_DIR}/fraud_class_weight.json") as f:
        spw = json.load(f)["scale_pos_weight"]
    print(f"  scale_pos_weight: {spw:.2f}")

    # ── Create validation split for early stopping ───────────────────────────
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )

    # ── Train LightGBM with class weight ─────────────────────────────────────
    print("Training LightGBM fraud classifier...")
    model = LGBMClassifier(
        n_estimators=1000,
        scale_pos_weight=spw,
        max_depth=8,
        learning_rate=0.03,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
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
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"  ROC-AUC: {roc_auc:.4f}  (target > 0.90)")
    print(f"  PR-AUC:  {pr_auc:.4f}")

    # ── Find optimal threshold ───────────────────────────────────────────────
    threshold = find_optimal_threshold(y_test, y_prob, target_recall=0.80)

    # ── Save model artifact ──────────────────────────────────────────────────
    joblib.dump(model, f"{ARTIFACTS_DIR}/fraud_lgbm.pkl")
    print(f"  Saved: {ARTIFACTS_DIR}/fraud_lgbm.pkl")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc), "threshold": threshold}
    with open(f"{METRICS_DIR}/fraud_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{METRICS_DIR}/fraud_threshold.json", "w") as f:
        json.dump({"threshold": threshold}, f, indent=2)
    print(f"  Saved metrics and threshold to {METRICS_DIR}/")
    print("\nFraud model training complete.")


if __name__ == "__main__":
    main()
