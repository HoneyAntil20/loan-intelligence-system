"""
Phase 3 — Train the X-Learner uplift model using EconML + LightGBM base learners.
Uses econml.metalearners.XLearner (equivalent to causalml BaseXLearner).
Produces: models/artifacts/xlearner_lgbm.pkl
          models/metrics/uplift_metrics.json
Run: python models/train_uplift.py
"""

import os
import json
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from econml.metalearners import XLearner
from sklearn.metrics import roc_auc_score

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
METRICS_DIR = "models/metrics"


def auuc_score(tau: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """
    Area Under the Uplift Curve (AUUC).
    Approximated by comparing AUC of tau against actual treatment outcomes.
    """
    # Treated group: AUC of predicted ITE vs actual outcome
    treated_mask = t == 1
    if treated_mask.sum() < 10:
        return 0.5
    return roc_auc_score(y[treated_mask], tau[treated_mask])


def assign_segment(tau: float, p0: float) -> str:
    """
    Assign uplift segment based on ITE (tau) and baseline repay probability (p0).
    Persuadable: rate offer meaningfully boosts repayment.
    Sure Thing: will repay regardless — no incentive needed.
    Do Not Disturb: rate offer is counterproductive.
    Lost Cause: low baseline + low uplift.
    """
    if tau < -0.03:
        return "Do Not Disturb"
    if p0 >= 0.75:
        return "Sure Thing"
    if tau >= 0.05 and p0 < 0.75:
        return "Persuadable"
    return "Lost Cause"


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Load feature arrays ──────────────────────────────────────────────────
    print("Loading Lending Club feature arrays...")
    X_train = np.load(f"{FEATURES_DIR}/lending_X_train.npy")
    X_test = np.load(f"{FEATURES_DIR}/lending_X_test.npy")
    y_train = np.load(f"{FEATURES_DIR}/lending_y_train.npy")
    y_test = np.load(f"{FEATURES_DIR}/lending_y_test.npy")
    T_train = np.load(f"{FEATURES_DIR}/lending_T_train.npy")
    T_test = np.load(f"{FEATURES_DIR}/lending_T_test.npy")
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Define base LightGBM learner ─────────────────────────────────────────
    # Same hyperparameters as specified in the plan (section 4.1)
    base_learner = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )

    # ── Train X-Learner ──────────────────────────────────────────────────────
    print("Training X-Learner (this may take 10-20 minutes on full dataset)...")
    learner = XLearner(models=base_learner)
    learner.fit(y_train, T_train, X=X_train)
    print("  Training complete.")

    # ── Predict individual treatment effects (ITE) ───────────────────────────
    tau_test = learner.effect(X_test).flatten()
    print(f"  ITE stats — mean: {tau_test.mean():.4f}, std: {tau_test.std():.4f}")

    # ── Evaluate AUUC ────────────────────────────────────────────────────────
    auuc = auuc_score(tau_test, y_test, T_test)
    print(f"  AUUC: {auuc:.4f}  (target > 0.55)")

    # ── Segment distribution ─────────────────────────────────────────────────
    # Estimate baseline repay probability using control group model
    p0_test = learner.models_t[0].predict_proba(X_test)[:, 1]
    segments = [assign_segment(t, p) for t, p in zip(tau_test, p0_test)]
    seg_counts = {s: segments.count(s) for s in set(segments)}
    persuadable_pct = seg_counts.get("Persuadable", 0) / len(segments)
    print(f"  Segment distribution: {seg_counts}")
    print(f"  Persuadable: {persuadable_pct:.1%}  (expect ~28%)")

    # ── Save model artifact ──────────────────────────────────────────────────
    joblib.dump(learner, f"{ARTIFACTS_DIR}/xlearner_lgbm.pkl")
    print(f"  Saved: {ARTIFACTS_DIR}/xlearner_lgbm.pkl")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "mean_ite": float(tau_test.mean()),
        "auuc": float(auuc),
        "ite_std": float(tau_test.std()),
        "segment_distribution": seg_counts,
        "persuadable_pct": float(persuadable_pct),
    }
    with open(f"{METRICS_DIR}/uplift_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {METRICS_DIR}/uplift_metrics.json")
    print("\nUplift model training complete.")


if __name__ == "__main__":
    main()
