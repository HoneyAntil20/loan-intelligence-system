"""
Phase 3 — Train uplift model on Lending Club data.
Uses a T-Learner approach (two separate outcome models) which is more robust
on observational data than X-Learner when propensity scores are near-uniform.
ITE = P(repay | T=1, X) - P(repay | T=0, X)
Produces: models/artifacts/xlearner_lgbm.pkl  (T-Learner wrapper, same interface)
          models/metrics/uplift_metrics.json
Run: python models/train_uplift.py
"""

import os
import sys
import json
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

# Ensure project root is on path so models.tlearner is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.tlearner import TLearner

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = "data/features"
ARTIFACTS_DIR = "models/artifacts"
METRICS_DIR = "models/metrics"


def auuc_score(tau: np.ndarray, y: np.ndarray, t: np.ndarray) -> float:
    """Qini-based AUUC: measures how well predicted ITE ranks actual uplift."""
    n = len(tau)
    order = np.argsort(-tau)
    y_s, t_s = y[order], t[order]

    n_t = t.sum()
    n_c = (1 - t).sum()
    if n_t == 0 or n_c == 0:
        return 0.5

    cum_t = np.cumsum(y_s * t_s)
    cum_c = np.cumsum(y_s * (1 - t_s))
    cum_nt = np.cumsum(t_s)
    cum_nc = np.cumsum(1 - t_s)

    with np.errstate(divide="ignore", invalid="ignore"):
        uplift = np.where(cum_nt > 0, cum_t / n_t - cum_c / n_c, 0.0)

    return float(min(max(np.trapezoid(uplift) / n + 0.5, 0.0), 1.0))


def assign_segment(tau: float, p0: float, tau_p25: float, tau_p75: float) -> str:
    """Assign uplift segment using percentile-based ITE thresholds."""
    if tau < tau_p25:
        return "Do Not Disturb"
    if p0 >= 0.75:
        return "Sure Thing"
    if tau >= tau_p75:
        return "Persuadable"
    return "Lost Cause"


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # ── Load feature arrays ──────────────────────────────────────────────────
    print("Loading Lending Club feature arrays...")
    X_train = np.load(f"{FEATURES_DIR}/lending_X_train.npy")
    X_test  = np.load(f"{FEATURES_DIR}/lending_X_test.npy")
    y_train = np.load(f"{FEATURES_DIR}/lending_y_train.npy")
    y_test  = np.load(f"{FEATURES_DIR}/lending_y_test.npy")
    T_train = np.load(f"{FEATURES_DIR}/lending_T_train.npy")
    T_test  = np.load(f"{FEATURES_DIR}/lending_T_test.npy")
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # ── Build T-Learner with LightGBM base models ────────────────────────────
    # Two independent models — one per treatment arm
    model_t = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=63, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1,
    )
    model_c = LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        num_leaves=63, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1,
    )

    print("Training T-Learner (two LightGBM models)...")
    learner = TLearner(model_t=model_t, model_c=model_c)
    learner.fit(X_train, T_train, y_train)
    print("  Training complete.")

    # ── Predict ITE on test set ──────────────────────────────────────────────
    tau_test = learner.effect(X_test)
    p0_test  = learner.predict_proba_control(X_test)
    print(f"  ITE stats — mean: {tau_test.mean():.4f}, std: {tau_test.std():.4f}")
    print(f"  ITE range — min: {tau_test.min():.4f}, max: {tau_test.max():.4f}")

    # ── AUUC ─────────────────────────────────────────────────────────────────
    auuc = auuc_score(tau_test, y_test, T_test)
    print(f"  AUUC: {auuc:.4f}  (target > 0.55)")

    # ── Percentile thresholds for segment assignment ──────────────────────────
    tau_p25 = float(np.percentile(tau_test, 25))
    tau_p75 = float(np.percentile(tau_test, 75))
    print(f"  ITE percentiles — p25: {tau_p25:.4f}, p75: {tau_p75:.4f}")

    # ── Segment distribution ─────────────────────────────────────────────────
    segments = [assign_segment(t, p, tau_p25, tau_p75)
                for t, p in zip(tau_test, p0_test)]
    seg_counts = {s: segments.count(s) for s in set(segments)}
    persuadable_pct = seg_counts.get("Persuadable", 0) / len(segments)
    print(f"  Segments: {seg_counts}")
    print(f"  Persuadable: {persuadable_pct:.1%}")

    # ── Save model ───────────────────────────────────────────────────────────
    joblib.dump(learner, f"{ARTIFACTS_DIR}/xlearner_lgbm.pkl")
    print(f"  Saved: {ARTIFACTS_DIR}/xlearner_lgbm.pkl")

    # ── Save metrics ─────────────────────────────────────────────────────────
    metrics = {
        "mean_ite": float(tau_test.mean()),
        "auuc": float(auuc),
        "ite_std": float(tau_test.std()),
        "ite_min": float(tau_test.min()),
        "ite_max": float(tau_test.max()),
        "ite_p25": tau_p25,
        "ite_p75": tau_p75,
        "segment_distribution": seg_counts,
        "persuadable_pct": float(persuadable_pct),
    }
    with open(f"{METRICS_DIR}/uplift_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved: {METRICS_DIR}/uplift_metrics.json")
    print("\nUplift model training complete.")


if __name__ == "__main__":
    main()
