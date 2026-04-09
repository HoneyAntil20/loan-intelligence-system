"""
Phase 3 — LangChain @tool wrapper for the LightGBM fraud classifier.
Artifacts are loaded once at module level — never per-call.
"""

import json
import joblib
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel

# ── Load artifacts at module level ───────────────────────────────────────────
_model = joblib.load("models/artifacts/fraud_lgbm.pkl")
_feature_names: list = joblib.load("models/artifacts/fraud_feature_names.pkl")
_imputer = joblib.load("models/artifacts/fraud_imputer.pkl")

with open("models/metrics/fraud_threshold.json") as f:
    _threshold = json.load(f)["threshold"]


def _get_risk_level(score: float) -> str:
    """Map fraud probability to a human-readable risk level."""
    if score >= 0.85:
        return "CRITICAL"
    if score >= 0.60:
        return "HIGH"
    if score >= 0.30:
        return "MEDIUM"
    return "LOW"


def _detect_anomalies(applicant_features: dict) -> list:
    """
    Simple rule-based anomaly flags to supplement the model score.
    These are domain heuristics, not model outputs.
    """
    flags = []
    amt = applicant_features.get("TransactionAmt", 0)
    if amt and float(amt) > 5000:
        flags.append("high_transaction_amount")
    if applicant_features.get("addr_dist", 0) and float(applicant_features.get("addr_dist", 0)) > 500:
        flags.append("large_address_distance")
    return flags


class FraudInput(BaseModel):
    applicant_features: dict


@tool("run_fraud_model", args_schema=FraudInput)
def run_fraud_model(applicant_features: dict) -> dict:
    """
    Score a loan application for fraud risk using the IEEE-CIS trained LightGBM model.
    Returns fraud_score, risk_level, anomaly_flags.
    """
    try:
        # Build feature vector aligned to training column order
        row = [applicant_features.get(col, np.nan) for col in _feature_names]
        X = np.array([row], dtype=float)

        # Impute missing values using the saved imputer
        X = _imputer.transform(X)

        # Predict fraud probability
        fraud_score = float(_model.predict_proba(X)[0, 1])
        risk_level = _get_risk_level(fraud_score)
        anomaly_flags = _detect_anomalies(applicant_features)

        return {
            "fraud_score": round(fraud_score, 4),
            "risk_level": risk_level,
            "anomaly_flags": anomaly_flags,
            "threshold_used": round(_threshold, 4),
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
