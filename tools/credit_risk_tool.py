"""
Phase 3 — LangChain @tool wrapper for the LightGBM credit risk model.
Artifacts are loaded once at module level — never per-call.
"""

import joblib
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel

# ── Load artifacts at module level ───────────────────────────────────────────
_model = joblib.load("models/artifacts/credit_lgbm.pkl")
_explainer = joblib.load("models/artifacts/credit_shap_explainer.pkl")
_feature_names: list = joblib.load("models/artifacts/credit_feature_names.pkl")
_imputer = joblib.load("models/artifacts/credit_imputer.pkl")


def _get_risk_band(prob: float) -> str:
    """Map default probability to a risk band label."""
    if prob >= 0.70:
        return "HIGH"
    if prob >= 0.40:
        return "MEDIUM"
    if prob >= 0.20:
        return "LOW-MEDIUM"
    return "LOW"


def _get_shap_top3(X: np.ndarray) -> list:
    """
    Compute SHAP values and return the top 3 features by absolute impact.
    Returns list of {feature, value, shap_value, direction}.
    """
    shap_values = _explainer.shap_values(X)
    # For binary classification, shap_values may be a list [neg_class, pos_class]
    if isinstance(shap_values, list):
        sv = shap_values[1][0]  # Positive class (default=1)
    else:
        sv = shap_values[0]

    # Sort by absolute SHAP value descending
    top_indices = np.argsort(np.abs(sv))[::-1][:3]
    top3 = []
    for idx in top_indices:
        top3.append({
            "feature": _feature_names[idx],
            "value": float(X[0, idx]),
            "shap_value": float(sv[idx]),
            "direction": "increases_risk" if sv[idx] > 0 else "decreases_risk",
        })
    return top3


class CreditRiskInput(BaseModel):
    applicant_features: dict


@tool("run_credit_risk_model", args_schema=CreditRiskInput)
def run_credit_risk_model(applicant_features: dict) -> dict:
    """
    Predict default probability for a loan applicant using the Home Credit trained model.
    Returns default_probability, risk_band, shap_top3.
    """
    try:
        # Build feature vector aligned to training column order
        row = [applicant_features.get(col, np.nan) for col in _feature_names]
        X = np.array([row], dtype=float)

        # Impute missing values
        X = _imputer.transform(X)

        # Predict default probability
        default_prob = float(_model.predict_proba(X)[0, 1])
        risk_band = _get_risk_band(default_prob)

        # SHAP top-3 explanatory features
        shap_top3 = _get_shap_top3(X)

        return {
            "default_probability": round(default_prob, 4),
            "risk_band": risk_band,
            "shap_top3": shap_top3,
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
