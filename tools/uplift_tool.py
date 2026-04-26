"""
Phase 3 — LangChain @tool wrapper for the T-Learner uplift model.
Artifacts are loaded once at module level — never per-call.
"""

import json as _json
import joblib
import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel
from models.tlearner import TLearner  # noqa: F401 — needed for joblib unpickling

# ── Load artifacts at module level ───────────────────────────────────────────
_learner = joblib.load("models/artifacts/xlearner_lgbm.pkl")
_imputer = joblib.load("models/artifacts/lending_imputer.pkl")
_encoder = joblib.load("models/artifacts/lending_encoder.pkl")
_feature_names: list = joblib.load("models/artifacts/lending_feature_names.pkl")

# Load percentile thresholds computed during training
with open("models/metrics/uplift_metrics.json") as _f:
    _uplift_metrics = _json.load(_f)
_ITE_P25 = _uplift_metrics.get("ite_p25", -0.01)
_ITE_P75 = _uplift_metrics.get("ite_p75", 0.01)

# Categorical columns requiring OrdinalEncoder before float conversion
_CAT_COLS = ["grade", "home_ownership", "purpose"]
_CAT_INDICES = [_feature_names.index(c) for c in _CAT_COLS if c in _feature_names]


def _assign_segment(tau: float, p0: float) -> str:
    """Map ITE and baseline probability to a named uplift segment."""
    if tau < _ITE_P25:
        return "Do Not Disturb"
    if p0 >= 0.75:
        return "Sure Thing"
    if tau >= _ITE_P75:
        return "Persuadable"
    return "Lost Cause"


def _build_feature_vector(applicant_features: dict) -> np.ndarray:
    """
    Build a (1, n_features) float array.
    Encodes categorical columns BEFORE casting to float to avoid string→float errors.
    Also computes engineered features (rate_sensitivity, debt_burden, income_adequacy).
    """
    # Compute engineered features if not provided
    feats = dict(applicant_features)
    fico = float(feats.get("fico_range_low", 680))
    int_rate = float(feats.get("int_rate", 15.0))
    dti = float(feats.get("dti", 20.0))
    loan_amnt = float(feats.get("loan_amnt", 10000))
    annual_inc = float(feats.get("annual_inc", 50000))

    feats.setdefault("rate_sensitivity", (850 - fico) * int_rate / 100)
    feats.setdefault("debt_burden", dti * loan_amnt / 10_000)
    feats.setdefault("income_adequacy", loan_amnt / (annual_inc + 1))

    # Collect raw values in feature order
    raw_row = [feats.get(col, None) for col in _feature_names]

    # Encode categorical columns using saved OrdinalEncoder
    if _CAT_INDICES:
        cat_raw = [[str(raw_row[i]) if raw_row[i] is not None else "nan"
                    for i in _CAT_INDICES]]
        cat_encoded = _encoder.transform(cat_raw)[0]
        for list_pos, feat_idx in enumerate(_CAT_INDICES):
            raw_row[feat_idx] = cat_encoded[list_pos]

    # Convert to float array
    X = np.array([[float(v) if v is not None else np.nan for v in raw_row]],
                 dtype=np.float64)
    return X


class UpliftInput(BaseModel):
    applicant_features: dict


@tool("run_uplift_model", args_schema=UpliftInput)
def run_uplift_model(applicant_features: dict) -> dict:
    """
    Predict the causal uplift score for a loan applicant using the T-Learner.
    ITE = P(repay | rate_offer, X) - P(repay | no_offer, X)
    Returns uplift_score, segment, baseline_repay_prob, confidence_interval.
    """
    try:
        X = _build_feature_vector(applicant_features)
        X = _imputer.transform(X)

        # ITE: difference between treated and control outcome probabilities
        tau = float(_learner.effect(X).flatten()[0])

        # Baseline repay probability (control group — no rate incentive)
        p0 = float(_learner.predict_proba_control(X)[0])

        ci_half = float(_uplift_metrics.get("ite_std", 0.05))
        ci = [round(tau - ci_half, 4), round(tau + ci_half, 4)]

        segment = _assign_segment(tau, p0)

        return {
            "uplift_score": round(tau, 4),
            "segment": segment,
            "baseline_repay_prob": round(p0, 4),
            "confidence_interval": ci,
            "ci_width": round(ci[1] - ci[0], 4),
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
