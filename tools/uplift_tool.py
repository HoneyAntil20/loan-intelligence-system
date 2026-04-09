"""
Phase 3 — LangChain @tool wrapper for the X-Learner uplift model.
Artifacts are loaded once at module level — never per-call.
"""

import json
import joblib
import numpy as np
from typing import Any
from langchain_core.tools import tool
from pydantic import BaseModel

# ── Load artifacts at module level (startup cost, not per-request) ───────────
_learner = joblib.load("models/artifacts/xlearner_lgbm.pkl")
_imputer = joblib.load("models/artifacts/lending_imputer.pkl")
_encoder = joblib.load("models/artifacts/lending_encoder.pkl")
_feature_names: list = joblib.load("models/artifacts/lending_feature_names.pkl")

# Uplift segment thresholds (section 4.1 of the plan)
_CAT_COLS = ["grade", "home_ownership", "purpose"]


def _assign_segment(tau: float, p0: float) -> str:
    """Map ITE and baseline probability to a named uplift segment."""
    if tau < -0.03:
        return "Do Not Disturb"
    if p0 >= 0.75:
        return "Sure Thing"
    if tau >= 0.05 and p0 < 0.75:
        return "Persuadable"
    return "Lost Cause"


class UpliftInput(BaseModel):
    applicant_features: dict


@tool("run_uplift_model", args_schema=UpliftInput)
def run_uplift_model(applicant_features: dict) -> dict:
    """
    Predict the causal uplift score for a loan applicant.
    Returns uplift_score, segment, baseline_repay_prob, confidence_interval.
    """
    try:
        # Build feature vector in the correct column order
        row = {col: applicant_features.get(col, np.nan) for col in _feature_names}
        X = np.array([[row[col] for col in _feature_names]], dtype=float)

        # Encode categorical columns using the saved OrdinalEncoder
        cat_indices = [_feature_names.index(c) for c in _CAT_COLS if c in _feature_names]
        if cat_indices:
            cat_vals = X[:, cat_indices]
            X[:, cat_indices] = _encoder.transform(cat_vals)

        # Impute missing values
        X = _imputer.transform(X)

        # Predict ITE (individual treatment effect = uplift score)
        tau = float(_learner.effect(X).flatten()[0])

        # Baseline repay probability (control group model)
        p0 = float(_learner.models_t[0].predict_proba(X)[0, 1])

        # Bootstrap confidence interval approximation (±1.96 * model std)
        ci_half = 0.05  # Placeholder — replace with bootstrap in production
        ci = [round(tau - ci_half, 4), round(tau + ci_half, 4)]
        ci_width = ci[1] - ci[0]

        segment = _assign_segment(tau, p0)

        return {
            "uplift_score": round(tau, 4),
            "segment": segment,
            "baseline_repay_prob": round(p0, 4),
            "confidence_interval": ci,
            "ci_width": round(ci_width, 4),
            "status": "success",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
