"""
Shared T-Learner class — imported by both train_uplift.py and uplift_tool.py.
Must live in a stable importable location so joblib.load() can unpickle it.
"""

import numpy as np


class TLearner:
    """
    T-Learner uplift model: trains separate outcome models for treatment/control.
    ITE(x) = mu1(x) - mu0(x)
    where mu1 = P(Y=1 | T=1, X) and mu0 = P(Y=1 | T=0, X).
    """

    def __init__(self, model_t, model_c):
        self.model_t = model_t   # Outcome model for treated group (T=1)
        self.model_c = model_c   # Outcome model for control group (T=0)

    def fit(self, X: np.ndarray, T: np.ndarray, y: np.ndarray) -> "TLearner":
        """Fit separate models on treated and control subgroups."""
        mask_t = T == 1
        mask_c = T == 0
        print(f"  Treated: {mask_t.sum():,}  |  Control: {mask_c.sum():,}")
        self.model_t.fit(X[mask_t], y[mask_t])
        self.model_c.fit(X[mask_c], y[mask_c])
        return self

    def effect(self, X: np.ndarray) -> np.ndarray:
        """ITE = P(repay|T=1,X) - P(repay|T=0,X)."""
        p1 = self.model_t.predict_proba(X)[:, 1]
        p0 = self.model_c.predict_proba(X)[:, 1]
        return p1 - p0

    def predict_proba_control(self, X: np.ndarray) -> np.ndarray:
        """Baseline repay probability under control (no rate incentive)."""
        return self.model_c.predict_proba(X)[:, 1]
