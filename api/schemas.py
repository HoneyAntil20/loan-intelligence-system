"""
Phase 5 — Pydantic request/response schemas for the FastAPI REST layer.
Defines the contract between the HTTP client and the LangGraph pipeline.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    """Input schema for POST /evaluate-loan."""

    # ── Core loan fields ──────────────────────────────────────────────────────
    loan_amnt: float = Field(..., description="Requested loan amount in USD", gt=0)
    fico_range_low: float = Field(..., description="Lower bound of applicant FICO score", ge=300, le=850)
    dti: float = Field(..., description="Debt-to-income ratio (%)", ge=0)
    annual_inc: float = Field(..., description="Annual income in USD", gt=0)
    int_rate: Optional[float] = Field(None, description="Proposed interest rate (%)")

    # ── Categorical fields ────────────────────────────────────────────────────
    grade: Optional[str] = Field(None, description="Loan grade (A-G)")
    purpose: Optional[str] = Field(None, description="Loan purpose (e.g. debt_consolidation)")
    term: Optional[str] = Field(None, description="Loan term (36 months / 60 months)")
    home_ownership: Optional[str] = Field(None, description="RENT / OWN / MORTGAGE")

    # ── Employment ────────────────────────────────────────────────────────────
    emp_length_num: Optional[float] = Field(None, description="Employment length in years")

    # ── Additional fields (passed through to models as-is) ───────────────────
    installment: Optional[float] = None
    open_acc: Optional[float] = None
    pub_rec: Optional[float] = None
    revol_bal: Optional[float] = None
    revol_util: Optional[float] = None
    total_acc: Optional[float] = None

    # ── Fraud model fields (IEEE-CIS aligned) ─────────────────────────────────
    TransactionAmt: Optional[float] = None
    ProductCD: Optional[str] = None
    card4: Optional[str] = None
    card6: Optional[str] = None
    P_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None

    # ── Home Credit fields ────────────────────────────────────────────────────
    AMT_CREDIT: Optional[float] = None
    AMT_INCOME_TOTAL: Optional[float] = None
    AMT_ANNUITY: Optional[float] = None
    NAME_CONTRACT_TYPE: Optional[str] = None
    CODE_GENDER: Optional[str] = None
    FLAG_OWN_CAR: Optional[str] = None
    FLAG_OWN_REALTY: Optional[str] = None


class SHAPFactor(BaseModel):
    """A single SHAP explanatory factor."""
    feature: str
    value: float
    shap_value: float
    direction: str


class LoanDecisionResponse(BaseModel):
    """Output schema returned by POST /evaluate-loan."""

    # ── Decision ──────────────────────────────────────────────────────────────
    decision: str = Field(..., description="approve / approve_with_rate / approve_standard / decline / human_review")
    decision_reason: str

    # ── Risk scores ───────────────────────────────────────────────────────────
    default_probability: Optional[float] = None
    risk_band: Optional[str] = None
    fraud_score: Optional[float] = None
    risk_level: Optional[str] = None

    # ── Uplift ────────────────────────────────────────────────────────────────
    uplift_score: Optional[float] = None
    segment: Optional[str] = None
    baseline_repay_prob: Optional[float] = None
    confidence_interval: Optional[List[float]] = None

    # ── Explainability ────────────────────────────────────────────────────────
    shap_top3: Optional[List[SHAPFactor]] = None
    shap_narrative: Optional[str] = None
    audit_pdf_url: Optional[str] = None

    # ── Meta ──────────────────────────────────────────────────────────────────
    errors: Optional[List[str]] = None
