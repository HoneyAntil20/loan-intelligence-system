"""
Phase 4 — Shared TypedDict state schema for the LangGraph multi-agent system.
Every agent reads from and writes to this single shared state object.
LangGraph manages its lifecycle across the graph.
"""

from typing import TypedDict, Optional, List


class LoanState(TypedDict):
    # ── Input ────────────────────────────────────────────────────────────────
    applicant_features: dict          # Raw applicant data passed to all agents

    # ── Credit Risk Agent outputs ────────────────────────────────────────────
    default_probability: Optional[float]   # Probability of loan default (0-1)
    risk_band: Optional[str]               # LOW / LOW-MEDIUM / MEDIUM / HIGH
    shap_top3: Optional[list]              # Top 3 SHAP features driving the score

    # ── Fraud Detection Agent outputs ────────────────────────────────────────
    fraud_score: Optional[float]           # Fraud probability (0-1)
    risk_level: Optional[str]             # LOW / MEDIUM / HIGH / CRITICAL
    anomaly_flags: Optional[list]          # Rule-based anomaly signals

    # ── Uplift Agent outputs ─────────────────────────────────────────────────
    uplift_score: Optional[float]          # Individual treatment effect (ITE)
    segment: Optional[str]                 # Persuadable / Sure Thing / Lost Cause / Do Not Disturb
    baseline_repay_prob: Optional[float]   # Repay probability without rate intervention
    confidence_interval: Optional[list]    # [lower, upper] CI for ITE estimate

    # ── Supervisor outputs ───────────────────────────────────────────────────
    decision: Optional[str]               # approve / approve_with_rate / decline / human_review
    decision_reason: Optional[str]        # Plain-text reason for the decision

    # ── Explainability Agent outputs ─────────────────────────────────────────
    shap_narrative: Optional[str]          # LLM-generated plain-English explanation
    audit_pdf_path: Optional[str]          # Path to generated PDF audit report

    # ── Error tracking ───────────────────────────────────────────────────────
    errors: Optional[list]                 # Agent errors — routes to human_review
