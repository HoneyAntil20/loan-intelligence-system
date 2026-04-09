"""
Phase 4 — Supervisor agent with conditional routing rules.
Applies explicit business logic to reach a final auditable decision.
Routing rules are from section 5.3 of the implementation plan.
"""

from agents.state import LoanState


def supervisor_node(state: LoanState) -> LoanState:
    """
    Apply conditional routing rules to produce a final loan decision.

    Priority order (highest to lowest):
    1. Agent errors → human_review
    2. Fraud score > 0.85 → decline (immediate, skips uplift consideration)
    3. Default probability > 0.70 → decline (too risky regardless of uplift)
    4. CI width > 0.15 → human_review (model is uncertain)
    5. Uplift segment routing → approve / approve_with_rate / decline
    """
    errors = state.get("errors") or []
    fraud_score = state.get("fraud_score") or 0.0
    default_prob = state.get("default_probability") or 0.0
    segment = state.get("segment") or "Lost Cause"
    ci = state.get("confidence_interval") or [0.0, 0.0]
    ci_width = ci[1] - ci[0] if len(ci) == 2 else 0.0

    # ── Rule 1: Agent errors → human review ──────────────────────────────────
    if errors:
        return {
            **state,
            "decision": "human_review",
            "decision_reason": f"Agent error(s) require manual review: {'; '.join(errors)}",
        }

    # ── Rule 2: High fraud score → immediate decline ──────────────────────────
    if fraud_score > 0.85:
        return {
            **state,
            "decision": "decline",
            "decision_reason": (
                f"Application declined: fraud score {fraud_score:.2f} exceeds threshold 0.85. "
                "Fraud risk overrides all other signals."
            ),
        }

    # ── Rule 3: High default probability → decline ────────────────────────────
    if default_prob > 0.70:
        return {
            **state,
            "decision": "decline",
            "decision_reason": (
                f"Application declined: default probability {default_prob:.2f} exceeds 0.70. "
                "Credit risk is too high regardless of rate offer."
            ),
        }

    # ── Rule 4: Wide confidence interval → human review ──────────────────────
    if ci_width > 0.15:
        return {
            **state,
            "decision": "human_review",
            "decision_reason": (
                f"Referred for human review: uplift model confidence interval width {ci_width:.3f} "
                "exceeds 0.15. Model uncertainty is too high to automate this decision."
            ),
        }

    # ── Rule 5: Uplift segment routing ───────────────────────────────────────
    segment_routing = {
        "Persuadable": (
            "approve_with_rate",
            "Approved with rate offer: applicant is in the Persuadable segment — "
            "a better rate meaningfully increases repayment probability.",
        ),
        "Sure Thing": (
            "approve_standard",
            "Approved at standard rate: applicant is a Sure Thing — "
            "will repay regardless of rate incentive.",
        ),
        "Lost Cause": (
            "decline",
            "Application declined: applicant is in the Lost Cause segment — "
            "low baseline repayment probability and rate offer provides no uplift.",
        ),
        "Do Not Disturb": (
            "decline",
            "Application declined: applicant is in the Do Not Disturb segment — "
            "a rate offer would be counterproductive.",
        ),
    }

    decision, reason = segment_routing.get(
        segment,
        ("human_review", f"Unknown segment '{segment}' — referred for manual review.")
    )

    return {**state, "decision": decision, "decision_reason": reason}
