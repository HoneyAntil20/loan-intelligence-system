"""
Phase 4 — Fraud Detection specialist agent node.
Calls the fraud model @tool and writes results to shared LoanState.
"""

from agents.state import LoanState
from tools.fraud_tool import run_fraud_model


def fraud_node(state: LoanState) -> LoanState:
    """
    Invoke the fraud detection model tool and update shared state.
    On error, appends to state['errors'] for supervisor routing.
    """
    try:
        result = run_fraud_model.invoke(
            {"applicant_features": state["applicant_features"]}
        )

        if result.get("status") == "error":
            errors = state.get("errors") or []
            errors.append(f"fraud_agent: {result['error']}")
            return {**state, "errors": errors}

        return {
            **state,
            "fraud_score": result["fraud_score"],
            "risk_level": result["risk_level"],
            "anomaly_flags": result["anomaly_flags"],
        }
    except Exception as e:
        errors = state.get("errors") or []
        errors.append(f"fraud_agent: {str(e)}")
        return {**state, "errors": errors}
