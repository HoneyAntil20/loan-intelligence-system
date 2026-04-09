"""
Phase 4 — Credit Risk specialist agent node.
Calls the credit risk @tool and writes results to shared LoanState.
"""

from agents.state import LoanState
from tools.credit_risk_tool import run_credit_risk_model


def credit_risk_node(state: LoanState) -> LoanState:
    """
    Invoke the credit risk model tool and update shared state.
    On error, appends to state['errors'] for supervisor routing.
    """
    try:
        result = run_credit_risk_model.invoke(
            {"applicant_features": state["applicant_features"]}
        )

        if result.get("status") == "error":
            errors = state.get("errors") or []
            errors.append(f"credit_risk_agent: {result['error']}")
            return {**state, "errors": errors}

        return {
            **state,
            "default_probability": result["default_probability"],
            "risk_band": result["risk_band"],
            "shap_top3": result["shap_top3"],
        }
    except Exception as e:
        errors = state.get("errors") or []
        errors.append(f"credit_risk_agent: {str(e)}")
        return {**state, "errors": errors}
