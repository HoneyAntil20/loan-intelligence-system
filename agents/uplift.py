"""
Phase 4 — Uplift specialist agent node.
Calls the X-Learner uplift @tool and writes results to shared LoanState.
"""

from agents.state import LoanState
from tools.uplift_tool import run_uplift_model


def uplift_node(state: LoanState) -> LoanState:
    """
    Invoke the uplift model tool and update shared state.
    On error, appends to state['errors'] for supervisor routing.
    """
    try:
        result = run_uplift_model.invoke(
            {"applicant_features": state["applicant_features"]}
        )

        if result.get("status") == "error":
            errors = state.get("errors") or []
            errors.append(f"uplift_agent: {result['error']}")
            return {**state, "errors": errors}

        return {
            **state,
            "uplift_score": result["uplift_score"],
            "segment": result["segment"],
            "baseline_repay_prob": result["baseline_repay_prob"],
            "confidence_interval": result["confidence_interval"],
        }
    except Exception as e:
        errors = state.get("errors") or []
        errors.append(f"uplift_agent: {str(e)}")
        return {**state, "errors": errors}
