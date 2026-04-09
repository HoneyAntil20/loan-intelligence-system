"""
Phase 4 — LangGraph StateGraph definition.
Wires all agent nodes together with conditional routing.
Usage:
    from agents.graph import build_graph
    graph = build_graph()
    result = graph.invoke({"applicant_features": {...}})
"""

from langgraph.graph import StateGraph, END
from agents.state import LoanState
from agents.credit_risk import credit_risk_node
from agents.fraud import fraud_node
from agents.uplift import uplift_node
from agents.supervisor import supervisor_node
from agents.explainability import explainability_node


def _route_after_supervisor(state: LoanState) -> str:
    """
    Conditional edge: after supervisor decides, always run explainability.
    In future, could skip explainability for certain fast-path declines.
    """
    return "explainability"


def build_graph() -> StateGraph:
    """
    Build and compile the LangGraph multi-agent pipeline.

    Graph topology:
        credit_risk ─┐
        fraud        ├─→ supervisor ─→ explainability ─→ END
        uplift      ─┘
    """
    builder = StateGraph(LoanState)

    # ── Register agent nodes ──────────────────────────────────────────────────
    builder.add_node("credit_risk", credit_risk_node)
    builder.add_node("fraud", fraud_node)
    builder.add_node("uplift", uplift_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("explainability", explainability_node)

    # ── Entry point: run all three specialist agents in sequence ──────────────
    # (LangGraph parallel execution requires async; sequential is safe for now)
    builder.set_entry_point("credit_risk")
    builder.add_edge("credit_risk", "fraud")
    builder.add_edge("fraud", "uplift")

    # ── Supervisor receives all agent outputs ─────────────────────────────────
    builder.add_edge("uplift", "supervisor")

    # ── Conditional routing after supervisor ─────────────────────────────────
    builder.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {"explainability": "explainability"},
    )

    # ── Explainability is the terminal node ───────────────────────────────────
    builder.add_edge("explainability", END)

    return builder.compile()


# ── Convenience: pre-compiled graph instance ─────────────────────────────────
loan_graph = build_graph()
