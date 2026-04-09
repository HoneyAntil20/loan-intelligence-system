"""
Phase 5 — LangSmith tracing wrapper.
Decorates the graph invocation with metadata for per-decision replay.
Requires LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY in .env
"""

import datetime
import uuid
from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()


@traceable(name="loan_evaluation")
def evaluate_with_tracing(graph, applicant_features: dict, application_id: str = None) -> dict:
    """
    Invoke the LangGraph pipeline with LangSmith tracing metadata.
    Every call creates a trace with application_id and timestamp for audit replay.
    """
    if application_id is None:
        application_id = str(uuid.uuid4())

    initial_state = {
        "applicant_features": applicant_features,
        "default_probability": None,
        "risk_band": None,
        "shap_top3": None,
        "fraud_score": None,
        "risk_level": None,
        "anomaly_flags": None,
        "uplift_score": None,
        "segment": None,
        "baseline_repay_prob": None,
        "confidence_interval": None,
        "decision": None,
        "decision_reason": None,
        "shap_narrative": None,
        "audit_pdf_path": None,
        "errors": None,
        # Metadata passed to LangSmith trace
        "_metadata": {
            "application_id": application_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        },
    }

    return graph.invoke(initial_state)
