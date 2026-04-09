"""
Phase 5 — FastAPI REST layer wrapping the LangGraph multi-agent pipeline.
All model artifacts are loaded at startup — never per-request.
Run: uvicorn api.main:app --reload --port 8000
"""

import os
import json
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

from api.schemas import LoanApplicationRequest, LoanDecisionResponse

# ── Load environment variables (.env) ────────────────────────────────────────
load_dotenv()

# ── Global state: models loaded flag ─────────────────────────────────────────
_models_loaded = False
_graph = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: load all model artifacts into memory once.
    This is critical — loading per-request would add 2-5s latency.
    """
    global _models_loaded, _graph
    try:
        # Import graph (triggers artifact loading in tool modules)
        from agents.graph import loan_graph
        _graph = loan_graph
        _models_loaded = True
        print("✓ All model artifacts loaded successfully.")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        _models_loaded = False
    yield
    # Shutdown: nothing to clean up for in-memory models


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Multi-Agent Loan Intelligence System",
    description="LangGraph-powered loan evaluation with credit risk, fraud detection, and uplift scoring.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve generated PDF reports as static files
os.makedirs("reports/audit_pdfs", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")


@app.get("/health")
def health_check() -> dict:
    """Health endpoint — confirms models are loaded and service is ready."""
    return {"status": "ok", "models_loaded": _models_loaded}


@app.get("/metrics")
def get_metrics() -> dict:
    """Return latest evaluation metrics from all three models."""
    metrics = {}
    metrics_dir = "models/metrics"
    for fname in ["uplift_metrics.json", "fraud_metrics.json", "credit_metrics.json"]:
        fpath = os.path.join(metrics_dir, fname)
        if os.path.exists(fpath):
            with open(fpath) as f:
                metrics[fname.replace("_metrics.json", "")] = json.load(f)
    return metrics


@app.post("/evaluate-loan", response_model=LoanDecisionResponse)
def evaluate_loan(request: LoanApplicationRequest) -> LoanDecisionResponse:
    """
    Main endpoint: evaluate a loan application through the full multi-agent pipeline.
    Runs credit risk, fraud detection, and uplift agents, then supervisor + explainability.
    """
    if not _models_loaded or _graph is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Check /health.")

    # ── Build initial LoanState from request ──────────────────────────────────
    applicant_features = request.model_dump(exclude_none=False)

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
    }

    # ── Run the LangGraph pipeline ────────────────────────────────────────────
    try:
        final_state = _graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # ── Build audit PDF URL ───────────────────────────────────────────────────
    audit_pdf_url = None
    if final_state.get("audit_pdf_path"):
        # Convert local path to URL path
        pdf_rel = final_state["audit_pdf_path"].replace("\\", "/")
        audit_pdf_url = f"/{pdf_rel}"

    # ── Return structured response ────────────────────────────────────────────
    return LoanDecisionResponse(
        decision=final_state.get("decision", "error"),
        decision_reason=final_state.get("decision_reason", "Pipeline did not complete."),
        default_probability=final_state.get("default_probability"),
        risk_band=final_state.get("risk_band"),
        fraud_score=final_state.get("fraud_score"),
        risk_level=final_state.get("risk_level"),
        uplift_score=final_state.get("uplift_score"),
        segment=final_state.get("segment"),
        baseline_repay_prob=final_state.get("baseline_repay_prob"),
        confidence_interval=final_state.get("confidence_interval"),
        shap_top3=final_state.get("shap_top3"),
        shap_narrative=final_state.get("shap_narrative"),
        audit_pdf_url=audit_pdf_url,
        errors=final_state.get("errors"),
    )
