"""
Phase 4 — Explainability agent: SHAP narrative + PDF audit report.
Uses Claude (via LangChain) to generate a plain-English explanation,
then builds a PDF with reportlab.
"""

import os
import datetime
from agents.state import LoanState

# ── LangChain / Anthropic ────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

# ── PDF generation ───────────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# ── Config ───────────────────────────────────────────────────────────────────
AUDIT_PDF_DIR = "reports/audit_pdfs"
os.makedirs(AUDIT_PDF_DIR, exist_ok=True)

# Initialise Claude — model loaded once at module level
_llm = ChatAnthropic(model="claude-3-5-haiku-20241022", max_tokens=400)


def _build_narrative_prompt(state: LoanState) -> str:
    """
    Build a structured prompt for Claude to generate a 150-word audit explanation.
    Strictly constrained to SHAP values and decision — no hallucination surface.
    """
    shap_top3 = state.get("shap_top3") or []
    shap_lines = "\n".join(
        f"  - {f['feature']}: value={f['value']:.2f}, SHAP={f['shap_value']:.4f} ({f['direction']})"
        for f in shap_top3
    )
    return f"""You are a credit risk compliance officer writing a regulatory audit explanation.
Write a plain-English explanation (maximum 150 words) for the following loan decision.
You MUST only reference the SHAP features listed below. Do not invent other reasons.

Decision: {state.get('decision', 'N/A')}
Reason: {state.get('decision_reason', 'N/A')}
Default Probability: {state.get('default_probability', 'N/A')}
Fraud Score: {state.get('fraud_score', 'N/A')}
Uplift Segment: {state.get('segment', 'N/A')}

Top 3 SHAP factors driving the credit risk score:
{shap_lines}

Write the explanation now:"""


def _generate_narrative(state: LoanState) -> str:
    """Call Claude to generate the audit narrative. Returns plain text."""
    try:
        prompt = _build_narrative_prompt(state)
        response = _llm.invoke([HumanMessage(content=prompt)])
        narrative = response.content.strip()

        # Post-generation validation: ensure top features are mentioned
        shap_top3 = state.get("shap_top3") or []
        for factor in shap_top3[:2]:  # At least top 2 must appear
            if factor["feature"] not in narrative:
                narrative += f"\n[Key factor: {factor['feature']} ({factor['direction']})]"

        return narrative
    except Exception as e:
        # Fallback: template-based narrative if LLM fails
        return (
            f"Decision: {state.get('decision', 'N/A')}. "
            f"Default probability: {state.get('default_probability', 'N/A'):.2%}. "
            f"Fraud score: {state.get('fraud_score', 'N/A'):.2f}. "
            f"Uplift segment: {state.get('segment', 'N/A')}. "
            f"[LLM narrative unavailable: {e}]"
        )


def _generate_pdf(state: LoanState, narrative: str, app_id: str) -> str:
    """
    Build a PDF audit report using reportlab.
    Includes: decision header, risk scores grid, SHAP factors, LLM narrative, timestamp.
    """
    pdf_path = f"{AUDIT_PDF_DIR}/{app_id}.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story = []

    # ── Header ────────────────────────────────────────────────────────────────
    decision = state.get("decision", "N/A").upper()
    decision_color = {
        "APPROVE": colors.green, "APPROVE_WITH_RATE": colors.green,
        "APPROVE_STANDARD": colors.green, "DECLINE": colors.red,
        "HUMAN_REVIEW": colors.orange,
    }.get(decision, colors.black)

    header_style = ParagraphStyle("header", parent=styles["Title"], textColor=decision_color)
    story.append(Paragraph(f"Loan Decision: {decision}", header_style))
    story.append(Spacer(1, 0.3*cm))
    story.append(Paragraph(f"Application ID: {app_id}", styles["Normal"]))
    story.append(Paragraph(
        f"Generated: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        styles["Normal"]
    ))
    story.append(Spacer(1, 0.5*cm))

    # ── Decision reason ───────────────────────────────────────────────────────
    story.append(Paragraph("Decision Reason", styles["Heading2"]))
    story.append(Paragraph(state.get("decision_reason", "N/A"), styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Risk scores grid ──────────────────────────────────────────────────────
    story.append(Paragraph("Risk Scores", styles["Heading2"]))
    score_data = [
        ["Metric", "Value", "Band/Level"],
        ["Default Probability", f"{state.get('default_probability', 'N/A'):.2%}" if state.get('default_probability') else "N/A", state.get("risk_band", "N/A")],
        ["Fraud Score", f"{state.get('fraud_score', 'N/A'):.4f}" if state.get('fraud_score') else "N/A", state.get("risk_level", "N/A")],
        ["Uplift Score (ITE)", f"{state.get('uplift_score', 'N/A'):.4f}" if state.get('uplift_score') else "N/A", state.get("segment", "N/A")],
    ]
    score_table = Table(score_data, colWidths=[6*cm, 4*cm, 5*cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f3f4")]),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.5*cm))

    # ── SHAP top-3 factors ────────────────────────────────────────────────────
    story.append(Paragraph("Top 3 SHAP Factors (Credit Risk Model)", styles["Heading2"]))
    shap_top3 = state.get("shap_top3") or []
    shap_data = [["Feature", "Value", "SHAP Impact", "Direction"]]
    for f in shap_top3:
        shap_data.append([
            f["feature"], f"{f['value']:.4f}",
            f"{f['shap_value']:+.4f}", f["direction"]
        ])
    if shap_data:
        shap_table = Table(shap_data, colWidths=[6*cm, 3*cm, 4*cm, 5*cm])
        shap_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a5276")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        story.append(shap_table)
    story.append(Spacer(1, 0.5*cm))

    # ── LLM narrative ─────────────────────────────────────────────────────────
    story.append(Paragraph("Audit Narrative (AI-Generated)", styles["Heading2"]))
    story.append(Paragraph(narrative, styles["Normal"]))
    story.append(Spacer(1, 0.5*cm))

    # ── Footer ────────────────────────────────────────────────────────────────
    story.append(Paragraph(
        "This report was generated automatically by the Multi-Agent Loan Intelligence System. "
        "All decisions are subject to human review and applicable regulations.",
        ParagraphStyle("footer", parent=styles["Normal"], fontSize=8, textColor=colors.grey)
    ))

    doc.build(story)
    return pdf_path


def explainability_node(state: LoanState) -> LoanState:
    """
    Generate SHAP narrative via Claude and produce a PDF audit report.
    This is the final node — graph ends after this.
    """
    # Generate a unique application ID from timestamp
    app_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")

    # ── Generate LLM narrative ────────────────────────────────────────────────
    narrative = _generate_narrative(state)

    # ── Generate PDF audit report ─────────────────────────────────────────────
    pdf_path = _generate_pdf(state, narrative, app_id)

    return {
        **state,
        "shap_narrative": narrative,
        "audit_pdf_path": pdf_path,
    }
