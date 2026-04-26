"""
Phase 5 — Streamlit demo frontend for the Multi-Agent Loan Intelligence System.
Submits loan applications to the FastAPI backend and renders decisions with SHAP charts.
Run: streamlit run streamlit_app.py
"""

import httpx
import streamlit as st
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Loan Intelligence System",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Multi-Agent Loan Intelligence System")
st.caption("Powered by LangGraph · LightGBM · X-Learner · SHAP · Gemini")

# ── Sidebar: Application Input Form ──────────────────────────────────────────
with st.sidebar:
    st.header("Loan Application")

    loan_amnt = st.number_input("Loan Amount ($)", min_value=500.0, max_value=500_000.0, value=15_000.0, step=500.0)
    annual_inc = st.number_input("Annual Income ($)", min_value=1_000.0, max_value=10_000_000.0, value=65_000.0, step=1_000.0)
    dti = st.number_input("Debt-to-Income Ratio (%)", min_value=0.0, max_value=100.0, value=18.5, step=0.5)
    fico = st.number_input("FICO Score (low)", min_value=300, max_value=850, value=700, step=5)
    int_rate = st.number_input("Interest Rate (%)", min_value=1.0, max_value=40.0, value=12.5, step=0.25)

    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    purpose = st.selectbox("Purpose", [
        "debt_consolidation", "credit_card", "home_improvement",
        "other", "major_purchase", "medical", "small_business",
    ])
    term = st.selectbox("Term", ["36 months", "60 months"])
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    emp_length = st.slider("Employment Length (years)", 0, 10, 3)

    # Home Credit fields
    st.subheader("Credit Bureau Fields")
    amt_credit = st.number_input("AMT_CREDIT", value=float(loan_amnt))
    amt_annuity = st.number_input("AMT_ANNUITY (monthly payment)", value=round(loan_amnt / 36, 2))

    submit = st.button("Evaluate Application", type="primary", use_container_width=True)

# ── Main Panel ────────────────────────────────────────────────────────────────
if submit:
    payload = {
        "loan_amnt": loan_amnt,
        "fico_range_low": float(fico),
        "dti": dti,
        "annual_inc": annual_inc,
        "int_rate": int_rate,
        "grade": grade,
        "purpose": purpose,
        "term": term,
        "home_ownership": home_ownership,
        "emp_length_num": float(emp_length),
        "AMT_CREDIT": amt_credit,
        "AMT_INCOME_TOTAL": annual_inc,
        "AMT_ANNUITY": amt_annuity,
        "TransactionAmt": loan_amnt,
    }

    with st.spinner("Running multi-agent evaluation..."):
        try:
            resp = httpx.post(f"{API_BASE}/evaluate-loan", json=payload, timeout=120.0)
            resp.raise_for_status()
            data = resp.json()
        except httpx.ReadTimeout:
            st.error("Request timed out after 120s. The server may be overloaded — try again.")
            st.stop()
        except httpx.ConnectError:
            st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── Decision banner ───────────────────────────────────────────────────────
    decision = data.get("decision", "error").upper()
    color_map = {
        "APPROVE": "🟢", "APPROVE_WITH_RATE": "🟢", "APPROVE_STANDARD": "🟢",
        "DECLINE": "🔴", "HUMAN_REVIEW": "🟡",
    }
    icon = color_map.get(decision, "⚪")
    st.markdown(f"## {icon} Decision: **{decision}**")
    st.info(data.get("decision_reason", ""))

    # ── Risk score metrics ────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        dp = data.get("default_probability")
        st.metric("Default Probability", f"{dp:.2%}" if dp is not None else "N/A", delta=data.get("risk_band"))
    with col2:
        fs = data.get("fraud_score")
        st.metric("Fraud Score", f"{fs:.4f}" if fs is not None else "N/A", delta=data.get("risk_level"))
    with col3:
        us = data.get("uplift_score")
        st.metric("Uplift Score (ITE)", f"{us:.4f}" if us is not None else "N/A", delta=data.get("segment"))

    # ── SHAP bar chart ────────────────────────────────────────────────────────
    shap_top3 = data.get("shap_top3")
    if shap_top3:
        st.subheader("Top 3 SHAP Factors (Credit Risk)")
        shap_df = pd.DataFrame(shap_top3)
        shap_df["color"] = shap_df["shap_value"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")
        st.bar_chart(shap_df.set_index("feature")["shap_value"])

    # ── Audit narrative ───────────────────────────────────────────────────────
    narrative = data.get("shap_narrative")
    if narrative:
        st.subheader("Audit Narrative")
        st.write(narrative)

    # ── PDF download ──────────────────────────────────────────────────────────
    pdf_url = data.get("audit_pdf_url")
    if pdf_url is not None:
        try:
            pdf_resp = httpx.get(f"{API_BASE}{pdf_url}", timeout=10.0)
            st.download_button(
                label="📄 Download Audit PDF",
                data=pdf_resp.content,
                file_name="loan_audit_report.pdf",
                mime="application/pdf",
            )
        except Exception:
            st.warning("PDF download unavailable.")

    # ── Errors ────────────────────────────────────────────────────────────────
    errors = data.get("errors")
    if errors:
        with st.expander("Agent Errors"):
            for err in errors:
                st.error(err)

else:
    st.info("Fill in the application form on the left and click **Evaluate Application**.")

    # ── Health check ──────────────────────────────────────────────────────────
    try:
        health = httpx.get(f"{API_BASE}/health", timeout=3.0).json()
        if health.get("models_loaded"):
            st.success("API is online and models are loaded.")
        else:
            st.warning("API is online but models are not loaded yet.")
    except Exception:
        st.warning("API server is not running. Start it with: `uvicorn api.main:app --port 8000`")
