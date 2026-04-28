# Multi-Agent Loan Intelligence System

A production-grade loan evaluation pipeline powered by a multi-agent AI architecture. The system combines classical ML models with a LangGraph orchestration layer to deliver explainable, auditable loan decisions in real time.

---

## What It Does

Evaluates a loan application through three specialist AI agents running in sequence, then routes the combined output through a supervisor that applies business rules to produce a final decision. Every decision generates a PDF audit report with SHAP explanations and an AI-written narrative.

**Decision outcomes:** `approve_with_rate` · `approve_standard` · `decline` · `human_review`

---

## Architecture

```
Loan Application (HTTP POST)
        │
        ▼
┌─────────────────────────────────────────┐
│           LangGraph Pipeline            │
│                                         │
│  credit_risk ──┐                        │
│  fraud         ├──▶ supervisor ──▶ explainability ──▶ PDF + Narrative
│  uplift    ────┘                        │
└─────────────────────────────────────────┘
        │
        ▼
  JSON Response (FastAPI)
        │
        ▼
  Streamlit UI
```

---

## Agents

| Agent | Model | Dataset | Output |
|---|---|---|---|
| Credit Risk | LightGBM + SHAP | Home Credit Default | Default probability, risk band, top-3 SHAP factors |
| Fraud Detection | LightGBM | IEEE-CIS Fraud Detection | Fraud score, risk level, anomaly flags |
| Uplift | T-Learner (LightGBM) | Lending Club | ITE score, segment (Persuadable / Sure Thing / Lost Cause / Do Not Disturb) |
| Supervisor | Rule-based | — | Final decision + reason |
| Explainability | Gemini 2.0 Flash + ReportLab | — | Plain-English narrative + PDF audit report |

---

## Tech Stack

### ML & Explainability
- **LightGBM** — gradient boosted trees for credit risk and fraud classification
- **scikit-learn** — `SimpleImputer`, `LabelEncoder`, `OrdinalEncoder` for preprocessing
- **SHAP** (`TreeExplainer`) — feature attribution for credit risk model
- **T-Learner** (custom implementation) — causal uplift model estimating individual treatment effect (ITE)
- **joblib** — model artifact serialization/deserialization

### Agent Orchestration
- **LangGraph** — stateful multi-agent graph with conditional routing edges
- **LangChain Core** — `@tool` decorators, `HumanMessage`, tool invocation wrappers
- **LangSmith** — tracing and observability for agent runs

### LLM
- **Google Gemini 2.0 Flash** (`langchain-google-genai`) — generates plain-English audit narratives from SHAP factors and decision context

### API Layer
- **FastAPI** — REST API with lifespan startup (models loaded once, never per-request)
- **Pydantic v2** — request/response schema validation
- **Uvicorn** — ASGI server

### Frontend
- **Streamlit** — interactive loan application form, decision display, SHAP visualization, PDF download

### PDF Generation
- **ReportLab** — programmatic PDF audit reports with decision header, risk score grid, SHAP factor table, and LLM narrative

### Infrastructure & Utilities
- **python-dotenv** — environment variable management
- **httpx** — async HTTP client (Streamlit → FastAPI)
- **kaggle** — dataset download automation

---

## Project Structure

```
loan-intelligence-system/
├── agents/
│   ├── graph.py          # LangGraph StateGraph definition
│   ├── state.py          # Shared LoanState TypedDict
│   ├── credit_risk.py    # Credit risk agent node
│   ├── fraud.py          # Fraud detection agent node
│   ├── uplift.py         # Uplift agent node
│   ├── supervisor.py     # Business rule routing
│   ├── explainability.py # Gemini narrative + PDF generation
│   └── tracing.py        # LangSmith tracing setup
├── tools/
│   ├── credit_risk_tool.py  # LangChain @tool wrapping LightGBM credit model
│   ├── fraud_tool.py        # LangChain @tool wrapping LightGBM fraud model
│   └── uplift_tool.py       # LangChain @tool wrapping T-Learner
├── models/
│   ├── train_credit_risk.py # Home Credit model training
│   ├── train_fraud.py       # IEEE-CIS fraud model training
│   ├── train_uplift.py      # Lending Club uplift model training
│   ├── tlearner.py          # Custom T-Learner implementation
│   ├── artifacts/           # Pickled models, imputers, encoders, SHAP explainer
│   └── metrics/             # JSON files with thresholds and evaluation metrics
├── api/
│   ├── main.py           # FastAPI app with lifespan model loading
│   └── schemas.py        # Pydantic request/response models
├── data_prep/
│   ├── home_credit_prep.py   # Home Credit feature engineering
│   ├── ieee_fraud_prep.py    # IEEE-CIS feature engineering
│   └── lending_club_prep.py  # Lending Club feature engineering
├── reports/
│   └── audit_pdfs/       # Generated PDF audit reports
├── tests/
│   ├── smoke_test.py
│   └── test_uplift.py
├── streamlit_app.py      # Frontend UI
├── requirements.txt
└── .env                  # API keys (not committed)
```

---

## Running the Project

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set environment variables** — create a `.env` file:
```
GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=loan-intelligence-system
```

**3. Start the API** (from inside `loan-intelligence-system/`):
```bash
uvicorn api.main:app --port 8000
```

**4. Start the frontend** (separate terminal):
```bash
streamlit run streamlit_app.py
```

- API: http://localhost:8000
- UI: http://localhost:8501
- API docs: http://localhost:8000/docs

---

## Supervisor Decision Rules

Applied in priority order:

1. Any agent error → `human_review`
2. Fraud score > 0.85 → `decline`
3. Default probability > 0.70 → `decline`
4. Uplift CI width > 0.15 → `human_review` (model uncertainty too high)
5. Segment = `Persuadable` → `approve_with_rate`
6. Segment = `Sure Thing` → `approve_standard`
7. Segment = `Lost Cause` / `Do Not Disturb` → `decline`

---

## Key Design Decisions

- **Models loaded at startup, never per-request** — eliminates 2–5s cold-start latency on every call
- **Lazy LLM initialization** — Gemini client instantiated on first call after `.env` is loaded, avoiding import-time failures
- **Graceful LLM fallback** — if Gemini is rate-limited or unavailable, a clean template-based narrative is generated from the model outputs instead of surfacing raw API errors
- **SHAP for auditability** — every credit decision includes the top-3 features driving the prediction, required for regulatory explainability
- **T-Learner uplift** — goes beyond risk scoring to estimate the causal effect of a rate offer on repayment probability, enabling smarter approval strategies
