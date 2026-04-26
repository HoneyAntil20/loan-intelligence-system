import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.uplift_tool import run_uplift_model

cases = [
    {"loan_amnt": 5000,  "int_rate": 28.0, "grade": "G", "fico_range_low": 560, "dti": 45.0, "annual_inc": 25000, "home_ownership": "RENT", "purpose": "other", "emp_length_num": 0},
    {"loan_amnt": 15000, "int_rate": 12.5, "grade": "A", "fico_range_low": 700, "dti": 18.0, "annual_inc": 65000, "home_ownership": "RENT", "purpose": "debt_consolidation", "emp_length_num": 3},
    {"loan_amnt": 30000, "int_rate": 20.0, "grade": "D", "fico_range_low": 620, "dti": 30.0, "annual_inc": 45000, "home_ownership": "OWN",  "purpose": "credit_card",        "emp_length_num": 5},
]

for i, c in enumerate(cases):
    r = run_uplift_model.invoke({"applicant_features": c})
    print(f"Case {i+1}: uplift={r['uplift_score']}, segment={r['segment']}, p0={r['baseline_repay_prob']}, status={r['status']}")
