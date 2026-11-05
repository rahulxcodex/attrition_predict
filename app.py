import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Page Configuration & Anti-AI Editorial Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Attrition Risk Prediction — Key Predictor Model",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #141413;
        background-color: #FAF9F6;
    }

    /* Masthead */
    .masthead-meta {
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #78716C;
        border-bottom: 1px solid #E7E5E4;
        padding-bottom: 0.5rem;
        margin-bottom: 1.2rem;
    }
    .masthead-title {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 2.8rem;
        font-weight: 400;
        letter-spacing: -0.02em;
        line-height: 1.1;
        color: #1C1917;
        margin-bottom: 0.4rem;
    }
    .masthead-subtitle {
        font-family: 'Newsreader', Georgia, serif;
        font-style: italic;
        font-size: 1.15rem;
        color: #57534E;
        margin-bottom: 2rem;
    }

    /* Section Divider */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #44403C;
        border-bottom: 1px solid #1C1917;
        padding-bottom: 0.3rem;
        margin-top: 1.2rem;
        margin-bottom: 1.2rem;
    }

    /* Form Labels */
    label p {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: #292524 !important;
        font-weight: 700 !important;
    }

    /* Button */
    div.stButton > button:first-child {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        background-color: #1C1917 !important;
        color: #FAF9F6 !important;
        border: 1px solid #1C1917 !important;
        border-radius: 0px !important;
        padding: 0.7rem 1.6rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #292524 !important;
        color: #FFFFFF !important;
    }
    div.stButton > button:first-child:active {
        transform: translateY(1px) !important;
    }

    /* Output Docket */
    .docket-container {
        border: 1px solid #1C1917;
        background-color: #FFFFFF;
        padding: 1.5rem 1.8rem;
        margin-top: 2rem;
    }
    .docket-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #78716C;
        border-bottom: 1px solid #E7E5E4;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
    .verdict-positive {
        border-left: 4px solid #991B1B;
        background-color: #FEF2F2;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .verdict-negative {
        border-left: 4px solid #166534;
        background-color: #F0FDF4;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .verdict-title {
        font-family: 'Space Mono', monospace;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        margin-bottom: 0.2rem;
    }
    .verdict-body {
        font-family: 'Newsreader', Georgia, serif;
        font-size: 1.05rem;
        color: #292524;
    }
    .metric-data {
        font-family: 'Space Mono', monospace;
        font-size: 0.88rem;
        color: #1C1917;
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Asset Loader
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.json"


@st.cache_resource
def load_assets():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, feature_cols


model, feature_columns = load_assets()

# -----------------------------------------------------------------------------
# Dataset Empirical Baseline (Medians / Modes for unasked non-top features)
# -----------------------------------------------------------------------------
BASELINE_VECTOR = {col: 0 for col in feature_columns}
BASELINE_VECTOR.update({
    "Age": 36,
    "DailyRate": 802,
    "DistanceFromHome": 7,
    "Education": 3,
    "EnvironmentSatisfaction": 3,
    "Gender": 0,
    "HourlyRate": 66,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "MonthlyIncome": 4919,
    "MonthlyRate": 14236,
    "NumCompaniesWorked": 2,
    "OverTime": 0,
    "PercentSalaryHike": 14,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsInCurrentRole": 3,
    "YearsSinceLastPromotion": 1,
    "YearsWithCurrManager": 3,
    "BusinessTravel_Travel_Rarely": 1,
    "Department_Research & Development": 1,
    "EducationField_Life Sciences": 1,
    "JobRole_Research Scientist": 1,
    "MaritalStatus_Married": 1,
})

# -----------------------------------------------------------------------------
# Masthead
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead-meta">
        Docket: HR-ATTRITION-KEY-PREDICTORS &nbsp;|&nbsp; Model: Random Forest Ensemble (N=60, D=16) &nbsp;|&nbsp; Top Gini Features
    </div>
    <div class="masthead-title">Attrition Prediction via Key Variables</div>
    <div class="masthead-subtitle">Model inference constrained to the top primary importance features from random forest training.</div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Input Form: Top 10 Important Variables
# -----------------------------------------------------------------------------
with st.form("key_features_form"):
    st.markdown('<div class="section-label">Primary Predictor Variables (Top 10 Feature Importances)</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        monthly_income = st.number_input(
            "MonthlyIncome (USD) — Rank 1 [Importance: 8.17%]",
            min_value=1000,
            max_value=30000,
            value=4919,
            step=100,
        )
        overtime = st.selectbox(
            "OverTime — Rank 2 [Importance: 7.73%]",
            ["No", "Yes"],
            index=0,
        )
        age = st.number_input(
            "Age (Years) — Rank 3 [Importance: 5.69%]",
            min_value=18,
            max_value=65,
            value=36,
            step=1,
        )
        daily_rate = st.number_input(
            "DailyRate (USD) — Rank 4 [Importance: 5.25%]",
            min_value=100,
            max_value=2000,
            value=802,
            step=10,
        )
        total_working_years = st.number_input(
            "TotalWorkingYears — Rank 5 [Importance: 4.87%]",
            min_value=0,
            max_value=40,
            value=10,
            step=1,
        )

    with c2:
        monthly_rate = st.number_input(
            "MonthlyRate (USD) — Rank 6 [Importance: 4.60%]",
            min_value=1000,
            max_value=40000,
            value=14236,
            step=100,
        )
        distance_from_home = st.number_input(
            "DistanceFromHome (km) — Rank 7 [Importance: 4.51%]",
            min_value=1,
            max_value=50,
            value=7,
            step=1,
        )
        hourly_rate = st.number_input(
            "HourlyRate (USD) — Rank 8 [Importance: 4.25%]",
            min_value=20,
            max_value=150,
            value=66,
            step=1,
        )
        years_at_company = st.number_input(
            "YearsAtCompany — Rank 9 [Importance: 4.11%]",
            min_value=0,
            max_value=40,
            value=5,
            step=1,
        )
        num_companies_worked = st.number_input(
            "NumCompaniesWorked — Rank 10 [Importance: 3.45%]",
            min_value=0,
            max_value=15,
            value=2,
            step=1,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("Run Prediction Inference →")

# -----------------------------------------------------------------------------
# Inference Execution & Outcome Docket
# -----------------------------------------------------------------------------
if submit:
    # Populate feature vector starting from empirical dataset baseline
    feature_row = dict(BASELINE_VECTOR)

    # Inject the 10 important variables from user input
    feature_row["MonthlyIncome"] = monthly_income
    feature_row["OverTime"] = 1 if overtime == "Yes" else 0
    feature_row["Age"] = age
    feature_row["DailyRate"] = daily_rate
    feature_row["TotalWorkingYears"] = total_working_years
    feature_row["MonthlyRate"] = monthly_rate
    feature_row["DistanceFromHome"] = distance_from_home
    feature_row["HourlyRate"] = hourly_rate
    feature_row["YearsAtCompany"] = years_at_company
    feature_row["NumCompaniesWorked"] = num_companies_worked

    # Construct exact 44-column DataFrame required by rf_model.pkl
    input_df = pd.DataFrame([feature_row])[feature_columns]

    # Predict
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    p_retention = float(probabilities[0] * 100)
    p_attrition = float(probabilities[1] * 100)

    # Result presentation
    if prediction == 1:
        verdict_class = "verdict-positive"
        verdict_text = "CLASSIFICATION: ATTRITION PREDICTED (CLASS 1)"
        verdict_desc = "The model classifies this employee profile under the attrition category based on the evaluated key predictor variables."
    else:
        verdict_class = "verdict-negative"
        verdict_text = "CLASSIFICATION: RETENTION PREDICTED (CLASS 0)"
        verdict_desc = "The model classifies this employee profile under the retention category based on the evaluated key predictor variables."

    st.markdown(
        f"""
        <div class="docket-container">
            <div class="docket-header">Analytical Evaluation Result // Feature Alignment: Complete</div>
            <div class="{verdict_class}">
                <div class="verdict-title">{verdict_text}</div>
                <div class="verdict-body">{verdict_desc}</div>
            </div>
            <div class="metric-data">
                <strong>Probability of Attrition (Class 1):</strong> {p_attrition:.2f}%<br>
                <strong>Probability of Retention (Class 0):</strong> {p_retention:.2f}%<br>
                <strong>Decision Boundary:</strong> 0.5000 &nbsp;|&nbsp; <strong>Active Predictors:</strong> Top 10 Primary Gini Features
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
