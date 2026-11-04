import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Page Configuration & Anti-AI Editorial Aesthetics
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Attrition Risk Evaluation — IBM HR Analytics",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=Space+Mono:wght@400;700&family=Inter:wght@400;500;600&display=swap');

    /* Global Typography & Palette */
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

    /* Section Separators */
    .section-label {
        font-family: 'Space Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #44403C;
        border-bottom: 1px solid #1C1917;
        padding-bottom: 0.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* Form Inputs */
    label p {
        font-family: 'Space Mono', monospace !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.04em !important;
        text-transform: uppercase !important;
        color: #292524 !important;
        font-weight: 700 !important;
    }

    /* Primary Action Button */
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

    /* Result Docket */
    .docket-container {
        border: 1px solid #1C1917;
        background-color: #FFFFFF;
        padding: 1.5rem 1.8rem;
        margin-top: 2rem;
        border-radius: 0px;
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
    .verdict-box-positive {
        border-left: 4px solid #991B1B;
        background-color: #FEF2F2;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .verdict-box-negative {
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
def load_model_and_features():
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    return model, feature_cols


model, feature_columns = load_model_and_features()

# -----------------------------------------------------------------------------
# Editorial Masthead
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="masthead-meta">
        Docket: HR-ATTRITION-EVAL &nbsp;|&nbsp; Model: Random Forest Ensemble (N=60, D=16) &nbsp;|&nbsp; Vector: 44 Dimension Specification
    </div>
    <div class="masthead-title">Employee Attrition Analysis</div>
    <div class="masthead-subtitle">Statistical classification derived from trained IBM HR telemetry parameters.</div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------
with st.form("attrition_form"):
    st.markdown('<div class="section-label">01 // Demographic & Personal Identity</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        age = st.number_input("Age", min_value=18, max_value=65, value=32, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
    with c2:
        marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
        distance_from_home = st.number_input("DistanceFromHome", min_value=1, max_value=50, value=7, step=1)
    with c3:
        education = st.selectbox("Education (Level 1–5)", [1, 2, 3, 4, 5], index=2)
        education_field = st.selectbox(
            "EducationField",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        )
    with c4:
        num_companies_worked = st.number_input("NumCompaniesWorked", min_value=0, max_value=15, value=2, step=1)
        total_working_years = st.number_input("TotalWorkingYears", min_value=0, max_value=40, value=9, step=1)

    st.markdown('<div class="section-label">02 // Workplace & Role Specification</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        department = st.selectbox("Department", ["Research & Development", "Sales", "Human Resources"])
        job_role = st.selectbox(
            "JobRole",
            [
                "Research Scientist",
                "Laboratory Technician",
                "Sales Executive",
                "Manufacturing Director",
                "Healthcare Representative",
                "Manager",
                "Sales Representative",
                "Research Director",
                "Human Resources",
            ],
        )
    with c2:
        job_level = st.selectbox("JobLevel (Level 1–5)", [1, 2, 3, 4, 5], index=1)
        business_travel = st.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
    with c3:
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        stock_option_level = st.selectbox("StockOptionLevel (0–3)", [0, 1, 2, 3], index=1)
    with c4:
        training_times_last_year = st.number_input("TrainingTimesLastYear", min_value=0, max_value=10, value=3, step=1)
        percent_salary_hike = st.number_input("PercentSalaryHike (%)", min_value=10, max_value=30, value=14, step=1)

    st.markdown('<div class="section-label">03 // Financial Compensation (USD)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=30000, value=5200, step=100)
    with c2:
        daily_rate = st.number_input("DailyRate", min_value=100, max_value=2000, value=800, step=10)
    with c3:
        hourly_rate = st.number_input("HourlyRate", min_value=20, max_value=150, value=65, step=1)
    with c4:
        monthly_rate = st.number_input("MonthlyRate", min_value=1000, max_value=40000, value=14200, step=100)

    st.markdown('<div class="section-label">04 // Organizational Tenure (Years)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=6, step=1)
    with c2:
        years_in_current_role = st.number_input("YearsInCurrentRole", min_value=0, max_value=25, value=4, step=1)
    with c3:
        years_since_last_promotion = st.number_input("YearsSinceLastPromotion", min_value=0, max_value=20, value=1, step=1)
    with c4:
        years_with_curr_manager = st.number_input("YearsWithCurrManager", min_value=0, max_value=25, value=4, step=1)

    st.markdown('<div class="section-label">05 // Satisfaction & Evaluation Indexes (Scale 1–4)</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        environment_satisfaction = st.selectbox("EnvironmentSatisfaction", [1, 2, 3, 4], index=2)
        job_satisfaction = st.selectbox("JobSatisfaction", [1, 2, 3, 4], index=2)
    with c2:
        job_involvement = st.selectbox("JobInvolvement", [1, 2, 3, 4], index=2)
        relationship_satisfaction = st.selectbox("RelationshipSatisfaction", [1, 2, 3, 4], index=2)
    with c3:
        work_life_balance = st.selectbox("WorkLifeBalance", [1, 2, 3, 4], index=2)
    with c4:
        performance_rating = st.selectbox("PerformanceRating", [3, 4], index=0)

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.form_submit_button("Execute Inference Vector →")

# -----------------------------------------------------------------------------
# Inference Execution & Outcome Docket
# -----------------------------------------------------------------------------
if submit:
    # Initialize zero-vector matching feature_columns.json
    row = {col: 0 for col in feature_columns}

    # Numeric and ordinal assignments
    row["Age"] = age
    row["DailyRate"] = daily_rate
    row["DistanceFromHome"] = distance_from_home
    row["Education"] = education
    row["EnvironmentSatisfaction"] = environment_satisfaction
    row["HourlyRate"] = hourly_rate
    row["JobInvolvement"] = job_involvement
    row["JobLevel"] = job_level
    row["JobSatisfaction"] = job_satisfaction
    row["MonthlyIncome"] = monthly_income
    row["MonthlyRate"] = monthly_rate
    row["NumCompaniesWorked"] = num_companies_worked
    row["PercentSalaryHike"] = percent_salary_hike
    row["PerformanceRating"] = performance_rating
    row["RelationshipSatisfaction"] = relationship_satisfaction
    row["StockOptionLevel"] = stock_option_level
    row["TotalWorkingYears"] = total_working_years
    row["TrainingTimesLastYear"] = training_times_last_year
    row["WorkLifeBalance"] = work_life_balance
    row["YearsAtCompany"] = years_at_company
    row["YearsInCurrentRole"] = years_in_current_role
    row["YearsSinceLastPromotion"] = years_since_last_promotion
    row["YearsWithCurrManager"] = years_with_curr_manager

    # Binary mappings
    row["Gender"] = 1 if gender == "Female" else 0
    row["OverTime"] = 1 if overtime == "Yes" else 0

    # Categorical dummy encodings (drop_first=True baseline parity)
    if business_travel == "Travel_Frequently":
        row["BusinessTravel_Travel_Frequently"] = 1
    elif business_travel == "Travel_Rarely":
        row["BusinessTravel_Travel_Rarely"] = 1

    if department == "Research & Development":
        row["Department_Research & Development"] = 1
    elif department == "Sales":
        row["Department_Sales"] = 1

    ef_col = f"EducationField_{education_field}"
    if ef_col in row:
        row[ef_col] = 1

    jr_col = f"JobRole_{job_role}"
    if jr_col in row:
        row[jr_col] = 1

    if marital_status == "Married":
        row["MaritalStatus_Married"] = 1
    elif marital_status == "Single":
        row["MaritalStatus_Single"] = 1

    # Exact feature frame verification
    input_df = pd.DataFrame([row])[feature_columns]

    # Model inference
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    p_retention = float(probabilities[0] * 100)
    p_attrition = float(probabilities[1] * 100)

    # Output Docket
    if prediction == 1:
        verdict_class = "verdict-box-positive"
        verdict_text = "CLASSIFICATION: ATTRITION (CLASS 1)"
        verdict_desc = "The model classifies this feature profile under the attrition category based on the evaluated 44-parameter vector."
    else:
        verdict_class = "verdict-box-negative"
        verdict_text = "CLASSIFICATION: RETENTION (CLASS 0)"
        verdict_desc = "The model classifies this feature profile under the retention category based on the evaluated 44-parameter vector."

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
                <strong>Decision Threshold:</strong> 0.5000 &nbsp;|&nbsp; <strong>Vector Dimension:</strong> 44 features
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
