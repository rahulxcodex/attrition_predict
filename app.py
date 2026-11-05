import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Configuration & Clean Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="🏢",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Clean, human styling */
    .stApp {
        max-width: 860px;
        margin: 0 auto;
    }
    .main-header {
        margin-bottom: 1.5rem;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        font-size: 1rem;
        color: #4B5563;
        margin: 0;
    }
    .result-card {
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin-top: 1.5rem;
        border: 1px solid #E5E7EB;
        background: #FFFFFF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Model & Feature Assets
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

# Empirical dataset medians/modes for non-selected features
BASELINE_VALUES = {
    col: 0 for col in feature_columns
}
BASELINE_VALUES.update({
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
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>Employee Attrition Predictor</h1>
        <p>Predict turnover risk using the top 10 Random Forest features.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Input Form
# -----------------------------------------------------------------------------
with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        monthly_income = st.number_input("Monthly income ($)", min_value=1000, max_value=30000, value=4919, step=100)
        overtime = st.selectbox("Overtime", ["No", "Yes"])
        age = st.number_input("Age", min_value=18, max_value=65, value=36, step=1)
        daily_rate = st.number_input("Daily rate ($)", min_value=100, max_value=2000, value=802, step=10)
        total_working_years = st.number_input("Total working years", min_value=0, max_value=40, value=10, step=1)

    with col2:
        monthly_rate = st.number_input("Monthly rate ($)", min_value=1000, max_value=40000, value=14236, step=100)
        distance_from_home = st.number_input("Distance from home (km)", min_value=1, max_value=50, value=7, step=1)
        hourly_rate = st.number_input("Hourly rate ($)", min_value=20, max_value=150, value=66, step=1)
        years_at_company = st.number_input("Years at company", min_value=0, max_value=40, value=5, step=1)
        num_companies_worked = st.number_input("Companies worked at", min_value=0, max_value=15, value=2, step=1)

    submitted = st.form_submit_button("Predict Attrition", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Prediction Logic & Output
# -----------------------------------------------------------------------------
if submitted:
    row = dict(BASELINE_VALUES)
    row["MonthlyIncome"] = monthly_income
    row["OverTime"] = 1 if overtime == "Yes" else 0
    row["Age"] = age
    row["DailyRate"] = daily_rate
    row["TotalWorkingYears"] = total_working_years
    row["MonthlyRate"] = monthly_rate
    row["DistanceFromHome"] = distance_from_home
    row["HourlyRate"] = hourly_rate
    row["YearsAtCompany"] = years_at_company
    row["NumCompaniesWorked"] = num_companies_worked

    input_df = pd.DataFrame([row])[feature_columns]

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]
    p_stay = float(probabilities[0] * 100)
    p_leave = float(probabilities[1] * 100)

    st.markdown("---")

    if prediction == 1:
        st.error(f"**High Attrition Risk** — This employee is predicted likely to leave.")
    else:
        st.success(f"**Low Attrition Risk** — This employee is predicted likely to stay.")

    m1, m2 = st.columns(2)
    with m1:
        st.metric(label="Probability of Leaving", value=f"{p_leave:.1f}%")
    with m2:
        st.metric(label="Probability of Staying", value=f"{p_stay:.1f}%")

    st.caption("Prediction generated by Random Forest (60 estimators) using the 10 strongest features; baseline features held at dataset medians.")
