import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Suppress sklearn unpickling warnings
warnings.filterwarnings("ignore", category=UserWarning)

# -----------------------------------------------------------------------------
# Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid #E2E8F0;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #FEE2E2 0%, #FECACA 100%);
        border: 1px solid #F87171;
        border-radius: 10px;
        padding: 1.5rem;
        color: #991B1B;
    }
    .risk-low {
        background: linear-gradient(135deg, #DCFCE7 0%, #BBF7D0 100%);
        border: 1px solid #4ADE80;
        border-radius: 10px;
        padding: 1.5rem;
        color: #166534;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Asset Loading
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "rf_model.pkl"
FEATURE_PATH = BASE_DIR / "feature_columns.json"


@st.cache_resource
def load_assets():
    """Load the trained Random Forest model and the exact feature column list."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Feature columns file not found at: {FEATURE_PATH}")

    model = joblib.load(MODEL_PATH)
    with open(FEATURE_PATH, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)

    return model, feature_columns


try:
    rf_model, feature_columns = load_assets()
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# Helper: Preprocessing and Vector Construction
# -----------------------------------------------------------------------------
def transform_to_features(raw_input: dict, target_columns: list) -> pd.DataFrame:
    """Transform user input dictionary into the 44-feature DataFrame matching model training."""
    row = {col: 0 for col in target_columns}

    # Direct numeric and ordinal fields
    numeric_keys = [
        "Age",
        "DailyRate",
        "DistanceFromHome",
        "Education",
        "EnvironmentSatisfaction",
        "HourlyRate",
        "JobInvolvement",
        "JobLevel",
        "JobSatisfaction",
        "MonthlyIncome",
        "MonthlyRate",
        "NumCompaniesWorked",
        "PercentSalaryHike",
        "PerformanceRating",
        "RelationshipSatisfaction",
        "StockOptionLevel",
        "TotalWorkingYears",
        "TrainingTimesLastYear",
        "WorkLifeBalance",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsSinceLastPromotion",
        "YearsWithCurrManager",
    ]
    for key in numeric_keys:
        row[key] = raw_input.get(key, 0)

    # Binary encoded columns (Female=1, Male=0; Yes=1, No=0)
    row["Gender"] = 1 if raw_input.get("Gender") == "Female" else 0
    row["OverTime"] = 1 if raw_input.get("OverTime") == "Yes" else 0

    # Categorical one-hot encoded columns (with drop_first=True reference categories)
    # BusinessTravel (Base: Non-Travel)
    bt = raw_input.get("BusinessTravel")
    if bt in ["Travel_Frequently", "Travel_Rarely"]:
        col_name = f"BusinessTravel_{bt}"
        if col_name in row:
            row[col_name] = 1

    # Department (Base: Human Resources)
    dept = raw_input.get("Department")
    if dept in ["Research & Development", "Sales"]:
        col_name = f"Department_{dept}"
        if col_name in row:
            row[col_name] = 1

    # EducationField (Base: Human Resources)
    ef = raw_input.get("EducationField")
    if ef in ["Life Sciences", "Marketing", "Medical", "Other", "Technical Degree"]:
        col_name = f"EducationField_{ef}"
        if col_name in row:
            row[col_name] = 1

    # JobRole (Base: Healthcare Representative)
    jr = raw_input.get("JobRole")
    valid_roles = [
        "Human Resources",
        "Laboratory Technician",
        "Manager",
        "Manufacturing Director",
        "Research Director",
        "Research Scientist",
        "Sales Executive",
        "Sales Representative",
    ]
    if jr in valid_roles:
        col_name = f"JobRole_{jr}"
        if col_name in row:
            row[col_name] = 1

    # MaritalStatus (Base: Divorced)
    ms = raw_input.get("MaritalStatus")
    if ms in ["Married", "Single"]:
        col_name = f"MaritalStatus_{ms}"
        if col_name in row:
            row[col_name] = 1

    return pd.DataFrame([row])[target_columns]


# -----------------------------------------------------------------------------
# Sidebar Navigation & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/conference-call.png",
        width=70,
    )
    st.title("HR Analytics")
    st.caption("Random Forest Ensemble Classifier")

    app_mode = st.radio(
        "Navigation",
        [
            "🔍 Single Employee Evaluation",
            "📁 Batch CSV Prediction",
            "📊 Model Insights & Feature Importance",
        ],
    )

    st.markdown("---")
    st.markdown("### Model Architecture")
    st.info(
        f"""
        - **Algorithm**: Random Forest Classifier
        - **Trees (Estimators)**: 60
        - **Max Depth**: 16
        - **Features Expected**: {len(feature_columns)}
        - **Target**: Attrition (0: Stay, 1: Leave)
        """
    )


# -----------------------------------------------------------------------------
# MODE 1: Single Employee Evaluation
# -----------------------------------------------------------------------------
if app_mode == "🔍 Single Employee Evaluation":
    st.markdown('<div class="main-title">Employee Attrition Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Evaluate attrition probability for individual staff members using trained Random Forest ensemble modeling.</div>',
        unsafe_allow_html=True,
    )

    # Preset profiles for rapid testing
    col_preset, _ = st.columns([2, 3])
    with col_preset:
        preset_choice = st.selectbox(
            "Quick Profile Presets (Auto-fills inputs)",
            ["Custom", "Standard Experienced Employee", "High Risk Profile (Overtime, Low Pay)", "High Retention Senior Leader"],
        )

    # Preset default definitions
    defaults = {
        "Age": 35,
        "Gender": "Male",
        "MaritalStatus": "Married",
        "DistanceFromHome": 6,
        "Education": 3,
        "EducationField": "Life Sciences",
        "Department": "Research & Development",
        "JobRole": "Research Scientist",
        "JobLevel": 2,
        "BusinessTravel": "Travel_Rarely",
        "OverTime": "No",
        "MonthlyIncome": 5000,
        "DailyRate": 800,
        "HourlyRate": 65,
        "MonthlyRate": 14000,
        "PercentSalaryHike": 14,
        "StockOptionLevel": 1,
        "EnvironmentSatisfaction": 3,
        "JobSatisfaction": 3,
        "RelationshipSatisfaction": 3,
        "JobInvolvement": 3,
        "WorkLifeBalance": 3,
        "PerformanceRating": 3,
        "TotalWorkingYears": 10,
        "YearsAtCompany": 6,
        "YearsInCurrentRole": 4,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 4,
        "NumCompaniesWorked": 2,
        "TrainingTimesLastYear": 3,
    }

    if preset_choice == "High Risk Profile (Overtime, Low Pay)":
        defaults.update({
            "Age": 22,
            "Gender": "Female",
            "MaritalStatus": "Single",
            "DistanceFromHome": 26,
            "Education": 1,
            "EducationField": "Technical Degree",
            "Department": "Sales",
            "JobRole": "Sales Representative",
            "JobLevel": 1,
            "BusinessTravel": "Travel_Frequently",
            "OverTime": "Yes",
            "MonthlyIncome": 1800,
            "DailyRate": 300,
            "HourlyRate": 35,
            "MonthlyRate": 4000,
            "PercentSalaryHike": 11,
            "StockOptionLevel": 0,
            "EnvironmentSatisfaction": 1,
            "JobSatisfaction": 1,
            "RelationshipSatisfaction": 1,
            "JobInvolvement": 1,
            "WorkLifeBalance": 1,
            "PerformanceRating": 3,
            "TotalWorkingYears": 2,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
            "NumCompaniesWorked": 3,
            "TrainingTimesLastYear": 1,
        })
    elif preset_choice == "High Retention Senior Leader":
        defaults.update({
            "Age": 48,
            "Gender": "Male",
            "MaritalStatus": "Married",
            "DistanceFromHome": 2,
            "Education": 4,
            "EducationField": "Medical",
            "Department": "Research & Development",
            "JobRole": "Manager",
            "JobLevel": 4,
            "BusinessTravel": "Non-Travel",
            "OverTime": "No",
            "MonthlyIncome": 16000,
            "DailyRate": 1200,
            "HourlyRate": 85,
            "MonthlyRate": 22000,
            "PercentSalaryHike": 18,
            "StockOptionLevel": 2,
            "EnvironmentSatisfaction": 4,
            "JobSatisfaction": 4,
            "RelationshipSatisfaction": 4,
            "JobInvolvement": 4,
            "WorkLifeBalance": 3,
            "PerformanceRating": 3,
            "TotalWorkingYears": 24,
            "YearsAtCompany": 14,
            "YearsInCurrentRole": 10,
            "YearsSinceLastPromotion": 2,
            "YearsWithCurrManager": 8,
            "NumCompaniesWorked": 2,
            "TrainingTimesLastYear": 4,
        })

    # Interactive Input Form with 4 Logical Tabs
    with st.form("employee_form"):
        tab1, tab2, tab3, tab4 = st.tabs([
            "👤 Demographics & Education",
            "💼 Job & Workplace",
            "💰 Compensation & Performance",
            "⏳ Tenure & Experience",
        ])

        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.slider("Age", 18, 65, int(defaults["Age"]))
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if defaults["Gender"] == "Male" else 1)
            with col2:
                marital_status = st.selectbox(
                    "Marital Status",
                    ["Single", "Married", "Divorced"],
                    index=["Single", "Married", "Divorced"].index(defaults["MaritalStatus"]),
                )
                distance_from_home = st.slider("Distance From Home (km)", 1, 30, int(defaults["DistanceFromHome"]))
            with col3:
                education_labels = {
                    1: "1 - Below College",
                    2: "2 - College",
                    3: "3 - Bachelor's Degree",
                    4: "4 - Master's Degree",
                    5: "5 - Doctor",
                }
                education = st.selectbox(
                    "Education Level",
                    options=[1, 2, 3, 4, 5],
                    format_func=lambda x: education_labels[x],
                    index=defaults["Education"] - 1,
                )
                education_fields = ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"]
                education_field = st.selectbox(
                    "Education Field",
                    education_fields,
                    index=education_fields.index(defaults["EducationField"]),
                )

        with tab2:
            col1, col2, col3 = st.columns(3)
            with col1:
                departments = ["Research & Development", "Sales", "Human Resources"]
                department = st.selectbox(
                    "Department",
                    departments,
                    index=departments.index(defaults["Department"]),
                )
                roles = [
                    "Research Scientist",
                    "Laboratory Technician",
                    "Sales Executive",
                    "Manufacturing Director",
                    "Healthcare Representative",
                    "Manager",
                    "Sales Representative",
                    "Research Director",
                    "Human Resources",
                ]
                job_role = st.selectbox(
                    "Job Role",
                    roles,
                    index=roles.index(defaults["JobRole"]),
                )
            with col2:
                job_level = st.slider("Job Level (1 to 5)", 1, 5, int(defaults["JobLevel"]))
                travel_options = ["Non-Travel", "Travel_Rarely", "Travel_Frequently"]
                business_travel = st.selectbox(
                    "Business Travel Frequency",
                    travel_options,
                    index=travel_options.index(defaults["BusinessTravel"]),
                )
            with col3:
                overtime = st.radio(
                    "OverTime Required?",
                    ["No", "Yes"],
                    index=0 if defaults["OverTime"] == "No" else 1,
                    horizontal=True,
                )
                stock_option_level = st.selectbox(
                    "Stock Option Level (0 to 3)",
                    [0, 1, 2, 3],
                    index=int(defaults["StockOptionLevel"]),
                )

        with tab3:
            col1, col2, col3 = st.columns(3)
            with col1:
                monthly_income = st.number_input(
                    "Monthly Income (USD)",
                    min_value=1000,
                    max_value=25000,
                    value=int(defaults["MonthlyIncome"]),
                    step=250,
                )
                daily_rate = st.slider("Daily Rate (USD)", 100, 1500, int(defaults["DailyRate"]))
                hourly_rate = st.slider("Hourly Rate (USD)", 30, 100, int(defaults["HourlyRate"]))
                monthly_rate = st.number_input(
                    "Monthly Rate (USD)",
                    min_value=2000,
                    max_value=30000,
                    value=int(defaults["MonthlyRate"]),
                    step=500,
                )
            with col2:
                percent_salary_hike = st.slider(
                    "Percent Salary Hike (%)", 10, 25, int(defaults["PercentSalaryHike"])
                )
                perf_labels = {3: "3 - Excellent", 4: "4 - Outstanding"}
                performance_rating = st.selectbox(
                    "Performance Rating",
                    options=[3, 4],
                    format_func=lambda x: perf_labels[x],
                    index=0 if defaults["PerformanceRating"] == 3 else 1,
                )
                satisfaction_labels = {1: "1 - Low", 2: "2 - Medium", 3: "3 - High", 4: "4 - Very High"}
                job_satisfaction = st.selectbox(
                    "Job Satisfaction",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: satisfaction_labels[x],
                    index=defaults["JobSatisfaction"] - 1,
                )
            with col3:
                environment_satisfaction = st.selectbox(
                    "Environment Satisfaction",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: satisfaction_labels[x],
                    index=defaults["EnvironmentSatisfaction"] - 1,
                )
                relationship_satisfaction = st.selectbox(
                    "Relationship Satisfaction",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: satisfaction_labels[x],
                    index=defaults["RelationshipSatisfaction"] - 1,
                )
                job_involvement = st.selectbox(
                    "Job Involvement",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: satisfaction_labels[x],
                    index=defaults["JobInvolvement"] - 1,
                )
                work_life_balance_labels = {1: "1 - Bad", 2: "2 - Good", 3: "3 - Better", 4: "4 - Best"}
                work_life_balance = st.selectbox(
                    "Work-Life Balance",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: work_life_balance_labels[x],
                    index=defaults["WorkLifeBalance"] - 1,
                )

        with tab4:
            col1, col2, col3 = st.columns(3)
            with col1:
                total_working_years = st.slider("Total Working Years", 0, 40, int(defaults["TotalWorkingYears"]))
                years_at_company = st.slider("Years at Company", 0, 40, int(defaults["YearsAtCompany"]))
                years_in_current_role = st.slider("Years in Current Role", 0, 20, int(defaults["YearsInCurrentRole"]))
            with col2:
                years_since_promotion = st.slider("Years Since Last Promotion", 0, 15, int(defaults["YearsSinceLastPromotion"]))
                years_with_manager = st.slider("Years With Current Manager", 0, 20, int(defaults["YearsWithCurrManager"]))
            with col3:
                num_companies_worked = st.slider("Num Companies Worked", 0, 10, int(defaults["NumCompaniesWorked"]))
                training_times_last_year = st.slider("Training Times Last Year", 0, 6, int(defaults["TrainingTimesLastYear"]))

        submit_btn = st.form_submit_button("🔮 Predict Attrition Risk", use_container_width=True, type="primary")

    if submit_btn:
        input_data = {
            "Age": age,
            "DailyRate": daily_rate,
            "DistanceFromHome": distance_from_home,
            "Education": education,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime": overtime,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "RelationshipSatisfaction": relationship_satisfaction,
            "StockOptionLevel": stock_option_level,
            "TotalWorkingYears": total_working_years,
            "TrainingTimesLastYear": training_times_last_year,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_promotion,
            "YearsWithCurrManager": years_with_manager,
            "BusinessTravel": business_travel,
            "Department": department,
            "EducationField": education_field,
            "JobRole": job_role,
            "MaritalStatus": marital_status,
        }

        # Build feature vector
        X_eval = transform_to_features(input_data, feature_columns)

        # Run Prediction
        prediction = rf_model.predict(X_eval)[0]
        probabilities = rf_model.predict_proba(X_eval)[0]
        prob_stay = probabilities[0] * 100
        prob_leave = probabilities[1] * 100

        st.markdown("### Prediction Assessment")

        if prediction == 1:
            st.markdown(
                f"""
                <div class="risk-high">
                    <h3 style="margin-top:0;">⚠️ High Attrition Risk Detected</h3>
                    <p style="font-size: 1.15rem; margin-bottom: 0.5rem;">
                        This employee is predicted as <strong>Likely to Leave</strong>.
                    </p>
                    <p style="font-size: 1rem; margin-bottom: 0;">
                        Attrition Probability: <strong>{prob_leave:.1f}%</strong> | Retention Probability: <strong>{prob_stay:.1f}%</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="risk-low">
                    <h3 style="margin-top:0;">✅ Low Attrition Risk (High Retention Probability)</h3>
                    <p style="font-size: 1.15rem; margin-bottom: 0.5rem;">
                        This employee is predicted as <strong>Likely to Stay</strong>.
                    </p>
                    <p style="font-size: 1rem; margin-bottom: 0;">
                        Retention Probability: <strong>{prob_stay:.1f}%</strong> | Attrition Probability: <strong>{prob_leave:.1f}%</strong>
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("Predicted Attrition Status", "Attrition (1)" if prediction == 1 else "No Attrition (0)")
        with col_m2:
            st.metric("Risk Score", f"{prob_leave:.1f}%")
        with col_m3:
            st.metric("Retention Likelihood", f"{prob_stay:.1f}%")

        st.progress(float(prob_leave / 100), text=f"Attrition Risk Spectrum: {prob_leave:.1f}%")

        # Risk Factor & HR Strategy Analysis
        st.markdown("---")
        st.subheader("💡 Key Risk Factors & Retention Strategy")
        risk_flags = []
        if overtime == "Yes":
            risk_flags.append(("OverTime Active", "Working overtime is one of the strongest drivers of employee burnout and attrition."))
        if monthly_income < 3000:
            risk_flags.append(("Low Compensation Tier", f"Monthly income of {monthly_income} USD is in the bottom percentile."))
        if work_life_balance <= 2:
            risk_flags.append(("Low Work-Life Balance", "Score of <= 2 indicates heightened fatigue and job dissatisfaction."))
        if job_satisfaction <= 2:
            risk_flags.append(("Low Job Satisfaction", "Direct predictor of active external job searching."))
        if environment_satisfaction <= 2:
            risk_flags.append(("Unfavorable Work Environment", "Physical or team environment challenges noted."))
        if distance_from_home > 20:
            risk_flags.append(("Long Commute", f"Commuting {distance_from_home} km increases attrition likelihood."))
        if years_since_promotion >= 5:
            risk_flags.append(("Stagnant Promotion Timeline", f"{years_since_promotion} years without promotion increases career frustration."))
        if stock_option_level == 0:
            risk_flags.append(("Zero Stock Equity", "Absence of long-term equity incentive reduces employee lock-in."))

        if risk_flags:
            col_rf1, col_rf2 = st.columns(2)
            with col_rf1:
                st.markdown("**Identified Vulnerabilities:**")
                for title, desc in risk_flags:
                    st.warning(f"**{title}**: {desc}")
            with col_rf2:
                st.markdown("**Recommended HR Interventions:**")
                if overtime == "Yes":
                    st.info("• Re-evaluate workload distribution and offer compensatory time-off.")
                if monthly_income < 3500 or stock_option_level == 0:
                    st.info("• Conduct compensation benchmarking and consider equity grant adjustments.")
                if years_since_promotion >= 4:
                    st.info("• Initiate a structured career development plan or lateral growth role.")
                if work_life_balance <= 2:
                    st.info("• Introduce hybrid/flexible working arrangements.")
        else:
            st.success("No critical risk factors detected. This employee displays healthy engagement metrics across compensation, tenure, and satisfaction.")


# -----------------------------------------------------------------------------
# MODE 2: Batch CSV Prediction
# -----------------------------------------------------------------------------
elif app_mode == "📁 Batch CSV Prediction":
    st.markdown('<div class="main-title">Batch Attrition Prediction</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload a CSV file containing employee records to compute predictions and risk probabilities in bulk.</div>',
        unsafe_allow_html=True,
    )

    # Sample template download
    sample_df = pd.DataFrame([
        {
            "Age": 32,
            "DailyRate": 800,
            "DistanceFromHome": 4,
            "Education": 3,
            "EnvironmentSatisfaction": 3,
            "Gender": "Male",
            "HourlyRate": 60,
            "JobInvolvement": 3,
            "JobLevel": 2,
            "JobSatisfaction": 3,
            "MonthlyIncome": 5200,
            "MonthlyRate": 14000,
            "NumCompaniesWorked": 1,
            "OverTime": "No",
            "PercentSalaryHike": 15,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 3,
            "StockOptionLevel": 1,
            "TotalWorkingYears": 9,
            "TrainingTimesLastYear": 3,
            "WorkLifeBalance": 3,
            "YearsAtCompany": 7,
            "YearsInCurrentRole": 4,
            "YearsSinceLastPromotion": 1,
            "YearsWithCurrManager": 5,
            "BusinessTravel": "Travel_Rarely",
            "Department": "Research & Development",
            "EducationField": "Life Sciences",
            "JobRole": "Research Scientist",
            "MaritalStatus": "Married",
        },
        {
            "Age": 24,
            "DailyRate": 400,
            "DistanceFromHome": 25,
            "Education": 2,
            "EnvironmentSatisfaction": 1,
            "Gender": "Female",
            "HourlyRate": 45,
            "JobInvolvement": 1,
            "JobLevel": 1,
            "JobSatisfaction": 1,
            "MonthlyIncome": 2100,
            "MonthlyRate": 6000,
            "NumCompaniesWorked": 4,
            "OverTime": "Yes",
            "PercentSalaryHike": 11,
            "PerformanceRating": 3,
            "RelationshipSatisfaction": 1,
            "StockOptionLevel": 0,
            "TotalWorkingYears": 2,
            "TrainingTimesLastYear": 1,
            "WorkLifeBalance": 1,
            "YearsAtCompany": 1,
            "YearsInCurrentRole": 1,
            "YearsSinceLastPromotion": 0,
            "YearsWithCurrManager": 0,
            "BusinessTravel": "Travel_Frequently",
            "Department": "Sales",
            "EducationField": "Technical Degree",
            "JobRole": "Sales Representative",
            "MaritalStatus": "Single",
        },
    ])

    col_btn, _ = st.columns([2, 3])
    with col_btn:
        st.download_button(
            label="📥 Download Sample Batch Template (CSV)",
            data=sample_df.to_csv(index=False).encode("utf-8"),
            file_name="employee_attrition_sample_template.csv",
            mime="text/csv",
        )

    st.markdown("---")
    uploaded_file = st.file_uploader("Upload Employee Records (CSV)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_raw)} records.")

            with st.expander("Preview Uploaded Raw Data", expanded=False):
                st.dataframe(batch_raw.head(), use_container_width=True)

            # Check if columns are already pre-encoded or raw
            is_pre_encoded = all(col in batch_raw.columns for col in feature_columns)

            if is_pre_encoded:
                X_batch = batch_raw[feature_columns]
            else:
                # Transform each row
                transformed_rows = []
                for _, raw_row in batch_raw.iterrows():
                    row_df = transform_to_features(raw_row.to_dict(), feature_columns)
                    transformed_rows.append(row_df)
                X_batch = pd.concat(transformed_rows, ignore_index=True)

            # Run predictions
            preds = rf_model.predict(X_batch)
            probas = rf_model.predict_proba(X_batch)[:, 1]

            results_df = batch_raw.copy()
            results_df["Predicted_Attrition"] = np.where(preds == 1, "Yes", "No")
            results_df["Attrition_Probability (%)"] = (probas * 100).round(2)
            results_df["Risk_Level"] = pd.cut(
                probas,
                bins=[-0.01, 0.35, 0.65, 1.0],
                labels=["Low Risk", "Moderate Risk", "High Risk"],
            )

            # Metrics
            total_count = len(results_df)
            attrition_count = int((preds == 1).sum())
            attrition_rate = (attrition_count / total_count) * 100

            st.markdown("### Batch Analysis Summary")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Total Records Evaluated", total_count)
            with c2:
                st.metric("Predicted Attritions", attrition_count)
            with c3:
                st.metric("Projected Attrition Rate", f"{attrition_rate:.1f}%")

            # Risk Category Distribution
            risk_counts = results_df["Risk_Level"].value_counts()
            st.bar_chart(risk_counts)

            st.markdown("### Detailed Predictions Table")
            st.dataframe(results_df, use_container_width=True)

            # Download Predictions
            csv_export = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Export Predictions to CSV",
                data=csv_export,
                file_name="attrition_predictions_output.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Failed to process CSV file: {e}")


# -----------------------------------------------------------------------------
# MODE 3: Model Insights & Feature Importance
# -----------------------------------------------------------------------------
elif app_mode == "📊 Model Insights & Feature Importance":
    st.markdown('<div class="main-title">Random Forest Model Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Detailed diagnostic view of the trained ensemble model and feature importance ranking.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Configuration")
        config_data = {
            "Parameter": [
                "Estimator",
                "Number of Estimators (n_estimators)",
                "Max Depth",
                "Min Samples Split",
                "Min Samples Leaf",
                "Max Features",
                "Bootstrap",
                "OOB Score Enabled",
                "Feature Dimension",
            ],
            "Value": [
                "RandomForestClassifier",
                getattr(rf_model, "n_estimators", 60),
                getattr(rf_model, "max_depth", 16),
                getattr(rf_model, "min_samples_split", 5),
                getattr(rf_model, "min_samples_leaf", 1),
                str(getattr(rf_model, "max_features", "sqrt")),
                str(getattr(rf_model, "bootstrap", True)),
                str(getattr(rf_model, "oob_score", True)),
                f"{len(feature_columns)} columns",
            ],
        }
        st.dataframe(pd.DataFrame(config_data), hide_index=True, use_container_width=True)

    with col2:
        st.subheader("Target Distribution & Encoding")
        st.markdown(
            """
            - **Target Variable**: `Attrition`
              - `0`: Employee Remains (Retention)
              - `1`: Employee Departs (Attrition)
            - **Categorical Encoding**:
              - Dummy variables created with `drop_first=True`
              - Binary mapping: `Gender` (Female=1, Male=0), `OverTime` (Yes=1, No=0)
            """
        )

    st.markdown("---")
    st.subheader("Top Feature Importances (Gini Impurity Reduction)")

    if hasattr(rf_model, "feature_importances_"):
        importances = rf_model.feature_importances_
        fi_df = pd.DataFrame({
            "Feature": feature_columns,
            "Importance": importances,
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        col_top, col_chart = st.columns([1, 2])
        with col_top:
            st.markdown("**Top 15 Most Influential Features:**")
            st.dataframe(
                fi_df.head(15).style.format({"Importance": "{:.4f}"}),
                hide_index=True,
                use_container_width=True,
            )
        with col_chart:
            st.markdown("**Feature Importance Chart:**")
            top15 = fi_df.head(15).set_index("Feature")["Importance"]
            st.bar_chart(top15)
    else:
        st.info("Feature importances not available on this model instance.")
