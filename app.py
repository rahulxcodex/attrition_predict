import json
import warnings
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")

st.title("Employee Attrition Prediction")
st.write("Enter the employee details below to predict attrition based on the trained model features.")

# Load model and feature columns
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

# Input form
with st.form("attrition_prediction_form"):
    st.subheader("Employee Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("MaritalStatus", ["Single", "Married", "Divorced"])
        distance_from_home = st.number_input("DistanceFromHome", min_value=1, max_value=50, value=5, step=1)
        education = st.selectbox("Education (1-5)", [1, 2, 3, 4, 5], index=2)
        education_field = st.selectbox(
            "EducationField",
            ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Human Resources", "Other"],
        )
        num_companies_worked = st.number_input("NumCompaniesWorked", min_value=0, max_value=15, value=2, step=1)
        total_working_years = st.number_input("TotalWorkingYears", min_value=0, max_value=40, value=8, step=1)

    with col2:
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
        job_level = st.selectbox("JobLevel (1-5)", [1, 2, 3, 4, 5], index=1)
        business_travel = st.selectbox("BusinessTravel", ["Travel_Rarely", "Travel_Frequently", "Non-Travel"])
        overtime = st.selectbox("OverTime", ["No", "Yes"])
        years_at_company = st.number_input("YearsAtCompany", min_value=0, max_value=40, value=5, step=1)
        years_in_current_role = st.number_input("YearsInCurrentRole", min_value=0, max_value=25, value=3, step=1)
        years_since_last_promotion = st.number_input("YearsSinceLastPromotion", min_value=0, max_value=20, value=1, step=1)
        years_with_curr_manager = st.number_input("YearsWithCurrManager", min_value=0, max_value=25, value=3, step=1)

    with col3:
        monthly_income = st.number_input("MonthlyIncome", min_value=1000, max_value=30000, value=5000, step=100)
        daily_rate = st.number_input("DailyRate", min_value=100, max_value=2000, value=800, step=10)
        hourly_rate = st.number_input("HourlyRate", min_value=20, max_value=150, value=65, step=1)
        monthly_rate = st.number_input("MonthlyRate", min_value=1000, max_value=40000, value=14000, step=100)
        percent_salary_hike = st.number_input("PercentSalaryHike", min_value=10, max_value=30, value=15, step=1)
        stock_option_level = st.selectbox("StockOptionLevel (0-3)", [0, 1, 2, 3], index=1)
        environment_satisfaction = st.selectbox("EnvironmentSatisfaction (1-4)", [1, 2, 3, 4], index=2)
        job_satisfaction = st.selectbox("JobSatisfaction (1-4)", [1, 2, 3, 4], index=2)
        job_involvement = st.selectbox("JobInvolvement (1-4)", [1, 2, 3, 4], index=2)
        relationship_satisfaction = st.selectbox("RelationshipSatisfaction (1-4)", [1, 2, 3, 4], index=2)
        work_life_balance = st.selectbox("WorkLifeBalance (1-4)", [1, 2, 3, 4], index=2)
        performance_rating = st.selectbox("PerformanceRating (3-4)", [3, 4], index=0)
        training_times_last_year = st.number_input("TrainingTimesLastYear", min_value=0, max_value=10, value=3, step=1)

    submit = st.form_submit_button("Predict Attrition", type="primary")

if submit:
    # Construct exact feature dictionary matching feature_columns.json
    row = {col: 0 for col in feature_columns}

    # Numeric and ordinal features
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

    # Binary features
    row["Gender"] = 1 if gender == "Female" else 0
    row["OverTime"] = 1 if overtime == "Yes" else 0

    # Categorical one-hot encoded features
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

    # Create DataFrame strictly with the json columns
    input_df = pd.DataFrame([row])[feature_columns]

    # Predict
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.markdown("---")
    st.subheader("Prediction Result")

    col_res1, col_res2 = st.columns(2)
    with col_res1:
        if prediction == 1:
            st.error("Attrition: Yes (Employee is likely to leave)")
        else:
            st.success("Attrition: No (Employee is likely to stay)")

    with col_res2:
        st.write(f"**Probability of Attrition (Yes):** {probabilities[1] * 100:.2f}%")
        st.write(f"**Probability of Retention (No):** {probabilities[0] * 100:.2f}%")
