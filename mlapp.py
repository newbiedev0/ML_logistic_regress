import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression

def winsorize_all_columns(X):
    for column in X.columns:
        lower_bound = X[column].quantile(0.05)
        upper_bound = X[column].quantile(0.95)
        X[column] = np.clip(X[column], lower_bound, upper_bound)
    return X

@st.cache_resource
def load_models():
    try:
        with open('attrition_model.pkl', 'rb') as f:
            attrition_model = pickle.load(f)
        with open('performance_model.pkl', 'rb') as f:
            performance_model = pickle.load(f)
        return attrition_model, performance_model
    except FileNotFoundError:
        st.error("Error: Model pickle files not found. Please ensure 'attrition_model.pkl' and 'performance_model.pkl' are in the same directory.")
        st.stop()

attrition_model, performance_model = load_models()

try:
    df = pd.read_csv('C:/Users/venka/Downloads/Employee-Attrition.csv')
except FileNotFoundError:
    st.error("Error: 'Employee-Attrition.csv' not found. Please make sure the file is in the same directory as the app.")
    st.stop()

st.set_page_config(page_title="Employee Predictive Models", layout="wide")
st.title("Employee Predictive Models")
st.write("Use this application to predict an employee's Attrition or Performance Rating.")

tab1, tab2 = st.tabs(["Predict Attrition", "Predict Performance Rating"])

with tab1:
    st.header("Predict Employee Attrition")
    with st.sidebar:
        st.header("Employee Data Input")
        age = st.number_input("Age", min_value=18, max_value=60, value=30, key='age_att')
        business_travel = st.selectbox("Business Travel", df['BusinessTravel'].unique(), key='bt_att')
        daily_rate = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800, key='dr_att')
        department = st.selectbox("Department", df['Department'].unique(), key='dept_att')
        distance_from_home = st.number_input("Distance From Home", min_value=1, max_value=30, value=5, key='dfh_att')
        education = st.selectbox("Education", sorted(df['Education'].unique()), key='edu_att')
        education_field = st.selectbox("Education Field", df['EducationField'].unique(), key='ef_att')
        environment_satisfaction = st.selectbox("Environment Satisfaction", sorted(df['EnvironmentSatisfaction'].unique()), key='es_att')
        gender = st.selectbox("Gender", df['Gender'].unique(), key='gender_att')
        hourly_rate = st.number_input("Hourly Rate", min_value=30, max_value=100, value=70, key='hr_att')
        job_involvement = st.selectbox("Job Involvement", sorted(df['JobInvolvement'].unique()), key='ji_att')
        job_level = st.selectbox("Job Level", sorted(df['JobLevel'].unique()), key='jl_att')
        job_role = st.selectbox("Job Role", df['JobRole'].unique(), key='jr_att')
        job_satisfaction = st.selectbox("Job Satisfaction", sorted(df['JobSatisfaction'].unique()), key='js_att')
        marital_status = st.selectbox("Marital Status", df['MaritalStatus'].unique(), key='ms_att')
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, key='mi_att')
        monthly_rate = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=15000, key='mr_att')
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=9, value=1, key='ncw_att')
        overtime = st.selectbox("Over Time", df['OverTime'].unique(), key='ot_att')
        percent_salary_hike = st.number_input("Percent Salary Hike", min_value=11, max_value=25, value=15, key='psh_att')
        performance_rating = st.selectbox("Performance Rating", sorted(df['PerformanceRating'].unique()), key='pr_att')

        relationship_satisfaction = st.selectbox("Relationship Satisfaction", sorted(df['RelationshipSatisfaction'].unique()), key='rs_att')
        stock_option_level = st.selectbox("Stock Option Level", sorted(df['StockOptionLevel'].unique()), key='sol_att')
        total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, value=10, key='twy_att')
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=3, key='ttly_att')
        work_life_balance = st.selectbox("Work Life Balance", sorted(df['WorkLifeBalance'].unique()), key='wlb_att')
        years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=5, key='yac_att')
        years_in_current_role = st.number_input("Years in Current Role", min_value=0, max_value=18, value=3, key='yicr_att')
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1, key='yslp_att')
        years_with_curr_manager = st.number_input("Years with Current Manager", min_value=0, max_value=17, value=3, key='ywcm_att')
    
        if st.button("Predict Attrition"):
            input_data_att = pd.DataFrame([{
                'Age': age, 'BusinessTravel': business_travel, 'DailyRate': daily_rate, 'Department': department,
                'DistanceFromHome': distance_from_home, 'Education': education, 'EducationField': education_field,
                'EnvironmentSatisfaction': environment_satisfaction, 'Gender': gender, 'HourlyRate': hourly_rate,
                'JobInvolvement': job_involvement, 'JobLevel': job_level, 'JobRole': job_role, 'JobSatisfaction': job_satisfaction,
                'MaritalStatus': marital_status, 'MonthlyIncome': monthly_income, 'MonthlyRate': monthly_rate,
                'NumCompaniesWorked': num_companies_worked, 'OverTime': overtime, 'PercentSalaryHike': percent_salary_hike, 'PerformanceRating': performance_rating,
                'RelationshipSatisfaction': relationship_satisfaction, 'StockOptionLevel': stock_option_level,
                'TotalWorkingYears': total_working_years, 'TrainingTimesLastYear': training_times_last_year,
                'WorkLifeBalance': work_life_balance, 'YearsAtCompany': years_at_company,
                'YearsInCurrentRole': years_in_current_role, 'YearsSinceLastPromotion': years_since_last_promotion,
                'YearsWithCurrManager': years_with_curr_manager
            }])
            
            prediction = attrition_model.predict(input_data_att)[0]
            st.subheader("Prediction Result")
            st.success(f"The predicted attrition status is: **{prediction}**")

with tab2:
    st.header("Predict Employee Performance Rating")
    with st.sidebar:
        st.header("Employee Data Input (cont.)")
        age_p = st.number_input("Age", min_value=18, max_value=60, value=30, key='age_pr')
        business_travel_p = st.selectbox("Business Travel", df['BusinessTravel'].unique(), key='bt_pr')
        daily_rate_p = st.number_input("Daily Rate", min_value=100, max_value=1500, value=800, key='dr_pr')
        department_p = st.selectbox("Department", df['Department'].unique(), key='dept_pr')
        distance_from_home_p = st.number_input("Distance From Home", min_value=1, max_value=30, value=5, key='dfh_pr')
        education_p = st.selectbox("Education", sorted(df['Education'].unique()), key='edu_pr')
        education_field_p = st.selectbox("Education Field", df['EducationField'].unique(), key='ef_pr')
        environment_satisfaction_p = st.selectbox("Environment Satisfaction", sorted(df['EnvironmentSatisfaction'].unique()), key='es_pr')
        gender_p = st.selectbox("Gender", df['Gender'].unique(), key='gender_pr')
        hourly_rate_p = st.number_input("Hourly Rate", min_value=30, max_value=100, value=70, key='hr_pr')
        job_involvement_p = st.selectbox("Job Involvement", sorted(df['JobInvolvement'].unique()), key='ji_pr')
        job_level_p = st.selectbox("Job Level", sorted(df['JobLevel'].unique()), key='jl_pr')
        job_role_p = st.selectbox("Job Role", df['JobRole'].unique(), key='jr_pr')
        job_satisfaction_p = st.selectbox("Job Satisfaction", sorted(df['JobSatisfaction'].unique()), key='js_pr')
        marital_status_p = st.selectbox("Marital Status", df['MaritalStatus'].unique(), key='ms_pr')
        monthly_income_p = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, key='mi_pr')
        monthly_rate_p = st.number_input("Monthly Rate", min_value=2000, max_value=27000, value=15000, key='mr_pr')
        num_companies_worked_p = st.number_input("Number of Companies Worked", min_value=0, max_value=9, value=1, key='ncw_pr')
        overtime_p = st.selectbox("Over Time", df['OverTime'].unique(), key='ot_pr')
        percent_salary_hike_p = st.number_input("Percent Salary Hike", min_value=11, max_value=25, value=15, key='psh_pr')
        relationship_satisfaction_p = st.selectbox("Relationship Satisfaction", sorted(df['RelationshipSatisfaction'].unique()), key='rs_pr')
        stock_option_level_p = st.selectbox("Stock Option Level", sorted(df['StockOptionLevel'].unique()), key='sol_pr')
        total_working_years_p = st.number_input("Total Working Years", min_value=0, max_value=40, value=10, key='twy_pr')
        training_times_last_year_p = st.number_input("Training Times Last Year", min_value=0, max_value=6, value=3, key='ttly_pr')
        work_life_balance_p = st.selectbox("Work Life Balance", sorted(df['WorkLifeBalance'].unique()), key='wlb_pr')
        years_at_company_p = st.number_input("Years at Company", min_value=0, max_value=40, value=5, key='yac_pr')
        years_in_current_role_p = st.number_input("Years in Current Role", min_value=0, max_value=18, value=3, key='yicr_pr')
        years_since_last_promotion_p = st.number_input("Years Since Last Promotion", min_value=0, max_value=15, value=1, key='yslp_pr')
        years_with_curr_manager_p = st.number_input("Years with Current Manager", min_value=0, max_value=17, value=3, key='ywcm_pr')

        if st.button("Predict Performance Rating"):
            input_data_perf = pd.DataFrame([{
                'Age': age_p, 'BusinessTravel': business_travel_p, 'DailyRate': daily_rate_p, 'Department': department_p,
                'DistanceFromHome': distance_from_home_p, 'Education': education_p, 'EducationField': education_field_p,
                'EnvironmentSatisfaction': environment_satisfaction_p, 'Gender': gender_p, 'HourlyRate': hourly_rate_p,
                'JobInvolvement': job_involvement_p, 'JobLevel': job_level_p, 'JobRole': job_role_p, 'JobSatisfaction': job_satisfaction_p,
                'MaritalStatus': marital_status_p, 'MonthlyIncome': monthly_income_p, 'MonthlyRate': monthly_rate_p,
                'NumCompaniesWorked': num_companies_worked_p, 'OverTime': overtime_p, 'PercentSalaryHike': percent_salary_hike_p,
                'RelationshipSatisfaction': relationship_satisfaction_p, 'StockOptionLevel': stock_option_level_p,
                'TotalWorkingYears': total_working_years_p, 'TrainingTimesLastYear': training_times_last_year_p,
                'WorkLifeBalance': work_life_balance_p, 'YearsAtCompany': years_at_company_p,
                'YearsInCurrentRole': years_in_current_role_p, 'YearsSinceLastPromotion': years_since_last_promotion_p,
                'YearsWithCurrManager': years_with_curr_manager_p
            }])
            
            prediction = performance_model.predict(input_data_perf)[0]
            st.subheader("Prediction Result")
            st.success(f"The predicted performance rating is: **{prediction}**")