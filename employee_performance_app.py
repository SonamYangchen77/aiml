import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Load the trained model
model = joblib.load("decision_tree_model.pkl")

# Set up the Streamlit app
st.set_page_config(page_title="Employee Performance Predictor", layout="wide")

# Create navigation
page = st.sidebar.selectbox("Choose a page", ["Performance Predictor", "Exploratory Data Analysis"])

if page == "Performance Predictor":
    st.title("Employee Performance Predictor")
    st.write("🚀 Interactive prediction app")

    # Create input form in sidebar
    with st.sidebar:
        st.header("Employee Information")
        
        employee_id = st.text_input("Employee ID", "1")
        
        department = st.selectbox(
            "Department",
            ["Sales", "Engineering", "Marketing", "HR", "Finance", "IT"]
        )
        
        gender = st.radio("Gender", ["Male", "Female", "Other"])
        
        age = st.slider("Age", 18, 65, 30)
        
        job_title = st.text_input("Job Title", "Software Engineer")
        
        years_at_company = st.slider("Years at Company", 0, 30, 3)
        
        years_since_hire = st.slider("Years Since Hire", 0, 40, 5)
        
        education_level = st.selectbox(
            "Education Level",
            ["High School", "Bachelor's", "Master's", "PhD"]
        )
        
        work_hours = st.slider("Work Hours Per Week", 20, 80, 40)
        
        projects = st.slider("Projects Handled", 0, 20, 3)
        
        overtime = st.slider("Overtime Hours", 0, 40, 5)
        
        sick_days = st.slider("Sick Days Taken", 0, 30, 2)
        
        remote_work = st.select_slider(
            "Remote Work Frequency", 
            options=["Never", "Rarely", "Sometimes", "Often", "Always"]
        )
        
        remote_work_mapping = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3, "Always": 4}
        remote_work_frequency = remote_work_mapping[remote_work]
        
        team_size = st.slider("Team Size", 1, 20, 5)
        
        training_hours = st.slider("Training Hours", 0, 100, 10)
        
        promotions = st.slider("Promotions Received", 0, 10, 0)
        
        satisfaction = st.slider("Employee Satisfaction Score", 1.0, 5.0, 3.5, step=0.1)
        
        monthly_salary = st.number_input("Monthly Salary ($)", 1000, 100000, 5000, step=100)
        
        resigned = st.checkbox("Has Resigned")

    # Create dictionary from inputs
    employee_data = {
        "Employee_ID": employee_id,
        "Department": department,
        "Gender": gender,
        "Age": age,
        "Job_Title": job_title,
        "Years_At_Company": years_at_company,
        "Years_Since_Hire": years_since_hire,
        "Education_Level": education_level,
        "Work_Hours_Per_Week": work_hours,
        "Projects_Handled": projects,
        "Overtime_Hours": overtime,
        "Sick_Days": sick_days,
        "Remote_Work_Frequency": remote_work_frequency,
        "Team_Size": team_size,
        "Training_Hours": training_hours,
        "Promotions": promotions,
        "Employee_Satisfaction_Score": satisfaction,
        "Monthly_Salary": monthly_salary,
        "Resigned": resigned
    }

    # Display the input data
    st.subheader("Employee Data Summary")
    st.json(employee_data)

    # Prediction button
    if st.button("Predict Performance Score"):
        try:
            # Convert to DataFrame
            df = pd.DataFrame([employee_data])
            
            # Make prediction
            prediction = model.predict(df)[0]
            
            # Display result
            st.success(f"Predicted Performance Score: {round(float(prediction), 2)}")
            
            if prediction >= 4.0:
                st.balloons()
                st.write("🌟 Excellent performance predicted!")
            elif prediction >= 3.0:
                st.write("👍 Good performance predicted")
            else:
                st.warning("⚠️ Below average performance predicted - may need improvement")
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")
    st.write("📊 Explore patterns and insights from employee performance data")

    @st.cache_data
    def load_sample_data():
        # Updated sample data to include required columns for new plots
        data = {
            "Department": ["Sales", "Engineering", "Marketing", "HR", "Finance", "IT"] * 50,
            "Age": [30, 35, 28, 45, 32, 29] * 50,
            "Gender": ["Male", "Female", "Other", "Male", "Female", "Male"] * 50,
            "Education_Level": ["Bachelor's", "Master's", "PhD", "Bachelor's", "High School", "Master's"] * 50,
            "Years_At_Company": [3, 5, 2, 10, 4, 1] * 50,
            "Performance_Score": [4.2, 3.8, 3.5, 4.5, 3.2, 3.9] * 50,
            "Monthly_Salary": [5000, 6000, 4500, 5500, 7000, 6500] * 50,
            "Projects_Handled": [3, 5, 2, 4, 6, 3] * 50,
            "Employee_Satisfaction_Score": [4.1, 3.5, 3.8, 4.2, 3.0, 4.0] * 50,
            "Overtime_Hours": [5, 10, 7, 8, 6, 9] * 50,
            "Years_Since_Hire": [1000, 1500, 800, 3000, 1200, 700] * 50,
            "Training_Hours": [20, 35, 15, 40, 30, 25] * 50,
        }
        return pd.DataFrame(data)

    df = load_sample_data()

    # Show raw data option
    if st.checkbox("Show raw data"):
        st.subheader("Raw Data")
        st.dataframe(df)

    st.subheader("Data Visualizations")

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set(style="whitegrid", palette="muted")

    fig, axes = plt.subplots(4, 2, figsize=(16, 24))
    axes = axes.flatten()

    # Plot 1: Histogram of Monthly Salary
    sns.histplot(df['Monthly_Salary'], kde=True, bins=30, ax=axes[0])
    axes[0].set_title("Distribution of Monthly Salary")

    # Plot 2: Boxplot of Monthly Salary
    sns.boxplot(x=df['Monthly_Salary'], ax=axes[1])
    axes[1].set_title("Boxplot of Monthly Salary")

    # Plot 3: Heatmap of Correlation Matrix
    corr_matrix = df.corr(numeric_only=True)
    sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, ax=axes[2])
    axes[2].set_title("Correlation Heatmap")

    # Plot 4: Histogram of Employee Satisfaction Score
    sns.histplot(df['Employee_Satisfaction_Score'], kde=True, bins=30, ax=axes[3])
    axes[3].set_title("Distribution of Employee Satisfaction Score")

    # Plot 5: Scatterplot - Salary vs Performance Score
    sns.scatterplot(x='Performance_Score', y='Monthly_Salary', data=df, ax=axes[4])
    axes[4].set_title("Salary vs Performance Score")

    # Plot 6: Boxplot of Overtime Hours by Department
    sns.boxplot(x='Department', y='Overtime_Hours', data=df, ax=axes[5])
    axes[5].set_title("Overtime Hours by Department")
    for label in axes[5].get_xticklabels():
        label.set_rotation(45)

    # Plot 7: Histogram of Tenure Days (Years_Since_Hire)
    sns.histplot(df['Years_Since_Hire'], kde=True, bins=30, ax=axes[6])
    axes[6].set_title("Distribution of Tenure Days")

    # Plot 8: Scatterplot - Training Hours vs Satisfaction
    sns.scatterplot(x='Training_Hours', y='Employee_Satisfaction_Score', data=df, ax=axes[7])
    axes[7].set_title("Training vs Satisfaction")

    plt.tight_layout()
    st.pyplot(fig)

    # Download button for sample data
    st.subheader("Download Sample Data")
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="employee_performance_sample.csv",
        mime="text/csv"
    )
