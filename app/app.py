import streamlit as st
import pandas as pd
import joblib
from catboost import Pool
from datetime import datetime
import io
import json

# Load model
@st.cache_resource
def load_model_components():
    """Load model and related components once and cache them"""
    model = joblib.load("models/catboost_salary_model.pkl")
    cat_features = joblib.load("models/cat_features.pkl")
    column_order = joblib.load("models/column_order.pkl")
    return model, cat_features, column_order

model, cat_features, column_order = load_model_components()

# Streamlit page configuration
st.set_page_config(
    page_title="Employee Salary Prediction",
    page_icon="💼",
    layout="centered",
    initial_sidebar_state="expanded"
)

if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

# Theme toggle function
def toggle_theme():
    """Toggle between light and dark themes"""
    st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

def apply_theme_styles():
    """Apply comprehensive theme styles for better visibility"""
    if st.session_state["theme"] == "dark":
        st.markdown("""
            <style>
                .stApp {
                    background-color: #0e1117;
                    color: #FAFAFA;
                }
                
                .main-header {
                    color: #4CAF50;
                    text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
                }
                
                .metric-card {
                    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                    padding: 20px;
                    border-radius: 10px;
                    border: 1px solid #444;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
                    color: white;
                    margin: 10px 0;
                }
                
                .prediction-success {
                    background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    border: none;
                    box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
                    text-align: center;
                    font-weight: bold;
                    font-size: 18px;
                }
                
                .prediction-error {
                    background: linear-gradient(135deg, #f44336 0%, #da190b 100%);  
                    color: white;
                    padding: 20px;
                    border-radius: 10px;
                    border: none;
                    box-shadow: 0 4px 15px rgba(244, 67, 54, 0.3);
                    text-align: center;
                    font-weight: bold;
                    font-size: 18px;
                }
                
                .stSelectbox > div > div {
                    background-color: #262730;
                    color: #FAFAFA;
                }
                
                .stNumberInput > div > div > input {
                    background-color: #262730;
                    color: #FAFAFA;
                }
                
                .stSlider > div > div > div {
                    color: #FAFAFA;
                }
                
                .welcome-card {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px;
                    border-radius: 15px;
                    color: white;
                    text-align: center;
                    margin: 20px 0;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.3);
                }
            </style>
        """, unsafe_allow_html=True)

apply_theme_styles()

col_header, col_toggle = st.columns([10, 1])
with col_header:
    st.markdown("""
        <h1 class='main-header' style='text-align: center; margin-bottom: 0;'>💼 Employee Salary Prediction</h1>
    """, unsafe_allow_html=True)


# Model performance metrics
st.markdown("""
    <div class='metric-card'>
        <div style='text-align: center;'>
            <h3 style='margin: 0; color: inherit;'>🎯 Model Performance</h3>
            <div style='display: flex; justify-content: space-around; margin-top: 15px;'>
                <div>
                    <h2 style='margin: 0; color: inherit;'>87.35%</h2>
                    <p style='margin: 0; color: inherit;'>Accuracy</p>
                </div>
                <div>
                    <h2 style='margin: 0; color: inherit;'>0.72</h2>
                    <p style='margin: 0; color: inherit;'>F1 Score</p>
                </div>
                <div>
                    <h2 style='margin: 0; color: inherit;'>CatBoost</h2>
                    <p style='margin: 0; color: inherit;'>Algorithm</p>
                </div>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar navigation 
st.sidebar.markdown("## 📊 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "📥 Predict Salary", "📈 Batch Processing"])

# prediction history
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# Home Page
if page == "🏠 Home":
    st.markdown("""
        <div class='welcome-card'>
            <h2 style='margin-top: 0;'>👋 Welcome to Salary Predictor!</h2>
            <p style='font-size: 18px; margin: 20px 0;'>
                Predict whether an employee is likely to earn <strong>more than $50K/year</strong> 
                based on demographic and professional information.
            </p>
            <div style='display: flex; justify-content: center; gap: 30px; margin-top: 25px;'>
                <div>
                    <h3 style='margin: 5px 0;'>🎯</h3>
                    <p style='margin: 0;'>High Accuracy</p>
                </div>
                <div>
                    <h3 style='margin: 5px 0;'>⚡</h3>
                    <p style='margin: 0;'>Fast Predictions</p>
                </div>
                <div>
                    <h3 style='margin: 5px 0;'>📊</h3>
                    <p style='margin: 0;'>Detailed Reports</p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Features")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Individual Predictions**\n\nGet instant salary predictions for single employees with detailed reports.")
    
    with col2:
        st.info("**Batch Processing**\n\nUpload CSV files to predict salaries for multiple employees at once.")
    
    with col3:
        st.info("**Prediction History**\n\nTrack and download your previous predictions for analysis.")
    
    st.markdown("### 📋 How to Use")
    st.markdown("""
    1. **Navigate** to the 'Predict Salary' page using the sidebar
    2. **Fill in** the employee details in the form
    3. **Click** 'Predict Salary' to get instant results
    4. **Download** detailed reports for your records
    5. **Use** batch processing for multiple predictions
    """)

# Prediction Page
elif page == "📥 Predict Salary":
    st.markdown("## 📥 Enter Employee Details")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", 18, 90, 30, help="Employee's age in years")
            workclass = st.selectbox("Workclass", [
                'Private', 'Self-emp-not-inc', 'Local-gov', 'State-gov',
                'Self-emp-inc', 'Federal-gov', 'Without-pay'], 
                help="Type of employment")
            education = st.selectbox("Education", [
                'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
                'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th',
                'Doctorate', 'Prof-school'], 
                help="Highest education level achieved")
            marital_status = st.selectbox("Marital Status", [
                'Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Widowed'],
                help="Current marital status")
            occupation = st.selectbox("Occupation", [
                'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners', 'Prof-specialty',
                'Other-service', 'Sales', 'Craft-repair', 'Transport-moving',
                'Machine-op-inspct', 'Tech-support'], 
                help="Job category")
            relationship = st.selectbox("Relationship", [
                'Not-in-family', 'Husband', 'Wife', 'Own-child', 'Unmarried'],
                help="Relationship status in household")
        
        with col2:
            race = st.selectbox("Race", [
                'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
                help="Racial background")
            gender = st.radio("Gender", ['Male', 'Female'], help="Gender")
            hours_per_week = st.slider("Hours per week", 1, 99, 40, 
                                     help="Average working hours per week")
            native_country = st.selectbox("Native Country", [
                'United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada'],
                help="Country of origin")
            fnlwgt = st.number_input("Final Weight (fnlwgt)", 10000, 1000000, 200000,
                                   help="Census weight (statistical weighting)")
            education_num = st.slider("Education Number", 1, 16, 10,
                                    help="Numeric representation of education level")
            capital_gain = st.number_input("Capital Gain", 0, 100000, 0,
                                         help="Income from investments")
            capital_loss = st.number_input("Capital Loss", 0, 5000, 0,
                                         help="Loss from investments")
        
        # Center the submit button
        col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
        with col_btn2:
            submitted = st.form_submit_button("🎯 Predict Salary", use_container_width=True)
    
    if submitted:
        with st.spinner("🔍 Analyzing employee profile..."):
            # Prepare input data
            input_data = pd.DataFrame([{
                'age': age,
                'workclass': workclass,
                'fnlwgt': fnlwgt,
                'education': education,
                'educational-num': education_num,
                'marital-status': marital_status,
                'occupation': occupation,
                'relationship': relationship,
                'race': race,
                'gender': gender,
                'capital-gain': capital_gain,
                'capital-loss': capital_loss,
                'hours-per-week': hours_per_week,
                'native-country': native_country
            }])
            
            input_data = input_data[column_order]
            prediction_pool = Pool(input_data, cat_features=cat_features)
            pred = model.predict(prediction_pool)[0]
            
            # prediction probability
            pred_proba = model.predict_proba(prediction_pool)[0]
            confidence = max(pred_proba) * 100
        
        # Display prediction
        st.markdown("### 🎯 Prediction Result")
        
        if pred == 1:
            st.markdown(f"""
                <div class='prediction-success'>
                    ✅ <strong>Likely to earn MORE than $50K/year</strong><br>
                    <small>Confidence: {confidence:.1f}%</small>
                </div>
            """, unsafe_allow_html=True)
            prediction_text = "More than $50K/year"
        else:
            st.markdown(f"""
                <div class='prediction-error'>
                    ❌ <strong>Likely to earn LESS than or equal to $50K/year</strong><br>
                    <small>Confidence: {confidence:.1f}%</small>
                </div>
            """, unsafe_allow_html=True)
            prediction_text = "Less than or equal to $50K/year"
        
        # prediction history
        prediction_record = {
            'timestamp': datetime.now(),
            'prediction': prediction_text,
            'confidence': confidence,
            'details': {
                'age': age, 'workclass': workclass, 'education': education,
                'marital_status': marital_status, 'occupation': occupation,
                'relationship': relationship, 'race': race, 'gender': gender,
                'hours_per_week': hours_per_week, 'native_country': native_country,
                'capital_gain': capital_gain, 'capital_loss': capital_loss
            }
        }
        st.session_state.prediction_history.append(prediction_record)
        
        # report generation
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        report_text = f"""EMPLOYEE SALARY PREDICTION REPORT
{'='*50}

Prediction Timestamp: {timestamp}
Prediction: {prediction_text}
Confidence Level: {confidence:.1f}%

EMPLOYEE PROFILE
{'='*20}
Personal Information:
  • Age: {age} years
  • Gender: {gender}
  • Race: {race}
  • Marital Status: {marital_status}
  • Relationship: {relationship}
  • Native Country: {native_country}

Professional Information:
  • Workclass: {workclass}
  • Occupation: {occupation}
  • Education: {education} (Level: {education_num})
  • Hours per Week: {hours_per_week}

Financial Information:
  • Capital Gain: ${capital_gain:,}
  • Capital Loss: ${capital_loss:,}
  • Final Weight: {fnlwgt:,}

MODEL INFORMATION
{'='*20}
Algorithm: CatBoost Classifier
Model Accuracy: 87.35%
F1 Score: 0.72

DISCLAIMER
{'='*20}
This prediction is based on statistical modeling and should be used 
for informational purposes only. Actual salary outcomes may vary 
based on numerous factors not captured in this model.

Generated by Employee Salary Prediction System
        """
        
        # download options
        st.markdown("### 📄 Download Options")
        
        col_download1, col_download2 = st.columns(2)
        
        with col_download1:
            # Text report
            text_bytes = io.BytesIO(report_text.encode("utf-8"))
            st.download_button(
                label="📄 Download TXT Report",
                data=text_bytes,
                file_name=f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col_download2:
            # JSON report
            json_data = {
                'timestamp': timestamp,
                'prediction': prediction_text,
                'confidence': confidence,
                'employee_details': prediction_record['details'],
                'model_info': {
                    'algorithm': 'CatBoost Classifier',
                    'accuracy': '87.35%',
                    'f1_score': '0.72'
                }
            }
            json_bytes = io.BytesIO(json.dumps(json_data, indent=2).encode("utf-8"))
            st.download_button(
                label="📊 Download JSON Report",
                data=json_bytes,
                file_name=f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )

# New Batch Processing Page
elif page == "📈 Batch Processing":
    st.markdown("## 📈 Batch Processing")
    st.markdown("Upload a CSV file to predict salaries for multiple employees at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", 
                                   help="Upload a CSV file with employee data")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully! Found {len(df)} records.")
            
            # Show data preview
            st.markdown("### Data Preview")
            st.dataframe(df.head())
            
            if st.button("🚀 Process All Records"):
                with st.spinner("Processing all records..."):
                    df_processed = df[column_order] if all(col in df.columns for col in column_order) else df
                    
                    # Making predictions
                    prediction_pool = Pool(df_processed, cat_features=cat_features)
                    predictions = model.predict(prediction_pool)
                    probabilities = model.predict_proba(prediction_pool)
                    
                    # Adding the results to dataframe
                    df['Prediction'] = ['More than $50K' if p == 1 else 'Less than or equal to $50K' for p in predictions]
                    df['Confidence'] = [max(prob) * 100 for prob in probabilities]
                    
                    st.success(f"✅ Processed {len(df)} records successfully!")
                    
                    # Showing results summary
                    high_earners = sum(predictions)
                    low_earners = len(predictions) - high_earners
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("High Earners (>$50K)", high_earners)
                    with col3:
                        st.metric("Low Earners (≤$50K)", low_earners)
                    
                    # Displaying the results
                    st.markdown("### 📊 Results")
                    st.dataframe(df)
                    
                    # Download processed file
                    csv_bytes = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Download Results CSV",
                        data=csv_bytes,
                        file_name=f"salary_predictions_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please ensure your CSV file has the correct column names and data format.")

# Prediction History in Sidebar
if st.session_state.prediction_history:
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Recent Predictions")
    st.sidebar.markdown(f"**Total Predictions:** {len(st.session_state.prediction_history)}")
    
    # Show last 3 predictions
    for i, record in enumerate(st.session_state.prediction_history[-3:]):
        with st.sidebar.expander(f"Prediction {len(st.session_state.prediction_history) - i}"):
            st.write(f"**Time:** {record['timestamp'].strftime('%H:%M:%S')}")
            st.write(f"**Result:** {record['prediction']}")
            st.write(f"**Confidence:** {record['confidence']:.1f}%")
    
    # Clear history button
    if st.sidebar.button("🗑️ Clear History"):
        st.session_state.prediction_history = []
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Employee Salary Prediction System | Powered by CatBoost ML Algorithm</p>
    </div>
""", unsafe_allow_html=True)