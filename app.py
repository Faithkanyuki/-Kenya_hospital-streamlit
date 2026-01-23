import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS FOR MEDICAL THEME
# ============================================================================
st.markdown("""
<style>
    /* Main theme colors - Medical Professional */
    :root {
        --primary-blue: #1a73e8;
        --secondary-blue: #4285f4;
        --accent-teal: #00bfa5;
        --warning-red: #d32f2f;
        --success-green: #2e7d32;
        --light-bg: #f8fafc;
        --card-bg: #ffffff;
        --border-color: #e2e8f0;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #1e3a8a;
        font-weight: 600;
    }
    
    /* Cards */
    .medical-card {
        background: var(--card-bg);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .medical-card-header {
        border-bottom: 2px solid var(--primary-blue);
        padding-bottom: 0.75rem;
        margin-bottom: 1rem;
        color: #1e3a8a;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary-blue), var(--secondary-blue));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(26, 115, 232, 0.3);
    }
    
    /* Metrics cards */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid var(--primary-blue);
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Risk indicators */
    .high-risk {
        background: linear-gradient(135deg, #fee, #fff5f5);
        border-left: 4px solid var(--warning-red);
    }
    
    .low-risk {
        background: linear-gradient(135deg, #f0fdf4, #f7fee7);
        border-left: 4px solid var(--success-green);
    }
    
    /* Input styling */
    .stNumberInput, .stSelectbox, .stSlider {
        background: white;
        border-radius: 6px;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #ffffff 100%);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 16px;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 1rem;
        margin-top: 2rem;
        border-top: 1px solid var(--border-color);
        color: #64748b;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PAGE SETUP
# ============================================================================
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    layout="wide",
    page_icon="🏥",
    initial_sidebar_state="expanded"
)

# ============================================================================
# HEADER WITH LOGO AND TITLE
# ============================================================================
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="color: #1e3a8a; margin-bottom: 0.5rem;">🏥 Kenya Hospital Readmission Predictor</h1>
        <p style="color: #64748b; font-size: 1.1rem;">Clinical Decision Support Tool for 30-Day Readmission Risk Assessment</p>
        <div style="height: 3px; background: linear-gradient(90deg, #1a73e8, #00bfa5); width: 100px; margin: 0 auto;"></div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL FUNCTION (Same as before)
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """Load model, features, and metadata with strict validation"""
    try:
        model = joblib.load("hospital_rf_20260121_streamlit.joblib")
        features = joblib.load("hospital_features_20260121.pkl")
        metadata = joblib.load("hospital_metadata_20260121.pkl")
        
        # Set default values if not in metadata
        if 'performance_metrics' not in metadata:
            metadata['performance_metrics'] = {
                'recall': 0.690,
                'precision': 0.154,
                'f1_score': 0.252,
                'roc_auc': 0.660
            }
        
        if 'model_info' not in metadata:
            metadata['model_info'] = {
                'optimal_threshold': 0.48,
                'n_estimators': 285,
                'max_depth': 5
            }
        
        return model, features, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, [], {}

# Load model
model, features, metadata = load_model_and_data()
threshold = metadata.get('model_info', {}).get('optimal_threshold', 0.48)

# ============================================================================
# PREDICTION FUNCTION (Same as before - shortened for brevity)
# ============================================================================
def predict_readmission_risk(user_inputs):
    """Make prediction with exact feature engineering from training"""
    if model is None:
        return None
    
    try:
        feature_dict = {feat: 0.0 for feat in features}
        input_df = pd.DataFrame([feature_dict])
        
        # Set numeric features
        numeric_fields = ['time_in_hospital', 'num_lab_procedures', 'num_medications', 
                         'total_hospital_visits', 'number_emergency', 'age_numeric']
        for field in numeric_fields:
            input_df[field] = float(user_inputs[field])
        
        # Set categorical features (mappings as before)
        gender_map = {"Female": 0, "Male": 1, "Unknown/Other": 2}
        gender_idx = gender_map[user_inputs['gender']]
        input_df[f'gender_{gender_idx}'] = 1.0
        
        admission_map = {
            "Emergency": 0, "Urgent": 1, "Elective": 2, "Newborn": 3,
            "Trauma Center": 4, "Not Mapped": 5, "NULL": 6, "Not Available": 7
        }
        admission_idx = admission_map[user_inputs['admission_type']]
        input_df[f'admission_type_{admission_idx}'] = 1.0
        
        discharge_map = {
            "Discharged to home": 0,
            "Discharged/transferred to another short term hospital": 1,
            "Discharged/transferred to SNF": 2,
            "Discharged/transferred to ICF": 3,
            "Discharged/transferred to another type of inpatient care institution": 4,
            "Discharged/transferred to home with home health service": 5,
            "Left AMA": 6,
            "Discharged/transferred to home under care of Home IV provider": 7,
            "Admitted as an inpatient to this hospital": 8,
            "Neonate discharged to another hospital": 9,
            "Expired": 10,
            "Still patient": 11,
            "Hospice / home": 12,
            "Hospice / medical facility": 13,
            "Discharged/transferred within this institution": 14,
            "Discharged/transferred to rehab": 15,
            "Discharged/transferred to another Medicare certified swing bed": 16,
            "Discharged/transferred to a long term care hospital": 17,
            "Discharged/transferred to a nursing facility certified under Medicaid": 18,
            "Discharged/transferred to a psychiatric hospital": 19,
            "Discharged/transferred to a critical access hospital": 20,
            "Discharged/transferred to another Type of Facility": 21,
            "Discharged/transferred to a court/law enforcement": 22,
            "Discharged/transferred to a Federal health care facility": 23,
            "Discharged/transferred to a hospital-based Medicare approved swing bed": 24,
            "Discharged/transferred to an inpatient rehabilitation facility": 25
        }
        discharge_idx = discharge_map[user_inputs['discharge_disposition']]
        input_df[f'discharge_disposition_{discharge_idx}'] = 1.0
        
        age_group_map = {"18-45": 0, "46-65": 1, "66-85": 2, "86+": 3}
        age_group_idx = age_group_map[user_inputs['age_group']]
        input_df[f'age_group_{age_group_idx}'] = 1.0
        
        input_df = input_df.astype(np.float32)
        probability = model.predict_proba(input_df)[0, 1]
        
        return probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

# Create tabs for better organization
tab1, tab2, tab3 = st.tabs(["📋 Patient Assessment", "📊 Model Insights", "⚙️ Settings & Info"])

with tab1:
    # Patient Assessment Section
    st.markdown("""
    <div class="medical-card">
        <h3 class="medical-card-header">Patient Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Use columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🏥 Clinical Parameters")
        
        # Group clinical inputs in a card
        with st.container():
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                time_in_hospital = st.slider(
                    "Hospital Stay (days)",
                    1, 30, 7,
                    help="Length of current hospital admission"
                )
                
                num_lab_procedures = st.number_input(
                    "Lab Procedures",
                    0, 200, 45,
                    help="Number of laboratory tests performed"
                )
                
                num_medications = st.number_input(
                    "Medications",
                    0, 100, 12,
                    help="Number of prescribed medications"
                )
            
            with col_b:
                total_hospital_visits = st.number_input(
                    "Total Visits (past year)",
                    0, 50, 3,
                    help="Total hospital admissions in the last 12 months"
                )
                
                number_emergency = st.number_input(
                    "Emergency Visits",
                    0, 20, 1,
                    help="Emergency department visits in the last year"
                )
                
                age_numeric = st.slider(
                    "Patient Age",
                    18, 100, 58,
                    help="Patient's current age"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 👤 Demographic Information")
        
        with st.container():
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            
            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Unknown/Other"],
                help="Patient's gender"
            )
            
            col_c, col_d = st.columns(2)
            with col_c:
                admission_type = st.selectbox(
                    "Admission Type",
                    ["Emergency", "Urgent", "Elective", "Newborn", 
                     "Trauma Center", "Not Mapped", "NULL", "Not Available"],
                    index=0
                )
                
                age_group = st.selectbox(
                    "Age Group",
                    ["18-45", "46-65", "66-85", "86+"],
                    index=1
                )
            
            with col_d:
                discharge_disposition = st.selectbox(
                    "Discharge Plan",
                    [
                        "Discharged to home",
                        "Discharged/transferred to another short term hospital",
                        "Discharged/transferred to SNF",
                        "Discharged/transferred to ICF",
                        "Discharged/transferred to another type of inpatient care institution",
                        "Discharged/transferred to home with home health service",
                        "Left AMA",
                        "Discharged/transferred to home under care of Home IV provider",
                        "Admitted as an inpatient to this hospital",
                        "Neonate discharged to another hospital",
                        "Expired",
                        "Still patient",
                        "Hospice / home",
                        "Hospice / medical facility",
                        "Discharged/transferred within this institution",
                        "Discharged/transferred to rehab",
                        "Discharged/transferred to another Medicare certified swing bed",
                        "Discharged/transferred to a long term care hospital",
                        "Discharged/transferred to a nursing facility certified under Medicaid",
                        "Discharged/transferred to a psychiatric hospital",
                        "Discharged/transferred to a critical access hospital",
                        "Discharged/transferred to another Type of Facility",
                        "Discharged/transferred to a court/law enforcement",
                        "Discharged/transferred to a Federal health care facility",
                        "Discharged/transferred to a hospital-based Medicare approved swing bed",
                        "Discharged/transferred to an inpatient rehabilitation facility"
                    ],
                    index=0
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Predict Button - Centered and prominent
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_clicked = st.button(
            "🔍 Calculate Readmission Risk",
            type="primary",
            use_container_width=True,
            help="Click to analyze patient's readmission risk"
        )
    
    # Results Display
    if predict_clicked:
        if model is None:
            st.error("Model not loaded. Please check configuration.")
        else:
            with st.spinner("🔄 Analyzing patient data..."):
                user_inputs = {
                    'time_in_hospital': time_in_hospital,
                    'num_lab_procedures': num_lab_procedures,
                    'num_medications': num_medications,
                    'total_hospital_visits': total_hospital_visits,
                    'number_emergency': number_emergency,
                    'age_numeric': age_numeric,
                    'gender': gender,
                    'admission_type': admission_type,
                    'discharge_disposition': discharge_disposition,
                    'age_group': age_group
                }
                
                probability = predict_readmission_risk(user_inputs)
                
                if probability is not None:
                    # Results Header
                    st.markdown("---")
                    st.markdown("""
                    <div style="text-align: center; margin: 2rem 0;">
                        <h2 style="color: #1e3a8a;">📈 Risk Assessment Results</h2>
                        <p style="color: #64748b;">Based on analysis of 81,412 patient records</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk Score Card
                    risk_class = "high-risk" if probability >= threshold else "low-risk"
                    risk_label = "🔴 HIGH RISK" if probability >= threshold else "🟢 LOW RISK"
                    
                    col_result1, col_result2, col_result3 = st.columns(3)
                    
                    with col_result1:
                        st.markdown(f"""
                        <div class="metric-card {risk_class}" style="text-align: center;">
                            <h3 style="margin: 0; color: #334155;">Risk Probability</h3>
                            <h1 style="color: {'#d32f2f' if probability >= threshold else '#2e7d32'}; margin: 1rem 0;">
                                {probability:.1%}
                            </h1>
                            <p style="color: #64748b; margin: 0;">
                                Threshold: {threshold:.1%}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_result2:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h3 style="margin: 0; color: #334155;">Risk Classification</h3>
                            <h1 style="margin: 1rem 0; color: {'#d32f2f' if probability >= threshold else '#2e7d32'}">
                                {risk_label}
                            </h1>
                            <p style="color: #64748b; margin: 0;">
                                {'Priority intervention required' if probability >= threshold else 'Standard care protocol'}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_result3:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align: center;">
                            <h3 style="margin: 0; color: #334155;">Decision Support</h3>
                            <div style="margin: 1rem 0; font-size: 2rem;">
                                {'⚠️' if probability >= threshold else '✅'}
                            </div>
                            <p style="color: #64748b; margin: 0;">
                                {'Flag for care team' if probability >= threshold else 'Routine follow-up'}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Clinical Recommendations
                    st.markdown("---")
                    
                    if probability >= threshold:
                        st.markdown("""
                        <div class="medical-card high-risk">
                            <h4 style="color: #d32f2f; margin-bottom: 1rem;">🚨 Priority Clinical Actions Required</h4>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                                <div>
                                    <h5 style="color: #334155;">Immediate Actions</h5>
                                    <ul style="color: #475569;">
                                        <li>Schedule follow-up within 7 days</li>
                                        <li>Coordinate with home care services</li>
                                        <li>Review medication adherence plan</li>
                                        <li>Flag for multidisciplinary team review</li>
                                    </ul>
                                </div>
                                <div>
                                    <h5 style="color: #334155;">Patient Education</h5>
                                    <ul style="color: #475569;">
                                        <li>Provide emergency contact information</li>
                                        <li>Review warning signs and symptoms</li>
                                        <li>Schedule transportation assistance</li>
                                        <li>Arrange social work consultation</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="medical-card low-risk">
                            <h4 style="color: #2e7d32; margin-bottom: 1rem;">✅ Standard Care Protocol</h4>
                            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem;">
                                <div>
                                    <h5 style="color: #334155;">Recommended Actions</h5>
                                    <ul style="color: #475569;">
                                        <li>Schedule 30-day follow-up appointment</li>
                                        <li>Provide discharge instructions</li>
                                        <li>Review medications and side effects</li>
                                        <li>Coordinate with primary care provider</li>
                                    </ul>
                                </div>
                                <div>
                                    <h5 style="color: #334155;">Patient Resources</h5>
                                    <ul style="color: #475569;">
                                        <li>Educational materials provided</li>
                                        <li>Community resource information</li>
                                        <li>Self-management tools</li>
                                        <li>Telehealth options discussed</li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Key Risk Factors
                    st.markdown("""
                    <div class="medical-card">
                        <h4 style="color: #1e3a8a; margin-bottom: 1rem;">🎯 Identified Risk Factors</h4>
                    """, unsafe_allow_html=True)
                    
                    risk_factors = []
                    if total_hospital_visits >= 4:
                        risk_factors.append(f"**High hospital visits** ({total_hospital_visits} in past year)")
                    if discharge_disposition in ["Discharged/transferred to another short term hospital",
                                                 "Discharged/transferred to another Medicare certified swing bed",
                                                 "Discharged/transferred to home under care of Home IV provider"]:
                        risk_factors.append("**Complex discharge disposition**")
                    if number_emergency >= 2:
                        risk_factors.append(f"**Multiple ED visits** ({number_emergency})")
                    if time_in_hospital >= 10:
                        risk_factors.append(f"**Extended hospital stay** ({time_in_hospital} days)")
                    if num_medications >= 15:
                        risk_factors.append(f"**High medication burden** ({num_medications} medications)")
                    if age_numeric >= 75:
                        risk_factors.append(f"**Advanced age** ({age_numeric} years)")
                    
                    if risk_factors:
                        cols = st.columns(2)
                        for i, factor in enumerate(risk_factors):
                            with cols[i % 2]:
                                st.markdown(f"""
                                <div style="background: #f1f5f9; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem;">
                                    <span style="color: #475569;">• {factor}</span>
                                </div>
                                """, unsafe_allow_html=True)
                    else:
                        st.info("No significant risk factors identified. Patient profile suggests low readmission risk.")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    # Model Insights Tab
    st.markdown("""
    <div class="medical-card">
        <h3 class="medical-card-header">Model Performance & Insights</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_insight1, col_insight2 = st.columns(2)
    
    with col_insight1:
        st.markdown("### 📊 Model Performance")
        
        perf_metrics = metadata.get("performance_metrics", {})
        
        metrics_data = {
            "Metric": ["Recall", "Precision", "F1-Score", "ROC AUC"],
            "Value": [
                f"{perf_metrics.get('recall', 0.690):.1%}",
                f"{perf_metrics.get('precision', 0.154):.1%}",
                f"{perf_metrics.get('f1_score', 0.252):.3f}",
                f"{perf_metrics.get('roc_auc', 0.660):.3f}"
            ],
            "Description": [
                "Identifies 69% of actual readmissions",
                "15.4% of flagged cases are true positives",
                "Balance between precision and recall",
                "Overall model discrimination ability"
            ]
        }
        
        for i in range(len(metrics_data["Metric"])):
            with st.container():
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong style="color: #334155;">{metrics_data['Metric'][i]}</strong>
                            <p style="color: #64748b; margin: 0.25rem 0 0 0; font-size: 0.9rem;">
                                {metrics_data['Description'][i]}
                            </p>
                        </div>
                        <span style="font-size: 1.25rem; font-weight: 600; color: #1a73e8;">
                            {metrics_data['Value'][i]}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    with col_insight2:
        st.markdown("### 🎯 Top Predictors")
        
        predictors = [
            ("Total Hospital Visits", "47.98%", "Most important factor"),
            ("Discharge to Another Hospital", "15.15%", "High risk disposition"),
            ("Discharge to Swing Bed", "13.57%", "Complex care needs"),
            ("Home IV Care Discharge", "7.84%", "Home care complexity"),
            ("Emergency Visits", "3.49%", "Healthcare utilization")
        ]
        
        for predictor, weight, description in predictors:
            with st.container():
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; 
                            border-left: 4px solid #00bfa5; margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div>
                            <strong style="color: #334155;">{predictor}</strong>
                            <p style="color: #64748b; margin: 0.25rem 0 0 0; font-size: 0.9rem;">
                                {description}
                            </p>
                        </div>
                        <span style="font-size: 1rem; font-weight: 600; color: #00bfa5;">
                            {weight}
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("""
        <div style="background: #f0f9ff; padding: 1rem; border-radius: 8px; border: 1px solid #bae6fd;">
            <h5 style="color: #0369a1; margin: 0 0 0.5rem 0;">🧠 Clinical Interpretation</h5>
            <p style="color: #475569; margin: 0; font-size: 0.9rem;">
                The model prioritizes <strong>catching high-risk patients</strong> (69% recall) over 
                reducing false alarms. This aligns with Kenyan healthcare priorities where 
                preventing missed high-risk cases is critical.
            </p>
        </div>
        """, unsafe_allow_html=True)

with tab3:
    # Settings & Info Tab
    st.markdown("""
    <div class="medical-card">
        <h3 class="medical-card-header">System Information & Settings</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("### ⚙️ Model Configuration")
        
        model_info = metadata.get("model_info", {})
        
        config_items = [
            ("Algorithm", "Random Forest"),
            ("Features", f"{len(features)}"),
            ("Estimators", f"{model_info.get('n_estimators', 285)}"),
            ("Max Depth", f"{model_info.get('max_depth', 5)}"),
            ("Threshold", f"{threshold:.1%}"),
            ("Training Samples", "81,412"),
            ("Class Balance", "11.2% readmitted")
        ]
        
        for label, value in config_items:
            with st.container():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; 
                            padding: 0.75rem; border-bottom: 1px solid #e2e8f0;">
                    <span style="color: #475569;">{label}</span>
                    <span style="color: #334155; font-weight: 500;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
    
    with col_info2:
        st.markdown("### 🔧 Technical Information")
        
        # System info
        import platform
        tech_items = [
            ("Python Version", sys.version.split()[0]),
            ("pandas Version", pd.__version__),
            ("numpy Version", np.__version__),
            ("Platform", platform.system()),
            ("Model Size", "~50 MB"),
            ("Prediction Speed", "< 1 second"),
            ("Last Updated", datetime.now().strftime("%Y-%m-%d"))
        ]
        
        for label, value in tech_items:
            with st.container():
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; 
                            padding: 0.75rem; border-bottom: 1px solid #e2e8f0;">
                    <span style="color: #475569;">{label}</span>
                    <span style="color: #334155; font-weight: 500;">{value}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Control buttons
        st.markdown("---")
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🔄 Reload Model", use_container_width=True):
                st.cache_resource.clear()
                st.rerun()
        
        with col_btn2:
            if st.button("🧪 Test Prediction", use_container_width=True):
                st.info("Test feature available in sidebar")

# ============================================================================
# SIDEBAR - Professional Medical Theme
# ============================================================================
with st.sidebar:
    # Logo and Title
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <div style="font-size: 2.5rem; color: #1a73e8; margin-bottom: 0.5rem;">🏥</div>
        <h3 style="color: #1e3a8a; margin: 0;">Readmission<br>Predictor</h3>
        <div style="height: 2px; background: linear-gradient(90deg, #1a73e8, #00bfa5); 
                    width: 60px; margin: 0.5rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Stats
    st.markdown("""
    <div style="background: #f8fafc; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h4 style="color: #334155; margin: 0 0 0.75rem 0;">📈 Quick Stats</h4>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 600; color: #1a73e8;">69%</div>
                <div style="font-size: 0.8rem; color: #64748b;">Recall Rate</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 600; color: #00bfa5;">48%</div>
                <div style="font-size: 0.8rem; color: #64748b;">Risk Threshold</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 📋 Navigation")
    nav_col1, nav_col2, nav_col3 = st.columns(3)
    with nav_col1:
        if st.button("🔄", help="Refresh Page"):
            st.rerun()
    with nav_col2:
        if st.button("📊", help="View Insights"):
            st.query_params["tab"] = "2"
            st.rerun()
    with nav_col3:
        if st.button("⚙️", help="Settings"):
            st.query_params["tab"] = "3"
            st.rerun()
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    
    if st.button("🧪 Run Sample Test", use_container_width=True):
        sample_inputs = {
            'time_in_hospital': 7,
            'num_lab_procedures': 45,
            'num_medications': 12,
            'total_hospital_visits': 3,
            'number_emergency': 1,
            'age_numeric': 58,
            'gender': "Female",
            'admission_type': "Emergency",
            'discharge_disposition': "Discharged to home",
            'age_group': "46-65"
        }
        prob = predict_readmission_risk(sample_inputs)
        if prob:
            st.success(f"Test Result: {prob:.1%}")
    
    if st.button("📤 Export Report", use_container_width=True):
        st.info("Report export feature coming soon")
    
    st.markdown("---")
    
    # Support Information
    st.markdown("### 📞 Support")
    st.markdown("""
    <div style="font-size: 0.9rem; color: #64748b;">
        <p><strong>Clinical Support:</strong><br>
        Department of Clinical Informatics<br>
        📧 clinical.support@kenyahospital.ke</p>
        
        <p><strong>Technical Support:</strong><br>
        IT Department<br>
        📧 it.support@kenyahospital.ke</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FOOTER - Professional Medical Footer
# ============================================================================
st.markdown("---")
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: space-between; align-items: center; padding: 0 1rem;">
        <div style="text-align: left;">
            <strong style="color: #334155;">Kenya Hospital System</strong><br>
            <span style="color: #64748b; font-size: 0.9rem;">Clinical Decision Support v1.0</span>
        </div>
        <div style="text-align: center;">
            <span style="color: #64748b; font-size: 0.9rem;">
                © 2024 Ministry of Health, Kenya. For clinical use only.
            </span>
        </div>
        <div style="text-align: right;">
            <span style="color: #64748b; font-size: 0.9rem;">
                Last Updated: """ + datetime.now().strftime("%Y-%m-%d") + """
            </span>
        </div>
    </div>
    <div style="margin-top: 0.5rem; padding-top: 0.5rem; border-top: 1px solid #e2e8f0;">
        <span style="color: #94a3b8; font-size: 0.8rem;">
            This tool supports clinical decision making but does not replace professional judgment.
            Always verify predictions with clinical assessment.
        </span>
    </div>
</div>
""", unsafe_allow_html=True)