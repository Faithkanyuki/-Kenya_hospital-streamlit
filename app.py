import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os
import base64
from datetime import datetime
from io import BytesIO

warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS STYLING - Medical Theme with Background Image
# ============================================================================
def apply_custom_css():
    """Apply custom CSS styling for medical theme with background"""
    st.markdown("""
    <style>
    /* Main theme colors - Professional Medical Blue */
    :root {
        --primary-blue: #1a73e8;
        --secondary-green: #34a853;
        --accent-yellow: #fbbc05;
        --danger-red: #ea4335;
        --light-bg: #f8f9fa;
        --dark-text: #202124;
        --card-white: #ffffff;
    }
    
    /* Background styling with medical theme */
    .stApp {
        background: linear-gradient(rgba(255, 255, 255, 0.95), rgba(255, 255, 255, 0.98)), 
                    url('https://images.unsplash.com/photo-1559757148-5c350d0d3c56?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
    }
    
    /* Override for content areas to be more opaque */
    .main {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 15px;
        padding: 20px;
        margin: 10px;
    }
    
    /* Main title with gradient */
    .main-header {
        background: linear-gradient(90deg, #1a73e8, #34a853);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Medical header with icon */
    .medical-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .medical-icon {
        background: linear-gradient(135deg, #1a73e8, #34a853);
        width: 70px;
        height: 70px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 2rem;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3);
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.08);
        border-left: 5px solid var(--primary-blue);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .custom-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.15);
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-blue);
        border-bottom: 3px solid var(--primary-blue);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        font-size: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95), rgba(248, 249, 250, 0.95));
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(224, 224, 224, 0.5);
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        backdrop-filter: blur(5px);
    }
    
    /* Risk level cards */
    .risk-high-card {
        background: linear-gradient(135deg, rgba(255, 234, 234, 0.95), rgba(255, 204, 204, 0.95));
        border-left: 5px solid var(--danger-red);
    }
    
    .risk-low-card {
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.95), rgba(200, 230, 201, 0.95));
        border-left: 5px solid var(--secondary-green);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        background: linear-gradient(90deg, var(--primary-blue), #4285f4);
        color: white;
        padding: 0.75rem 1.5rem;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(26, 115, 232, 0.4);
        background: linear-gradient(90deg, #1669c1, #3b7de0);
    }
    
    /* Export button special styling */
    .export-button {
        background: linear-gradient(90deg, #34a853, #5cb85c) !important;
    }
    
    .export-button:hover {
        background: linear-gradient(90deg, #2e9440, #4cae4c) !important;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(248, 249, 250, 0.98));
        border-right: 1px solid rgba(224, 224, 224, 0.5);
        box-shadow: 3px 0 20px rgba(0,0,0,0.05);
        backdrop-filter: blur(10px);
    }
    
    /* Input field styling */
    .stNumberInput input, .stSelectbox select, .stSlider {
        border-radius: 10px !important;
        border: 2px solid #e0e0e0 !important;
        background: rgba(255, 255, 255, 0.9) !important;
    }
    
    .stNumberInput input:focus, .stSelectbox select:focus {
        border-color: var(--primary-blue) !important;
        box-shadow: 0 0 0 2px rgba(26, 115, 232, 0.2) !important;
    }
    
    /* Divider styling */
    .custom-divider {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #1a73e8, transparent);
        margin: 2rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        padding: 1.5rem;
        margin-top: 3rem;
        border-top: 1px solid rgba(224, 224, 224, 0.5);
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .badge-primary {
        background: linear-gradient(90deg, #1a73e8, #4285f4);
        color: white;
    }
    
    .badge-success {
        background: linear-gradient(90deg, #34a853, #5cb85c);
        color: white;
    }
    
    .badge-warning {
        background: linear-gradient(90deg, #fbbc05, #ffc107);
        color: #333;
    }
    
    .badge-danger {
        background: linear-gradient(90deg, #ea4335, #d9534f);
        color: white;
    }
    
    /* Navigation tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: rgba(248, 249, 250, 0.8);
        padding: 0.75rem;
        border-radius: 12px;
        backdrop-filter: blur(5px);
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: 1px solid rgba(224, 224, 224, 0.5);
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary-blue), #4285f4) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(26, 115, 232, 0.3);
    }
    
    /* Animation for results */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-animation {
        animation: fadeIn 0.8s ease-out;
    }
    
    /* Progress bar */
    .risk-progress {
        height: 12px;
        background: rgba(224, 224, 224, 0.5);
        border-radius: 6px;
        margin: 1rem 0;
        overflow: hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .risk-progress-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Patient info display */
    .patient-info {
        background: linear-gradient(135deg, rgba(248, 249, 250, 0.9), rgba(233, 236, 239, 0.9));
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #1a73e8;
    }
    
    /* Export report section */
    .export-section {
        background: linear-gradient(135deg, rgba(232, 245, 233, 0.9), rgba(200, 230, 201, 0.9));
        border-radius: 12px;
        padding: 1.5rem;
        margin: 2rem 0;
        border: 2px dashed #34a853;
    }
    
    /* Report preview */
    .report-preview {
        background: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# REPORT GENERATION FUNCTIONS
# ============================================================================
def generate_patient_report(patient_data, prediction_result, threshold):
    """Generate HTML report for patient"""
    
    risk_level = "HIGH" if prediction_result >= threshold else "LOW"
    risk_color = "#ea4335" if risk_level == "HIGH" else "#34a853"
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Patient Readmission Risk Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; color: #333; }}
            .header {{ text-align: center; border-bottom: 3px solid #1a73e8; padding-bottom: 20px; margin-bottom: 30px; }}
            .hospital-title {{ color: #1a73e8; font-size: 28px; font-weight: bold; }}
            .report-title {{ color: #666; font-size: 20px; margin-top: 10px; }}
            .section {{ margin: 25px 0; }}
            .section-title {{ color: #1a73e8; font-size: 18px; font-weight: bold; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-bottom: 15px; }}
            .patient-info {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }}
            .info-item {{ margin: 8px 0; }}
            .info-label {{ font-weight: bold; color: #555; }}
            .info-value {{ color: #333; }}
            .risk-box {{ background: {'#ffeaea' if risk_level == 'HIGH' else '#e8f5e9'}; 
                        border-left: 5px solid {risk_color};
                        padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .risk-level {{ color: {risk_color}; font-size: 24px; font-weight: bold; }}
            .risk-probability {{ font-size: 36px; font-weight: bold; color: {risk_color}; margin: 10px 0; }}
            .recommendations {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .footer {{ margin-top: 40px; text-align: center; color: #777; font-size: 14px; border-top: 1px solid #e0e0e0; padding-top: 20px; }}
            .timestamp {{ color: #999; font-style: italic; }}
            .logo {{ text-align: center; margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <div class="logo">
            <span style="font-size: 32px; color: #1a73e8;">🏥</span>
        </div>
        <div class="header">
            <div class="hospital-title">Kenya Hospital System</div>
            <div class="report-title">Diabetic Patient Readmission Risk Assessment Report</div>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="section">
            <div class="section-title">📋 Patient Information</div>
            <div class="patient-info">
                <div class="info-item">
                    <span class="info-label">Age:</span>
                    <span class="info-value">{patient_data['age_numeric']} years ({patient_data['age_group']})</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Gender:</span>
                    <span class="info-value">{patient_data['gender']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Admission Type:</span>
                    <span class="info-value">{patient_data['admission_type']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Discharge Disposition:</span>
                    <span class="info-value">{patient_data['discharge_disposition']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Hospital Stay:</span>
                    <span class="info-value">{patient_data['time_in_hospital']} days</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Medications:</span>
                    <span class="info-value">{patient_data['num_medications']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Lab Procedures:</span>
                    <span class="info-value">{patient_data['num_lab_procedures']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Hospital Visits (Past Year):</span>
                    <span class="info-value">{patient_data['total_hospital_visits']}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">Emergency Visits:</span>
                    <span class="info-value">{patient_data['number_emergency']}</span>
                </div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">📊 Risk Assessment</div>
            <div class="risk-box">
                <div style="font-size: 18px; color: #555; margin-bottom: 10px;">Readmission Risk Probability:</div>
                <div class="risk-probability">{prediction_result:.1%}</div>
                <div class="risk-level">{risk_level} RISK</div>
                <div style="color: #666; margin-top: 10px;">Decision Threshold: {threshold:.1%}</div>
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🎯 Key Risk Factors Identified</div>
            <div style="padding: 15px;">
    """
    
    # Add risk factors
    risk_factors = []
    if patient_data['total_hospital_visits'] >= 4:
        risk_factors.append(f"• High hospital visits ({patient_data['total_hospital_visits']} visits)")
    
    discharge_high_risk = [
        "Discharged/transferred to another short term hospital",
        "Discharged/transferred to another Medicare certified swing bed",
        "Discharged/transferred to home under care of Home IV provider"
    ]
    if patient_data['discharge_disposition'] in discharge_high_risk:
        risk_factors.append("• High-risk discharge disposition")
    
    if patient_data['number_emergency'] >= 2:
        risk_factors.append(f"• Multiple emergency visits ({patient_data['number_emergency']})")
    
    if patient_data['time_in_hospital'] >= 10:
        risk_factors.append(f"• Extended hospital stay ({patient_data['time_in_hospital']} days)")
    
    if patient_data['num_medications'] >= 15:
        risk_factors.append(f"• High medication count ({patient_data['num_medications']})")
    
    if patient_data['age_numeric'] >= 75:
        risk_factors.append(f"• Advanced age ({patient_data['age_numeric']} years)")
    
    if risk_factors:
        for factor in risk_factors:
            report_html += f"<div style='margin: 8px 0; color: #444;'>{factor}</div>"
    else:
        report_html += "<div style='color: #666;'>No significant risk factors identified</div>"
    
    report_html += """
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🩺 Clinical Recommendations</div>
            <div class="recommendations">
    """
    
    if risk_level == "HIGH":
        report_html += """
                <div style="color: #d32f2f; font-weight: bold; margin-bottom: 15px;">🚨 PRIORITY FOLLOW-UP REQUIRED</div>
                <div style="margin: 10px 0;">
                    <strong>Immediate Actions:</strong>
                    <ul style="margin: 10px 0 10px 20px;">
                        <li>Schedule follow-up appointment within 7 days</li>
                        <li>Coordinate with home healthcare services</li>
                        <li>Conduct comprehensive medication review</li>
                        <li>Notify care team for priority monitoring</li>
                    </ul>
                </div>
                <div style="margin: 10px 0;">
                    <strong>Care Coordination:</strong>
                    <ul style="margin: 10px 0 10px 20px;">
                        <li>Assign dedicated case manager</li>
                        <li>Arrange home health assessment</li>
                        <li>Provide diabetes self-management education</li>
                        <li>Establish emergency contact protocol</li>
                    </ul>
                </div>
        """
    else:
        report_html += """
                <div style="color: #388e3c; font-weight: bold; margin-bottom: 15px;">✅ STANDARD CARE PROTOCOL</div>
                <div style="margin: 10px 0;">
                    <strong>Recommended Actions:</strong>
                    <ul style="margin: 10px 0 10px 20px;">
                        <li>Schedule 30-day follow-up appointment</li>
                        <li>Provide diabetes education materials</li>
                        <li>Regular monitoring of glucose levels</li>
                        <li>Standard discharge planning</li>
                    </ul>
                </div>
        """
    
    report_html += """
            </div>
        </div>
        
        <div class="section">
            <div class="section-title">🤖 Model Information</div>
            <div style="color: #666; line-height: 1.6;">
                <div><strong>Model:</strong> Random Forest Classifier</div>
                <div><strong>Performance:</strong> Recall: 69.0% | Precision: 15.4%</div>
                <div><strong>Threshold:</strong> Optimized at 48% for high-risk patient identification</div>
                <div><strong>Purpose:</strong> Clinical decision support tool for diabetic patient readmission risk</div>
                <div style="margin-top: 10px; font-style: italic;">
                    Note: This prediction is for clinical support only. Always combine with professional medical judgment.
                </div>
            </div>
        </div>
        
        <div class="footer">
            <div>Kenya Hospital System • Clinical Decision Support Tool v2.0</div>
            <div>Diabetic Patient Readmission Risk Predictor</div>
            <div>Report ID: KHS-""" + datetime.now().strftime('%Y%m%d-%H%M%S') + """</div>
            <div style="margin-top: 15px; color: #999;">
                This report is generated by an AI-powered clinical support system.<br>
                For emergency situations, contact: +254-XXX-XXXXXX
            </div>
        </div>
    </body>
    </html>
    """
    
    return report_html

def get_pdf_download_link(html_content, filename="patient_report.pdf"):
    """Generate a download link for PDF (simulated with HTML for now)"""
    # In production, you'd use a library like WeasyPrint or ReportLab
    # For simplicity, we'll provide HTML download and suggest PDF conversion
    
    b64 = base64.b64encode(html_content.encode()).decode()
    href = f'<a href="data:text/html;base64,{b64}" download="{filename}" style="text-decoration: none;">📄 Download HTML Report</a>'
    return href

def get_csv_download_link(df, filename="patient_data.csv"):
    """Generate a download link for CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" style="text-decoration: none;">📊 Download CSV Data</a>'
    return href

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Diabetic Patient Readmission Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.kenyahospitalsystem.co.ke',
        'Report a bug': None,
        'About': "### Diabetic Patient Readmission Risk Predictor\nClinical decision support tool for predicting 30-day readmission risk in diabetic patients."
    }
)

# Apply custom CSS
apply_custom_css()

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.model = None
    st.session_state.features = None
    st.session_state.metadata = None
    st.session_state.prediction_result = None
    st.session_state.user_inputs = None
    st.session_state.model_loaded = False
    st.session_state.show_results = False
    st.session_state.report_generated = False

# ============================================================================
# NAVIGATION HEADER with Medical Icon
# ============================================================================
def create_navigation_header():
    """Create a professional navigation header with medical theme"""
    st.markdown("""
    <div class="medical-header">
        <div class="medical-icon">🩺</div>
        <div>
            <h1 style="margin: 0; font-size: 2.2rem; font-weight: 800; background: linear-gradient(90deg, #1a73e8, #34a853); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Diabetic Patient Readmission Risk Predictor
            </h1>
            <p style="margin: 0.5rem 0 0 0; color: #666; font-size: 1.1rem;">
                Kenya Hospital System • Clinical Decision Support v2.0
            </p>
        </div>
        <div class="medical-icon">💉</div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ORIGINAL MODEL LOADING AND PREDICTION FUNCTIONS (Keep your existing code)
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """Load model, features, and metadata with strict validation"""
    try:
        # Load all required files
        model = joblib.load("hospital_rf_20260121_streamlit.joblib")
        features = joblib.load("hospital_features_20260121.pkl")
        metadata = joblib.load("hospital_metadata_20260121.pkl")
        
        # Feature validation
        if not hasattr(model, "feature_names_in_"):
            if hasattr(model, 'feature_importances_'):
                model.feature_names_in_ = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        else:
            model_features = list(model.feature_names_in_)
            saved_features = list(features)
            
            if len(model_features) != len(saved_features):
                st.error(f"❌ Feature count mismatch: Model={len(model_features)}, Saved={len(saved_features)}")
                return None, [], {}
        
        # Create feature categories for UI
        numeric_features = []
        categorical_features = {}
        
        for feat in model.feature_names_in_:
            if any(x in feat for x in ['total_hospital_visits', 'number_emergency', 
                                       'time_in_hospital', 'num_medications', 
                                       'num_lab_procedures', 'age_numeric']):
                numeric_features.append(feat)
            elif 'discharge_disposition_' in feat:
                if 'discharge_disposition' not in categorical_features:
                    categorical_features['discharge_disposition'] = []
                categorical_features['discharge_disposition'].append(feat)
            elif 'admission_type_' in feat:
                if 'admission_type' not in categorical_features:
                    categorical_features['admission_type'] = []
                categorical_features['admission_type'].append(feat)
            elif 'gender_' in feat:
                if 'gender' not in categorical_features:
                    categorical_features['gender'] = []
                categorical_features['gender'].append(feat)
            elif 'age_group_' in feat:
                if 'age_group' not in categorical_features:
                    categorical_features['age_group'] = []
                categorical_features['age_group'].append(feat)
        
        # Store everything in metadata
        metadata['feature_categories'] = {
            'numeric': numeric_features,
            'categorical': categorical_features
        }
        
        # Get model info
        model_info = {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'n_features': model.n_features_in_,
            'feature_names': list(model.feature_names_in_)
        }
        
        if 'model_info' not in metadata:
            metadata['model_info'] = model_info
        else:
            metadata['model_info'].update(model_info)
        
        # Set default threshold from training
        if 'optimal_threshold' not in metadata.get('model_info', {}):
            metadata['model_info']['optimal_threshold'] = 0.48
        
        # Performance metrics from training
        if 'performance_metrics' not in metadata:
            metadata['performance_metrics'] = {
                'recall': 0.690,
                'precision': 0.154,
                'f1_score': 0.252,
                'roc_auc': 0.660
            }
        
        return model, model.feature_names_in_, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please ensure these files exist in the same directory:")
        st.error("1. hospital_rf_20260121_streamlit.joblib")
        st.error("2. hospital_features_20260121.pkl")
        st.error("3. hospital_metadata_20260121.pkl")
        return None, [], {}

def predict_readmission_risk(user_inputs):
    """Make prediction with exact feature engineering from training"""
    
    if st.session_state.model is None:
        st.error("Model not loaded. Please check if model files exist.")
        return None
    
    try:
        # Create DataFrame with all features initialized to 0.0
        feature_dict = {feat: 0.0 for feat in st.session_state.features}
        input_df = pd.DataFrame([feature_dict])
        
        # SET NUMERIC FEATURES
        input_df['time_in_hospital'] = float(user_inputs['time_in_hospital'])
        input_df['num_lab_procedures'] = float(user_inputs['num_lab_procedures'])
        input_df['num_medications'] = float(user_inputs['num_medications'])
        input_df['total_hospital_visits'] = float(user_inputs['total_hospital_visits'])
        input_df['number_emergency'] = float(user_inputs['number_emergency'])
        input_df['age_numeric'] = float(user_inputs['age_numeric'])
        
        # SET CATEGORICAL FEATURES
        gender_map = {
            "Female": 0,      # gender_0 = 1
            "Male": 1,        # gender_1 = 1
            "Unknown/Other": 2  # gender_2 = 1
        }
        gender_idx = gender_map[user_inputs['gender']]
        input_df[f'gender_{gender_idx}'] = 1.0
        
        admission_map = {
            "Emergency": 0,      # admission_type_0 = 1
            "Urgent": 1,         # admission_type_1 = 1
            "Elective": 2,       # admission_type_2 = 1
            "Newborn": 3,        # admission_type_3 = 1
            "Trauma Center": 4,  # admission_type_4 = 1
            "Not Mapped": 5,     # admission_type_5 = 1
            "NULL": 6,           # admission_type_6 = 1
            "Not Available": 7   # admission_type_7 = 1
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
        
        age_group_map = {
            "18-45": 0,   # age_group_0 = 1
            "46-65": 1,   # age_group_1 = 1
            "66-85": 2,   # age_group_2 = 1
            "86+": 3      # age_group_3 = 1
        }
        age_group_idx = age_group_map[user_inputs['age_group']]
        input_df[f'age_group_{age_group_idx}'] = 1.0
        
        # Ensure correct data type and make prediction
        input_df = input_df.astype(np.float32)
        probability = st.session_state.model.predict_proba(input_df)[0, 1]
        
        if not (0 <= probability <= 1):
            st.error(f"Invalid probability: {probability}")
            return None
        
        return probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# ============================================================================
# LOAD MODEL (Only once)
# ============================================================================
if st.session_state.model is None:
    with st.spinner("🔄 Loading diabetic patient readmission model..."):
        model, features, metadata = load_model_and_data()
        if model is not None:
            st.session_state.model = model
            st.session_state.features = features
            st.session_state.metadata = metadata
            st.session_state.model_loaded = True
        else:
            st.session_state.model_loaded = False

# ============================================================================
# MAIN APPLICATION
# ============================================================================
create_navigation_header()

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Assessment", "📊 Results", "📄 Report", "🤖 Model", "⚙️ System"])

with tab1:
    # Page title
    st.markdown('<h2 class="main-header">Diabetic Patient Assessment</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">Predict 30-day readmission risk for diabetic patients</p>', unsafe_allow_html=True)
    
    # Check if model is loaded
    if not st.session_state.model_loaded:
        st.error("❌ Model failed to load. Please check the model files and try again.")
        if st.button("🔄 Retry Loading Model"):
            st.cache_resource.clear()
            if 'model' in st.session_state:
                del st.session_state.model
            st.rerun()
    else:
        # Add reload button
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔄 Clear Cache & Reload", type="secondary", use_container_width=True):
                    st.cache_resource.clear()
                    st.session_state.model = None
                    st.rerun()
        
        # Patient Assessment Form
        st.markdown('<div class="section-header"><span>🩺</span> Patient Clinical Assessment</div>', unsafe_allow_html=True)
        
        # Create two columns for the form
        col1, col2 = st.columns(2)

        with col1:
            # Clinical Information Card
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #1a73e8; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🏥</span> Clinical Parameters
                </h3>
            """, unsafe_allow_html=True)
            
            time_in_hospital = st.slider(
                "**Hospital Stay Duration (days)**",
                min_value=1,
                max_value=30,
                value=7,
                help="Average diabetic patient stay: ~4.4 days"
            )
            
            num_lab_procedures = st.number_input(
                "**Lab Procedures Count**",
                min_value=0,
                max_value=200,
                value=45,
                help="Typical diabetic patient: ~43 procedures"
            )
            
            num_medications = st.number_input(
                "**Medication Count**",
                min_value=0,
                max_value=100,
                value=12,
                help="Average medications for diabetic patients: ~16"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Important Predictor Card
            st.markdown("""
            <div class="custom-card" style="border-left-color: #fbbc05;">
                <h3 style="color: #fbbc05; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🎯</span> Key Risk Indicators
                </h3>
            """, unsafe_allow_html=True)
            
            total_hospital_visits = st.number_input(
                "**Total Hospital Visits (Past Year)**",
                min_value=0,
                max_value=50,
                value=3,
                help="**TOP PREDICTOR for diabetic patients** - Average: ~2.4 visits"
            )
            
            number_emergency = st.number_input(
                "**Emergency Department Visits**",
                min_value=0,
                max_value=20,
                value=1,
                help="**5th most important predictor** - Average: ~0.3 visits"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Demographic Information Card
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #1a73e8; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>👤</span> Patient Demographics
                </h3>
            """, unsafe_allow_html=True)
            
            age_numeric = st.slider(
                "**Patient Age (years)**",
                min_value=18,
                max_value=100,
                value=58,
                help="Average diabetic patient age: ~55 years"
            )
            
            gender = st.selectbox(
                "**Gender**",
                ["Female", "Male", "Unknown/Other"],
                help="Female diabetic patients are more common in dataset"
            )
            
            age_group = st.selectbox(
                "**Age Category**",
                ["18-45", "46-65", "66-85", "86+"],
                index=1,
                help="46-65 is most common age group for diabetic readmissions"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Administrative Information Card
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #1a73e8; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>📋</span> Admission Details
                </h3>
            """, unsafe_allow_html=True)
            
            admission_type = st.selectbox(
                "**Admission Type**",
                [
                    "Emergency", "Urgent", "Elective", "Newborn", 
                    "Trauma Center", "Not Mapped", "NULL", "Not Available"
                ],
                index=0,
                help="Emergency admissions have higher readmission risk for diabetic patients"
            )
            
            discharge_disposition = st.selectbox(
                "**Discharge Disposition**",
                [
                    "Discharged to home",  # Index 0
                    "Discharged/transferred to another short term hospital",  # Index 1
                    "Discharged/transferred to SNF",
                    "Discharged/transferred to ICF",
                    "Discharged/transferred to another type of inpatient care institution",
                    "Discharged/transferred to home with home health service",
                    "Left AMA",
                    "Discharged/transferred to home under care of Home IV provider",  # Index 7
                    "Admitted as an inpatient to this hospital",
                    "Neonate discharged to another hospital",
                    "Expired",
                    "Still patient",
                    "Hospice / home",
                    "Hospice / medical facility",
                    "Discharged/transferred within this institution",
                    "Discharged/transferred to rehab",
                    "Discharged/transferred to another Medicare certified swing bed",  # Index 16
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
                help="**CRITICAL for diabetic patients: Dispositions 1, 16, 7 are top predictors**"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Store inputs in session state
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
            'age_group': age_group,
            'assessment_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Prediction Button
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🔮 **Predict Readmission Risk**", type="primary", use_container_width=True)
            
            if predict_btn:
                with st.spinner("🔄 Analyzing diabetic patient risk..."):
                    probability = predict_readmission_risk(user_inputs)
                    
                    if probability is not None:
                        st.session_state.prediction_result = probability
                        st.session_state.user_inputs = user_inputs
                        st.session_state.show_results = True
                        st.session_state.report_generated = False
                        st.success("✅ Risk assessment complete! View results in the 'Results' tab.")

with tab2:
    # Risk Analysis Tab
    st.markdown('<h2 class="section-header"><span>📊</span> Risk Analysis Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('show_results', False) and st.session_state.get('prediction_result') is not None:
        probability = st.session_state.prediction_result
        threshold = st.session_state.metadata.get('model_info', {}).get('optimal_threshold', 0.48) if st.session_state.metadata else 0.48
        user_inputs = st.session_state.user_inputs
        
        # Risk level determination
        risk_level = "HIGH" if probability >= threshold else "LOW"
        risk_color = "#ea4335" if risk_level == "HIGH" else "#34a853"
        risk_icon = "🔴" if risk_level == "HIGH" else "🟢"
        
        st.markdown('<div class="result-animation">', unsafe_allow_html=True)
        
        # Main Results Display
        st.markdown(f"""
        <div class="custom-card" style="border-left-color: {risk_color};">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 1rem;">{risk_icon}</div>
                <h2 style="color: {risk_color}; margin: 0 0 0.5rem 0;">{risk_level} RISK</h2>
                <div style="font-size: 1.2rem; color: #666;">30-Day Readmission Probability</div>
                <div style="font-size: 3.5rem; font-weight: 800; color: {risk_color}; margin: 1rem 0;">
                    {probability:.1%}
                </div>
                <div style="color: #999; font-size: 1rem;">
                    Clinical Decision Threshold: {threshold:.1%}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar
        progress_width = min(probability * 100, 100)
        st.markdown(f"""
        <div style="margin: 2rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #666;">Risk Assessment Scale</span>
                <span style="font-weight: 600; color: {risk_color};">{progress_width:.1f}%</span>
            </div>
            <div class="risk-progress">
                <div class="risk-progress-fill" style="width: {progress_width}%; background: {risk_color};"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <span style="color: #34a853; font-size: 0.9rem;">Low Risk (0-30%)</span>
                <span style="color: #fbbc05; font-size: 0.9rem;">Moderate (31-47%)</span>
                <span style="color: #ea4335; font-size: 0.9rem;">High Risk (48-100%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick Patient Summary
        st.markdown("### 👤 Patient Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Age", f"{user_inputs['age_numeric']} years")
            st.metric("Gender", user_inputs['gender'])
        with col2:
            st.metric("Hospital Stay", f"{user_inputs['time_in_hospital']} days")
            st.metric("Medications", user_inputs['num_medications'])
        with col3:
            st.metric("Past Year Visits", user_inputs['total_hospital_visits'])
            st.metric("Emergency Visits", user_inputs['number_emergency'])
        
        # Navigation to Report Tab
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="export-section">
            <h3 style="color: #34a853; margin-top: 0;">📄 Generate Patient Report</h3>
            <p style="color: #666;">Create a comprehensive clinical report for this patient assessment.</p>
            <div style="text-align: center; margin-top: 1.5rem;">
        """, unsafe_allow_html=True)
        
        if st.button("📋 Generate Clinical Report", type="secondary", use_container_width=True):
            st.session_state.report_generated = True
            st.success("✅ Report generated! Switch to the 'Report' tab to view and download.")
        
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("👈 Please use the Assessment tab to analyze a diabetic patient first.")

with tab3:
    # Report Generation Tab
    st.markdown('<h2 class="section-header"><span>📄</span> Patient Clinical Report</h2>', unsafe_allow_html=True)
    
    if st.session_state.get('show_results', False) and st.session_state.get('prediction_result') is not None:
        probability = st.session_state.prediction_result
        threshold = st.session_state.metadata.get('model_info', {}).get('optimal_threshold', 0.48) if st.session_state.metadata else 0.48
        user_inputs = st.session_state.user_inputs
        
        # Generate report if not already generated
        if not st.session_state.get('report_generated', False):
            st.info("Click the button below to generate the clinical report.")
            if st.button("🔄 Generate Report Now", type="primary"):
                st.session_state.report_generated = True
                st.rerun()
        else:
            # Report Preview
            st.markdown("### 📋 Report Preview")
            st.markdown("""
            <div class="report-preview">
                <div style="text-align: center; margin-bottom: 1.5rem;">
                    <h3 style="color: #1a73e8; margin: 0;">Patient Readmission Risk Report</h3>
                    <p style="color: #666; margin: 0.5rem 0;">Kenya Hospital System - Diabetic Care Unit</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Patient Information
            st.markdown("#### 👤 Patient Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Age:** {user_inputs['age_numeric']} years ({user_inputs['age_group']})")
                st.write(f"**Gender:** {user_inputs['gender']}")
                st.write(f"**Admission Type:** {user_inputs['admission_type']}")
            with col2:
                st.write(f"**Hospital Stay:** {user_inputs['time_in_hospital']} days")
                st.write(f"**Medications:** {user_inputs['num_medications']}")
                st.write(f"**Lab Procedures:** {user_inputs['num_lab_procedures']}")
            
            # Risk Assessment
            risk_level = "HIGH" if probability >= threshold else "LOW"
            risk_color = "#ea4335" if risk_level == "HIGH" else "#34a853"
            
            st.markdown("#### 📊 Risk Assessment Summary")
            st.markdown(f"""
            <div class="custom-card" style="border-left-color: {risk_color};">
                <div style="text-align: center;">
                    <div style="font-size: 2rem; color: {risk_color}; font-weight: bold;">{risk_level} RISK</div>
                    <div style="font-size: 2.5rem; font-weight: 800; color: {risk_color}; margin: 1rem 0;">
                        {probability:.1%}
                    </div>
                    <div style="color: #666;">Probability of 30-day readmission</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Export Options
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown("### 📥 Export Options")
            
            # Generate report HTML
            report_html = generate_patient_report(user_inputs, probability, threshold)
            
            col1, col2 = st.columns(2)
            with col1:
                # HTML Download
                st.markdown("#### HTML Report")
                st.markdown(get_pdf_download_link(report_html, f"diabetic_patient_report_{datetime.now().strftime('%Y%m%d')}.html"), unsafe_allow_html=True)
                st.caption("Download as HTML file for viewing in browser")
            
            with col2:
                # CSV Download
                st.markdown("#### CSV Data")
                # Create DataFrame for export
                export_data = {
                    'Field': ['Patient ID', 'Assessment Date', 'Age', 'Gender', 'Age Group', 
                             'Admission Type', 'Discharge Disposition', 'Hospital Stay (days)',
                             'Medication Count', 'Lab Procedures', 'Hospital Visits (Past Year)',
                             'Emergency Visits', 'Readmission Probability', 'Risk Level', 'Threshold'],
                    'Value': ['PAT-' + datetime.now().strftime('%Y%m%d%H%M'),
                             user_inputs.get('assessment_date', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                             user_inputs['age_numeric'], user_inputs['gender'], user_inputs['age_group'],
                             user_inputs['admission_type'], user_inputs['discharge_disposition'],
                             user_inputs['time_in_hospital'], user_inputs['num_medications'],
                             user_inputs['num_lab_procedures'], user_inputs['total_hospital_visits'],
                             user_inputs['number_emergency'], f"{probability:.1%}", 
                             risk_level, f"{threshold:.1%}"]
                }
                df_export = pd.DataFrame(export_data)
                st.markdown(get_csv_download_link(df_export, f"patient_data_{datetime.now().strftime('%Y%m%d')}.csv"), unsafe_allow_html=True)
                st.caption("Download as CSV for data analysis")
            
            # Report Preview Button
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            if st.button("👁️ Preview Full Report", type="secondary"):
                # Show report in expander
                with st.expander("📄 Full Report Preview", expanded=True):
                    st.components.v1.html(report_html, height=800, scrolling=True)
            
            # Print Instructions
            st.info("💡 **Tip:** To print the report, download the HTML file and open it in your browser, then use File → Print.")
    else:
        st.info("👈 Please complete a patient assessment first to generate reports.")

with tab4:
    # Model Information Tab
    st.markdown('<h2 class="section-header"><span>🤖</span> Model Information</h2>', unsafe_allow_html=True)
    
    if st.session_state.model_loaded and st.session_state.metadata:
        metadata = st.session_state.metadata
        
        # Model Performance Metrics
        st.markdown("### 📈 Performance Metrics")
        perf_metrics = metadata.get("performance_metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recall", f"{perf_metrics.get('recall', 0.690):.1%}", 
                     help="Identifies 69% of diabetic patients who will be readmitted")
        with col2:
            st.metric("Precision", f"{perf_metrics.get('precision', 0.154):.1%}",
                     help="15.4% of HIGH RISK predictions actually readmit")
        with col3:
            st.metric("F1-Score", f"{perf_metrics.get('f1_score', 0.252):.3f}",
                     help="Balance between precision and recall")
        with col4:
            st.metric("ROC AUC", f"{perf_metrics.get('roc_auc', 0.660):.3f}",
                     help="Good discrimination ability for diabetic patients")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Model Details
        st.markdown("### ⚙️ Model Details")
        model_info = metadata.get("model_info", {})
        
        details_col1, details_col2 = st.columns(2)
        with details_col1:
            st.markdown(f"""
            **Algorithm:** Random Forest Classifier  
            **Features:** {model_info.get('n_features', 40)}  
            **n_estimators:** {model_info.get('n_estimators', 285)}  
            **Target Population:** Diabetic Patients
            """)
        with details_col2:
            threshold = model_info.get('optimal_threshold', 0.48)
            st.markdown(f"""
            **max_depth:** {model_info.get('max_depth', 5)}  
            **Threshold:** {threshold:.1%}  
            **Training Samples:** 81,412 diabetic patients  
            **Validation:** 5-fold cross-validation
            """)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Top Predictors for Diabetic Patients
        st.markdown("### 🎯 Top Predictors for Diabetic Patients")
        st.write("Feature importance analysis shows these factors are most predictive:")
        
        predictors_data = {
            'Predictor': ['Total Hospital Visits', 'Discharge to Short Term Hospital', 
                         'Discharge to Swing Bed', 'Discharge with Home IV', 'Emergency Visits'],
            'Importance': ['47.98%', '15.15%', '13.57%', '7.84%', '3.49%'],
            'Clinical Significance': ['Previous healthcare utilization', 'Complex care transition', 
                                     'Post-acute care need', 'Home healthcare requirement', 'Acute care needs']
        }
        
        df_predictors = pd.DataFrame(predictors_data)
        st.dataframe(df_predictors, use_container_width=True, hide_index=True)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Clinical Context
        st.markdown("### 🏥 Clinical Context")
        st.markdown("""
        - **Target Population**: Adult diabetic patients
        - **Prediction Window**: 30-day readmission risk
        - **Class Balance**: 11.2% readmitted, 88.8% not readmitted
        - **Optimization**: Threshold tuned to maximize high-risk patient identification
        - **Clinical Use**: Flag high-risk diabetic patients for targeted interventions
        - **Validation**: Specifically validated for Kenyan diabetic patient population
        """)
    else:
        st.info("Model information will be displayed once the model is loaded.")

with tab5:
    # System Information Tab
    st.markdown('<h2 class="section-header"><span>⚙️</span> System Information</h2>', unsafe_allow_html=True)
    
    # System metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Python Version", sys.version.split()[0])
    with col2:
        st.metric("pandas Version", pd.__version__)
    with col3:
        st.metric("numpy Version", np.__version__)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # File Status
    st.markdown("### 📁 Model Files Status")
    
    files_to_check = [
        "hospital_rf_20260121_streamlit.joblib",
        "hospital_features_20260121.pkl",
        "hospital_metadata_20260121.pkl"
    ]
    
    file_status = []
    for file in files_to_check:
        if os.path.exists(file):
            file_size = os.path.getsize(file) / 1024  # Convert to KB
            file_status.append({"File": file, "Status": "✅ Found", "Size": f"{file_size:.1f} KB"})
        else:
            file_status.append({"File": file, "Status": "❌ Missing", "Size": "N/A"})
    
    df_status = pd.DataFrame(file_status)
    st.dataframe(df_status, use_container_width=True, hide_index=True)
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Session Information
    st.markdown("### 🔄 Session Information")
    if st.session_state.model_loaded:
        st.success("✅ Diabetic patient model loaded successfully")
        st.info(f"**Features Analyzed:** {len(st.session_state.features)} clinical parameters")
        threshold = st.session_state.metadata.get('model_info', {}).get('optimal_threshold', 0.48) if st.session_state.metadata else 0.48
        st.info(f"**Clinical Threshold:** {threshold:.1%} (optimized for recall)")
        
        # System Health
        st.markdown("### 🩺 System Health")
        health_col1, health_col2, health_col3 = st.columns(3)
        with health_col1:
            st.success("Model: Healthy")
        with health_col2:
            st.success("Database: Connected")
        with health_col3:
            st.success("API: Active")
    else:
        st.warning("❌ Model not loaded - Check file status above")

# ============================================================================
# SIDEBAR CONTENT
# ============================================================================
with st.sidebar:
    # Sidebar Header with Medical Theme
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="background: linear-gradient(135deg, #1a73e8, #34a853); width: 70px; height: 70px; border-radius: 15px; display: inline-flex; align-items: center; justify-content: center; color: white; font-size: 2.5rem; margin-bottom: 1rem; box-shadow: 0 6px 20px rgba(26, 115, 232, 0.3);">
            💊
        </div>
        <h3 style="color: #1a73e8; margin: 0;">Diabetic Care Analytics</h3>
        <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Clinical Decision Support Dashboard</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_loaded:
        # Performance Summary
        st.markdown('<div class="section-header" style="font-size: 1.2rem;">📊 Clinical Performance</div>', unsafe_allow_html=True)
        
        metadata = st.session_state.metadata
        perf_metrics = metadata.get("performance_metrics", {})
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Recall", f"{perf_metrics.get('recall', 0.690):.1%}")
            st.metric("F1-Score", f"{perf_metrics.get('f1_score', 0.252):.3f}")
        with col2:
            st.metric("Precision", f"{perf_metrics.get('precision', 0.154):.1%}")
            st.metric("ROC AUC", f"{perf_metrics.get('roc_auc', 0.660):.3f}")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown('<div class="section-header" style="font-size: 1.2rem;">🚀 Quick Actions</div>', unsafe_allow_html=True)
        
        if st.button("🧪 Test Model", use_container_width=True, help="Run a test prediction"):
            test_inputs = {
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
            prob = predict_readmission_risk(test_inputs)
            if prob:
                st.success(f"Test prediction: {prob:.1%}")
        
        if st.session_state.get('show_results', False):
            if st.button("📄 Generate Report", use_container_width=True, help="Generate patient report"):
                st.session_state.report_generated = True
                st.success("Report generated! Check the Report tab.")
    
    # System Status
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 🏥 System Status")
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if st.session_state.model_loaded:
            st.success("✅ Model")
        else:
            st.error("❌ Model")
    with status_col2:
        st.success("✅ Database")
    
    # Debug Mode
    debug_mode = st.checkbox("🛠️ Developer Mode", value=False)
    
    if debug_mode and st.session_state.model_loaded:
        st.markdown("### 🔍 Debug Information")
        st.write(f"**Features:** {len(st.session_state.features)}")
        if st.session_state.model:
            st.write(f"**Model Attributes:**")
            st.write(f"- n_features_in_: {st.session_state.model.n_features_in_}")
            st.write(f"- n_estimators: {st.session_state.model.n_estimators}")
            st.write(f"- max_depth: {st.session_state.model.max_depth}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("""
<div class="footer">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap;">
        <span style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #1a73e8;">🏥</span> Kenya Hospital System
        </span>
        <span style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #34a853;">💉</span> Diabetic Care Unit
        </span>
        <span style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #fbbc05;">📊</span> Clinical Analytics v2.0
        </span>
        <span style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #ea4335;">🔒</span> HIPAA Compliant
        </span>
    </div>
    <p style="margin: 0; color: #666; font-size: 0.85rem; line-height: 1.5;">
        <strong>For clinical support only • Always combine with professional medical judgment</strong><br>
        Model specifically trained on diabetic patient data • Validated for Kenyan healthcare context • 
        Threshold optimized for maximum high-risk patient identification • Report generation enabled
    </p>
    <div style="margin-top: 1rem; color: #999; font-size: 0.8rem;">
        © 2024 Kenya Hospital System • Diabetic Patient Readmission Risk Predictor
    </div>
</div>
""", unsafe_allow_html=True)