import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS - MEDICAL PROFESSIONAL THEME
# ============================================================================
def load_custom_css():
    st.markdown("""
    <style>
    /* Main theme colors - Medical blue palette */
    :root {
        --primary-color: #1e40af;
        --secondary-color: #3b82f6;
        --accent-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
        --background-light: #f8fafc;
        --text-dark: #1e293b;
    }
    
    /* Main container styling */
    .main {
        background-color: #f8fafc;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 600;
    }
    
    .main-header p {
        color: #e0e7ff;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    /* Section cards */
    .section-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #3b82f6;
        margin-bottom: 1.5rem;
    }
    
    .section-header {
        color: #1e40af;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Risk assessment results */
    .risk-high {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border-left: 6px solid #ef4444;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        border-left: 6px solid #10b981;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .risk-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Input field styling */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stSlider {
        border-radius: 6px;
        border: 2px solid #e2e8f0;
        transition: border-color 0.2s;
    }
    
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e40af;
    }
    
    /* Info boxes */
    .info-box {
        background: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #d1fae5;
        border-left: 4px solid #10b981;
        padding: 1rem;
        border-radius: 6px;
        margin: 1rem 0;
    }
    
    /* Clinical actions list */
    .clinical-actions {
        background: white;
        padding: 1.25rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    .clinical-actions ul {
        margin: 0.5rem 0;
        padding-left: 1.5rem;
    }
    
    .clinical-actions li {
        margin: 0.5rem 0;
        line-height: 1.6;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1e40af;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        padding: 2rem;
        background: white;
        border-top: 3px solid #3b82f6;
        border-radius: 8px;
        margin-top: 3rem;
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-top-color: #3b82f6 !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f1f5f9;
        border-radius: 6px;
        font-weight: 600;
        color: #1e40af;
    }
    
    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 1.5rem;
        }
        
        .section-header {
            font-size: 1.1rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Page setup
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_custom_css()

# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
<div class="main-header">
    <h1>🏥 Kenya Hospital Readmission Risk Predictor</h1>
    <p>Clinical Decision Support Tool for 30-Day Readmission Assessment</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL WITH VALIDATION
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """Load model, features, and metadata with strict validation"""
    try:
        # Load all required files
        model = joblib.load("hospital_rf_20260121_streamlit.joblib")
        features = joblib.load("hospital_features_20260121.pkl")
        metadata = joblib.load("hospital_metadata_20260121.pkl")
        
        # Validation checks
        if not hasattr(model, "feature_names_in_"):
            if hasattr(model, 'feature_importances_'):
                model.feature_names_in_ = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        else:
            model_features = list(model.feature_names_in_)
            saved_features = list(features)
            
            if len(model_features) != len(saved_features):
                st.error(f"❌ Feature count mismatch: Model={len(model_features)}, Saved={len(saved_features)}")
                st.stop()
        
        # Create feature categories
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
        
        if 'optimal_threshold' not in metadata.get('model_info', {}):
            metadata['model_info']['optimal_threshold'] = 0.48
        
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

# Load the model
with st.spinner("🔄 Loading clinical model..."):
    model, features, metadata = load_model_and_data()

# Get threshold from metadata
threshold = metadata.get('model_info', {}).get('optimal_threshold', 0.48)

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_readmission_risk(user_inputs):
    """Make prediction with exact feature engineering from training"""
    
    if model is None:
        st.error("Model not loaded. Please check model files.")
        return None
    
    try:
        feature_dict = {feat: 0.0 for feat in features}
        input_df = pd.DataFrame([feature_dict])
        
        # Set numeric features
        input_df['time_in_hospital'] = float(user_inputs['time_in_hospital'])
        input_df['num_lab_procedures'] = float(user_inputs['num_lab_procedures'])
        input_df['num_medications'] = float(user_inputs['num_medications'])
        input_df['total_hospital_visits'] = float(user_inputs['total_hospital_visits'])
        input_df['number_emergency'] = float(user_inputs['number_emergency'])
        input_df['age_numeric'] = float(user_inputs['age_numeric'])
        
        # Gender mapping
        gender_map = {"Female": 0, "Male": 1, "Unknown/Other": 2}
        gender_idx = gender_map[user_inputs['gender']]
        input_df[f'gender_{gender_idx}'] = 1.0
        
        # Admission type mapping
        admission_map = {
            "Emergency": 0, "Urgent": 1, "Elective": 2, "Newborn": 3,
            "Trauma Center": 4, "Not Mapped": 5, "NULL": 6, "Not Available": 7
        }
        admission_idx = admission_map[user_inputs['admission_type']]
        input_df[f'admission_type_{admission_idx}'] = 1.0
        
        # Discharge disposition mapping
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
        
        # Age group mapping
        age_group_map = {"18-45": 0, "46-65": 1, "66-85": 2, "86+": 3}
        age_group_idx = age_group_map[user_inputs['age_group']]
        input_df[f'age_group_{age_group_idx}'] = 1.0
        
        # Make prediction
        input_df = input_df.astype(np.float32)
        probability = model.predict_proba(input_df)[0, 1]
        
        if not (0 <= probability <= 1):
            st.error(f"Invalid probability: {probability}")
            return None
        
        return probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

# ============================================================================
# USER INTERFACE - PATIENT ASSESSMENT
# ============================================================================
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">📋 Patient Clinical Assessment</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🏥 Clinical Measurements")
    
    time_in_hospital = st.slider(
        "Length of Stay (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Number of days patient spent in hospital during current admission"
    )
    
    num_lab_procedures = st.number_input(
        "Laboratory Procedures Conducted",
        min_value=0,
        max_value=200,
        value=45,
        help="Total number of lab tests performed (average: ~43)"
    )
    
    num_medications = st.number_input(
        "Medications Prescribed",
        min_value=0,
        max_value=100,
        value=12,
        help="Number of distinct medications administered (average: ~16)"
    )
    
    st.markdown("#### 📊 Patient History")
    
    total_hospital_visits = st.number_input(
        "Total Hospital Encounters (Past 12 Months)",
        min_value=0,
        max_value=50,
        value=3,
        help="⭐ PRIMARY RISK FACTOR - Total inpatient and outpatient visits"
    )
    
    number_emergency = st.number_input(
        "Emergency Department Visits (Past 12 Months)",
        min_value=0,
        max_value=20,
        value=1,
        help="⭐ KEY PREDICTOR - Number of ED visits in past year"
    )

with col2:
    st.markdown("#### 👤 Patient Demographics")
    
    age_numeric = st.slider(
        "Patient Age (years)",
        min_value=18,
        max_value=100,
        value=58,
        help="Patient's current age"
    )
    
    age_group = st.selectbox(
        "Age Category",
        ["18-45", "46-65", "66-85", "86+"],
        index=1,
        help="Categorical age grouping for risk stratification"
    )
    
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Unknown/Other"],
        help="Patient's recorded gender"
    )
    
    st.markdown("#### 🏥 Administrative Data")
    
    admission_type = st.selectbox(
        "Admission Type",
        [
            "Emergency", "Urgent", "Elective", "Newborn", 
            "Trauma Center", "Not Mapped", "NULL", "Not Available"
        ],
        index=0,
        help="Type of hospital admission"
    )
    
    discharge_disposition = st.selectbox(
        "Discharge Disposition",
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
        help="⭐ CRITICAL PREDICTOR - Patient discharge destination"
    )

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# PREDICTION BUTTON
# ============================================================================
st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Calculate Readmission Risk", type="primary", use_container_width=True)

# ============================================================================
# PREDICTION RESULTS
# ============================================================================
if predict_button:
    if model is None:
        st.error("❌ Model not available. Please check if model files exist.")
    else:
        with st.spinner("🔄 Analyzing patient data and calculating risk probability..."):
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
                st.success("✅ Risk Assessment Complete")
                
                # Risk classification
                is_high_risk = probability >= threshold
                risk_class = "risk-high" if is_high_risk else "risk-low"
                risk_icon = "🔴" if is_high_risk else "🟢"
                risk_text = "HIGH RISK" if is_high_risk else "LOW RISK"
                risk_color = "#dc2626" if is_high_risk else "#16a34a"
                
                # Display risk level prominently
                st.markdown(f"""
                <div class="{risk_class}">
                    <div class="risk-title" style="color: {risk_color};">
                        {risk_icon} {risk_text} for 30-Day Readmission
                    </div>
                    <p style="font-size: 1.1rem; margin: 0;">
                        Risk Probability: <strong style="font-size: 1.3rem;">{probability:.1%}</strong>
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Detailed metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Risk Score",
                        f"{probability:.1%}",
                        help="Predicted probability of readmission within 30 days"
                    )
                
                with col2:
                    st.metric(
                        "Decision Threshold",
                        f"{threshold:.1%}",
                        help="Model threshold optimized for high sensitivity"
                    )
                
                with col3:
                    delta_val = probability - threshold
                    st.metric(
                        "Threshold Margin",
                        f"{delta_val:+.1%}",
                        delta=f"{delta_val:+.1%}",
                        delta_color="inverse",
                        help="Distance from decision threshold"
                    )
                
                with col4:
                    confidence = abs(probability - 0.5) * 2
                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.1%}",
                        help="Model confidence in prediction"
                    )
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Clinical recommendations
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">📋 Clinical Action Plan</div>', unsafe_allow_html=True)
                    
                    if is_high_risk:
                        st.markdown("""
                        <div class="clinical-actions">
                        <h4 style="color: #dc2626; margin-top: 0;">⚠️ Enhanced Care Protocol Required</h4>
                        <ul>
                            <li><strong>Schedule early follow-up</strong> within 7 days of discharge</li>
                            <li><strong>Coordinate home health services</strong> for post-discharge monitoring</li>
                            <li><strong>Medication reconciliation</strong> and adherence counseling</li>
                            <li><strong>Care team notification</strong> for enhanced discharge planning</li>
                            <li><strong>Patient education</strong> on warning signs and when to seek care</li>
                            <li><strong>Consider transitional care</strong> program enrollment</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="clinical-actions">
                        <h4 style="color: #16a34a; margin-top: 0;">✅ Standard Care Protocol</h4>
                        <ul>
                            <li><strong>Schedule routine follow-up</strong> within 30 days</li>
                            <li><strong>Standard discharge instructions</strong> and education materials</li>
                            <li><strong>Medication list</strong> and administration guidance</li>
                            <li><strong>Primary care coordination</strong> for continuity of care</li>
                            <li><strong>Emergency contact information</strong> provided to patient</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col_right:
                    st.markdown('<div class="section-card">', unsafe_allow_html=True)
                    st.markdown('<div class="section-header">🎯 Key Risk Factors Identified</div>', unsafe_allow_html=True)
                    
                    risk_factors = []
                    
                    if total_hospital_visits >= 4:
                        risk_factors.append(("🔴", f"Multiple hospital encounters ({total_hospital_visits} visits)", "Primary risk driver"))
                    
                    discharge_high_risk = [
                        "Discharged/transferred to another short term hospital",
                        "Discharged/transferred to another Medicare certified swing bed",
                        "Discharged/transferred to home under care of Home IV provider"
                    ]
                    if discharge_disposition in discharge_high_risk:
                        risk_factors.append(("🟠", "Complex discharge disposition", "Moderate risk factor"))
                    
                    if number_emergency >= 2:
                        risk_factors.append(("🟠", f"Frequent ED utilization ({number_emergency} visits)", "Significant predictor"))
                    
                    if time_in_hospital >= 10:
                        risk_factors.append(("🟡", f"Extended length of stay ({time_in_hospital} days)", "Contributing factor"))
                    
                    if num_medications >= 15:
                        risk_factors.append(("🟡", f"Polypharmacy ({num_medications} medications)", "Complexity indicator"))
                    
                    if age_numeric >= 75:
                        risk_factors.append(("🟡", f"Advanced age ({age_numeric} years)", "Age-related risk"))
                    
                    if risk_factors:
                        for icon, factor, description in risk_factors:
                            st.markdown(f"""
                            <div style="margin: 0.75rem 0; padding: 0.75rem; background: #f8fafc; border-radius: 6px; border-left: 3px solid #3b82f6;">
                                <strong>{icon} {factor}</strong><br>
                                <span style="color: #64748b; font-size: 0.9rem;">{description}</span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No significant high-risk factors identified in current assessment")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Model context
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("""
                **📊 Clinical Model Performance Context**
                
                This predictive model was validated on 81,412 patient encounters with the following performance metrics:
                - **Sensitivity (Recall)**: 69.0% - Identifies 69% of patients who will be readmitted
                - **Specificity**: Optimized to minimize missed high-risk patients
                - **Positive Predictive Value**: 15.4% - When flagged as high-risk, 15.4% actually readmit
                - **ROC AUC**: 0.660 - Good discriminative ability
                
                The model threshold is calibrated to prioritize patient safety by identifying most patients at risk,
                accepting higher false-positive rates to prevent missed interventions.
                """)
                st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================================
with st.sidebar:
    st.markdown("### 🤖 Clinical Model Information")
    
    with st.expander("📈 Performance Metrics", expanded=True):
        perf_metrics = metadata.get("performance_metrics", {})
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Sensitivity", f"{perf_metrics.get('recall', 0.690):.1%}")
            st.metric("F1-Score", f"{perf_metrics.get('f1_score', 0.252):.3f}")
        with metric_col2:
            st.metric("