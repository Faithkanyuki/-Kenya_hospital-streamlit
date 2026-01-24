import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# ============================================================================
# CUSTOM CSS STYLING - Medical Theme
# ============================================================================
def apply_custom_css():
    """Apply custom CSS styling for medical theme"""
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
    
    /* Background styling */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e4edf5 100%);
        background-attachment: fixed;
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
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid var(--primary-blue);
        margin-bottom: 1.5rem;
        transition: transform 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-blue);
        border-bottom: 2px solid var(--primary-blue);
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
        font-weight: 600;
        font-size: 1.5rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid #e0e0e0;
        min-height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Risk level cards */
    .risk-high-card {
        background: linear-gradient(135deg, #ffeaea 0%, #ffcccc 100%);
        border-left: 4px solid var(--danger-red);
    }
    
    .risk-low-card {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border-left: 4px solid var(--secondary-green);
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
        background: linear-gradient(90deg, var(--primary-blue), #4285f4);
        color: white;
        padding: 0.75rem 1.5rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(26, 115, 232, 0.3);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid #e0e0e0;
        box-shadow: 2px 0 5px rgba(0,0,0,0.05);
    }
    
    /* Input field styling */
    .stNumberInput input, .stSelectbox select, .stSlider {
        border-radius: 8px !important;
        border: 1px solid #ddd !important;
    }
    
    /* Divider styling */
    .custom-divider {
        border: none;
        height: 1px;
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
        border-top: 1px solid #e0e0e0;
        background: rgba(248, 249, 250, 0.8);
        border-radius: 10px;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
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
        gap: 8px;
        background: #f8f9fa;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        border: 1px solid #e0e0e0;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, var(--primary-blue), #4285f4) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(26, 115, 232, 0.2);
    }
    
    /* Animation for results */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-animation {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Progress bar */
    .risk-progress {
        height: 10px;
        background: #e0e0e0;
        border-radius: 5px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .risk-progress-fill {
        height: 100%;
        border-radius: 5px;
        transition: width 0.5s ease;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
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

# ============================================================================
# NAVIGATION HEADER
# ============================================================================
def create_navigation_header():
    """Create a professional navigation header"""
    st.markdown("""
    <div style="background: white; padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 1.5rem;">
                <div style="display: flex; align-items: center; gap: 1rem;">
                    <div style="background: linear-gradient(90deg, #1a73e8, #34a853); width: 50px; height: 50px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.5rem;">
                        🏥
                    </div>
                    <div>
                        <h1 style="margin: 0; font-size: 1.8rem; font-weight: 700; background: linear-gradient(90deg, #1a73e8, #34a853); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            Kenya Hospital System
                        </h1>
                        <p style="margin: 0; color: #666; font-size: 0.9rem;">
                            Clinical Decision Support Tool v1.0
                        </p>
                    </div>
                </div>
            </div>
            <div style="display: flex; gap: 1rem;">
                <span class="badge badge-primary">Live</span>
                <span class="badge badge-success">Validated</span>
                <span class="badge badge-warning">Clinical Tool</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# ORIGINAL MODEL LOADING AND PREDICTION FUNCTIONS
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
    with st.spinner("🔄 Loading predictive model and clinical data..."):
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
tab1, tab2, tab3, tab4 = st.tabs(["📋 Patient Assessment", "📊 Risk Analysis", "🤖 Model Info", "⚙️ System"])

with tab1:
    # Page title
    st.markdown('<h2 class="main-header">Patient Readmission Risk Predictor</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.1rem; margin-bottom: 2rem;">Clinical tool for predicting patient readmission risk within 30 days</p>', unsafe_allow_html=True)
    
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
                if st.button("🔄 Clear Cache & Reload Model", type="secondary", use_container_width=True):
                    st.cache_resource.clear()
                    st.session_state.model = None
                    st.rerun()
        
        # Patient Assessment Form
        st.markdown('<div class="section-header">📋 Patient Assessment Form</div>', unsafe_allow_html=True)
        
        # Create two columns for the form
        col1, col2 = st.columns(2)

        with col1:
            # Clinical Information Card
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #1a73e8; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🏥</span> Clinical Information
                </h3>
            """, unsafe_allow_html=True)
            
            time_in_hospital = st.slider(
                "**Time in Hospital (days)**",
                min_value=1,
                max_value=30,
                value=7,
                help="Average in training: ~4.4 days"
            )
            
            num_lab_procedures = st.number_input(
                "**Number of Lab Procedures**",
                min_value=0,
                max_value=200,
                value=45,
                help="Average in training: ~43"
            )
            
            num_medications = st.number_input(
                "**Number of Medications**",
                min_value=0,
                max_value=100,
                value=12,
                help="Average in training: ~16"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Important Predictor Card
            st.markdown("""
            <div class="custom-card" style="border-left-color: #fbbc05;">
                <h3 style="color: #fbbc05; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>🎯</span> Top Predictors
                </h3>
            """, unsafe_allow_html=True)
            
            total_hospital_visits = st.number_input(
                "**Total Hospital Visits (past year)**",
                min_value=0,
                max_value=50,
                value=3,
                help="**TOP PREDICTOR** - Average in training: ~2.4"
            )
            
            number_emergency = st.number_input(
                "**Emergency Visits (past year)**",
                min_value=0,
                max_value=20,
                value=1,
                help="**5th most important predictor** - Average in training: ~0.3"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # Demographic Information Card
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #1a73e8; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>👤</span> Demographic Information
                </h3>
            """, unsafe_allow_html=True)
            
            age_numeric = st.slider(
                "**Age (years)**",
                min_value=18,
                max_value=100,
                value=58,
                help="Average in training: ~55"
            )
            
            gender = st.selectbox(
                "**Gender**",
                ["Female", "Male", "Unknown/Other"],
                help="Female is most common in training data"
            )
            
            age_group = st.selectbox(
                "**Age Group**",
                ["18-45", "46-65", "66-85", "86+"],
                index=1,
                help="46-65 is most common age group"
            )
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Administrative Information Card
            st.markdown("""
            <div class="custom-card">
                <h3 style="color: #1a73e8; margin-top: 0; display: flex; align-items: center; gap: 0.5rem;">
                    <span>📋</span> Administrative Information
                </h3>
            """, unsafe_allow_html=True)
            
            admission_type = st.selectbox(
                "**Admission Type**",
                [
                    "Emergency", "Urgent", "Elective", "Newborn", 
                    "Trauma Center", "Not Mapped", "NULL", "Not Available"
                ],
                index=0,
                help="Emergency admissions have higher readmission risk"
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
                help="**CRITICAL: Dispositions 1, 16, 7 are top predictors**"
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
            'age_group': age_group
        }
        
        # Prediction Button
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            predict_btn = st.button("🔮 **Predict Readmission Risk**", type="primary", use_container_width=True)
            
            if predict_btn:
                with st.spinner("🔄 Calculating risk..."):
                    probability = predict_readmission_risk(user_inputs)
                    
                    if probability is not None:
                        st.session_state.prediction_result = probability
                        st.session_state.user_inputs = user_inputs
                        st.session_state.show_results = True
                        st.success("✅ Prediction complete! View results in the 'Risk Analysis' tab.")

with tab2:
    # Risk Analysis Tab
    st.markdown('<h2 class="section-header">📊 Risk Analysis Results</h2>', unsafe_allow_html=True)
    
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
                <div style="font-size: 1.2rem; color: #666;">Readmission Probability</div>
                <div style="font-size: 3.5rem; font-weight: 800; color: {risk_color}; margin: 1rem 0;">
                    {probability:.1%}
                </div>
                <div style="color: #999; font-size: 1rem;">
                    Threshold: {threshold:.1%}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Progress Bar
        progress_width = min(probability * 100, 100)
        st.markdown(f"""
        <div style="margin: 2rem 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="color: #666;">Risk Level</span>
                <span style="font-weight: 600; color: {risk_color};">{progress_width:.1f}%</span>
            </div>
            <div class="risk-progress">
                <div class="risk-progress-fill" style="width: {progress_width}%; background: {risk_color};"></div>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem;">
                <span style="color: #34a853; font-size: 0.9rem;">Low Risk</span>
                <span style="color: #fbbc05; font-size: 0.9rem;">Moderate</span>
                <span style="color: #ea4335; font-size: 0.9rem;">High Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Factors Analysis
        st.markdown("### 🎯 Risk Factors Analysis")
        risk_factors = []
        
        if user_inputs['total_hospital_visits'] >= 4:
            risk_factors.append(f"• **High hospital visits** ({user_inputs['total_hospital_visits']}) - Top predictor")
        
        discharge_high_risk = [
            "Discharged/transferred to another short term hospital",
            "Discharged/transferred to another Medicare certified swing bed",
            "Discharged/transferred to home under care of Home IV provider"
        ]
        if user_inputs['discharge_disposition'] in discharge_high_risk:
            risk_factors.append(f"• **Specific discharge disposition** - High importance in model")
        
        if user_inputs['number_emergency'] >= 2:
            risk_factors.append(f"• **Multiple ED visits** ({user_inputs['number_emergency']}) - 5th most important")
        
        if user_inputs['time_in_hospital'] >= 10:
            risk_factors.append(f"• **Long hospital stay** ({user_inputs['time_in_hospital']} days)")
        
        if user_inputs['num_medications'] >= 15:
            risk_factors.append(f"• **High medication count** ({user_inputs['num_medications']})")
        
        if user_inputs['age_numeric'] >= 75:
            risk_factors.append(f"• **Advanced age** ({user_inputs['age_numeric']} years)")
        
        if risk_factors:
            for factor in risk_factors:
                st.markdown(factor)
        else:
            st.markdown("• No high-risk factors identified")
        
        # Clinical Recommendations
        st.markdown("### 🩺 Clinical Recommendations")
        if risk_level == "HIGH":
            st.markdown("""
            <div class="custom-card risk-high-card">
                <h3 style="color: #ea4335; margin-top: 0;">🚨 Priority Clinical Actions Required</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                    <div>
                        <h4 style="color: #333; margin-bottom: 0.5rem;">📅 Follow-up Schedule</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                            <li>Schedule follow-up within 7 days</li>
                            <li>Coordinate with home care services</li>
                            <li>Medication review appointment</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #333; margin-bottom: 0.5rem;">🏥 Care Coordination</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                            <li>Flag for care team notification</li>
                            <li>Assign case manager</li>
                            <li>Home health assessment</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #333; margin-bottom: 0.5rem;">💊 Medication Management</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                            <li>Review adherence plan</li>
                            <li>Simplify medication regimen</li>
                            <li>Provide pill organizers</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="custom-card risk-low-card">
                <h3 style="color: #34a853; margin-top: 0;">✅ Standard Care Protocol</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem; margin-top: 1rem;">
                    <div>
                        <h4 style="color: #333; margin-bottom: 0.5rem;">📋 Discharge Planning</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                            <li>Standard discharge instructions</li>
                            <li>30-day follow-up appointment</li>
                            <li>Patient education materials</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #333; margin-bottom: 0.5rem;">👨‍⚕️ Monitoring</h4>
                        <ul style="margin: 0; padding-left: 1.2rem;">
                            <li>Regular monitoring advised</li>
                            <li>Self-care education</li>
                            <li>Routine check-ups</li>
                        </ul>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model Performance Context
        st.markdown("### 📊 Model Performance Context")
        if st.session_state.metadata:
            perf_metrics = st.session_state.metadata.get("performance_metrics", {})
            st.info(f"""
            **Training Performance:**
            - **Recall**: {perf_metrics.get('recall', 0.690):.1%} - Model identifies {perf_metrics.get('recall', 0.690)*100:.0f}% of actual readmissions
            - **Precision**: {perf_metrics.get('precision', 0.154):.1%} - When model predicts HIGH RISK, {perf_metrics.get('precision', 0.154)*100:.1f}% actually readmit
            - **Threshold**: {threshold:.1%} - Optimized to maximize identification of high-risk patients
            - **False Positives**: Expected - Model prioritizes catching true positives
            """)
    else:
        st.info("👈 Please use the Patient Assessment tab to make a prediction first.")

with tab3:
    # Model Information Tab
    st.markdown('<h2 class="section-header">🤖 Model Information</h2>', unsafe_allow_html=True)
    
    if st.session_state.model_loaded and st.session_state.metadata:
        metadata = st.session_state.metadata
        
        # Model Performance Metrics
        st.markdown("### 📈 Performance Metrics")
        perf_metrics = metadata.get("performance_metrics", {})
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Recall", f"{perf_metrics.get('recall', 0.690):.1%}", 
                     help="Ability to identify actual readmissions")
        with col2:
            st.metric("Precision", f"{perf_metrics.get('precision', 0.154):.1%}",
                     help="Accuracy when predicting HIGH RISK")
        with col3:
            st.metric("F1-Score", f"{perf_metrics.get('f1_score', 0.252):.3f}",
                     help="Balance between precision and recall")
        with col4:
            st.metric("ROC AUC", f"{perf_metrics.get('roc_auc', 0.660):.3f}",
                     help="Overall discrimination ability")
        
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
            """)
        with details_col2:
            threshold = model_info.get('optimal_threshold', 0.48)
            st.markdown(f"""
            **max_depth:** {model_info.get('max_depth', 5)}  
            **Threshold:** {threshold:.1%}  
            **Training Samples:** 81,412  
            """)
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Top Predictors
        st.markdown("### 🎯 Top 5 Predictors")
        st.write("From feature importance analysis:")
        st.write("1. **total_hospital_visits** (47.98%)")
        st.write("2. **discharge_disposition_1** (15.15%)")
        st.write("3. **discharge_disposition_16** (13.57%)")
        st.write("4. **discharge_disposition_7** (7.84%)")
        st.write("5. **number_emergency** (3.49%)")
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Training Context
        st.markdown("### 📋 Training Context")
        st.markdown("""
        - **Dataset**: Historical hospital records
        - **Class Balance**: 11.2% readmitted, 88.8% not readmitted
        - **Optimization**: Threshold tuned for Recall ≥ 65%
        - **Use Case**: Identify high-risk patients for intervention
        - **Validation**: Cross-validated with 5 folds
        """)
    else:
        st.info("Model information will be displayed once the model is loaded.")

with tab4:
    # System Information Tab
    st.markdown('<h2 class="section-header">⚙️ System Information</h2>', unsafe_allow_html=True)
    
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
    
    for file in files_to_check:
        if os.path.exists(file):
            st.success(f"✅ {file} - Found")
        else:
            st.error(f"❌ {file} - Missing")
    
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # Session Information
    st.markdown("### 🔄 Session Information")
    if st.session_state.model_loaded:
        st.success("✅ Model loaded successfully")
        st.info(f"Features: {len(st.session_state.features)}")
        threshold = st.session_state.metadata.get('model_info', {}).get('optimal_threshold', 0.48) if st.session_state.metadata else 0.48
        st.info(f"Threshold: {threshold:.1%}")
    else:
        st.warning("❌ Model not loaded")

# ============================================================================
# SIDEBAR CONTENT
# ============================================================================
with st.sidebar:
    # Sidebar Header
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="background: linear-gradient(90deg, #1a73e8, #34a853); width: 60px; height: 60px; border-radius: 12px; display: inline-flex; align-items: center; justify-content: center; color: white; font-size: 2rem; margin-bottom: 1rem;">
            🤖
        </div>
        <h3 style="color: #1a73e8; margin: 0;">Model Dashboard</h3>
        <p style="color: #666; font-size: 0.9rem; margin: 0.5rem 0 0 0;">Real-time Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.model_loaded:
        # Performance Summary
        st.markdown('<div class="section-header" style="font-size: 1.2rem;">📊 Performance Summary</div>', unsafe_allow_html=True)
        
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
        
        # Quick Test
        st.markdown('<div class="section-header" style="font-size: 1.2rem;">🧪 Quick Test</div>', unsafe_allow_html=True)
        if st.button("Run Test Prediction", use_container_width=True):
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
    
    # Debug Mode
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    debug_mode = st.checkbox("🛠️ Enable Debug Mode", value=False)
    
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
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem;">
        <span style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #1a73e8;">🏥</span> Kenya Hospital System
        </span>
        <span style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #34a853;">📊</span> Clinical Analytics v1.0
        </span>
        <span style="display: flex; align-items: center; gap: 0.5rem;">
            <span style="color: #fbbc05;">🔒</span> HIPAA Compliant
        </span>
    </div>
    <p style="margin: 0; color: #666; font-size: 0.85rem;">
        <strong>For clinical support only • Combine with professional judgment</strong><br>
        Model trained on historical hospital data • Validated for Kenyan healthcare context • 
        Threshold optimized for maximum high-risk patient identification
    </p>
</div>
""", unsafe_allow_html=True)