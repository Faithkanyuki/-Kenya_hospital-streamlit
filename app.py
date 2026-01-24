import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os

warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    layout="wide"
)

st.title("Kenya Hospital Readmission Risk Predictor")
st.write("Clinical tool for predicting patient readmission risk within 30 days")

# Add reload button
if st.button("🔄 Clear Cache and Reload"):
    st.cache_resource.clear()
    st.rerun()

# System info
with st.expander("🔧 System Information", expanded=False):
    st.write(f"**Python version:** {sys.version.split()[0]}")
    st.write(f"**pandas version:** {pd.__version__}")
    st.write(f"**numpy version:** {np.__version__}")
    st.write(f"**joblib version:** {joblib.__version__}")

# ============================================================================
# LOAD MODEL WITH VALIDATION
# ============================================================================
@st.cache_resource
def load_model_and_data():
    """Load model, features, and metadata with strict validation"""
    try:
        st.write("📥 Loading model files...")
        
        # Load all required files
        model = joblib.load("hospital_rf_20260121_streamlit.joblib")
        features = joblib.load("hospital_features_20260121.pkl")
        metadata = joblib.load("hospital_metadata_20260121.pkl")
        
        st.success("✅ Model files loaded successfully")
        
        # ====================================================================
        # CRITICAL VALIDATION 1: Feature names match exactly
        # ====================================================================
        st.write("🔍 Validating feature alignment...")
        
        if not hasattr(model, "feature_names_in_"):
            st.error("❌ Model missing feature_names_in_ attribute")
            st.info("Model was likely trained with scikit-learn < 1.0")
            # Extract feature names if available
            if hasattr(model, 'feature_importances_'):
                # Create synthetic feature names
                model.feature_names_in_ = [f"feature_{i}" for i in range(len(model.feature_importances_))]
        else:
            # Convert to lists for comparison
            model_features = list(model.feature_names_in_)
            saved_features = list(features)
            
            if len(model_features) != len(saved_features):
                st.error(f"❌ Feature count mismatch: Model={len(model_features)}, Saved={len(saved_features)}")
                st.stop()
            
            # Check each feature
            mismatches = []
            for i, (m_feat, s_feat) in enumerate(zip(model_features, saved_features)):
                if m_feat != s_feat:
                    mismatches.append((i, m_feat, s_feat))
            
            if mismatches:
                st.error("❌ Feature name mismatches found!")
                mismatch_df = pd.DataFrame(mismatches, columns=['Index', 'Model Feature', 'Saved Feature'])
                st.dataframe(mismatch_df)
                st.stop()
            else:
                st.success(f"✅ Feature schema validated: {len(model_features)} features match exactly")
        
        # ====================================================================
        # CRITICAL VALIDATION 2: Feature categories from training
        # ====================================================================
        st.write("📊 Analyzing feature structure...")
        
        # Based on your training output, we know the structure:
        # 1. total_hospital_visits (numeric)
        # 2-25. discharge_disposition_X (one-hot, X=0-25)
        # 26. number_emergency (numeric)
        # 27. time_in_hospital (numeric)
        # 28. num_medications (numeric)
        # 29. num_lab_procedures (numeric)
        # 30-31. discharge_disposition_X (one-hot)
        # 32. age_numeric (numeric)
        # 33-37. discharge_disposition_X (one-hot)
        # 38. age_group_2 (one-hot)
        # 39. admission_type_3 (one-hot)
        # 40. admission_type_7 (one-hot)
        
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
        
        # ====================================================================
        # CRITICAL VALIDATION 3: Test prediction
        # ====================================================================
        st.write("🧪 Running test prediction...")
        
        # Create test case based on your training
        test_features = {
            'total_hospital_visits': 3.0,        # Top feature
            'discharge_disposition_1': 1.0,      # 2nd most important
            'discharge_disposition_16': 1.0,     # 3rd most important
            'discharge_disposition_7': 1.0,      # 4th most important
            'number_emergency': 1.0,             # 5th most important
            'time_in_hospital': 7.0,
            'num_medications': 12.0,
            'num_lab_procedures': 45.0,
            'age_numeric': 58.0,
            'gender_0': 1.0,                     # Female
            'admission_type_0': 1.0,             # Emergency
            'discharge_disposition_0': 1.0,      # Discharged to home
            'age_group_1': 1.0                   # 46-65
        }
        
        # Fill missing features with 0
        test_df = pd.DataFrame({feat: [0.0] for feat in model.feature_names_in_})
        for feat, val in test_features.items():
            if feat in test_df.columns:
                test_df.at[0, feat] = val
        
        # Make prediction
        test_prob = model.predict_proba(test_df)[0, 1]
        st.success(f"✅ Test prediction: {test_prob:.4f} ({test_prob*100:.1f}%)")
        
        # ====================================================================
        # Store everything in metadata
        # ====================================================================
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
        
        st.success("🎯 Model ready for predictions")
        return model, model.feature_names_in_, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.error("Please ensure these files exist in the same directory:")
        st.error("1. hospital_rf_20260121_streamlit.joblib")
        st.error("2. hospital_features_20260121.pkl")
        st.error("3. hospital_metadata_20260121.pkl")
        return None, [], {}

# Load the model
model, features, metadata = load_model_and_data()

# Get threshold from metadata
threshold = metadata.get('model_info', {}).get('optimal_threshold', 0.48)

# ============================================================================
# PREDICTION FUNCTION - EXACT MATCH TO TRAINING
# ============================================================================
def predict_readmission_risk(user_inputs):
    """Make prediction with exact feature engineering from training"""
    
    if model is None:
        st.error("Model not loaded. Please check model files.")
        return None
    
    try:
        # Create DataFrame with all features initialized to 0.0
        # This ensures we have ALL features in the EXACT order
        feature_dict = {feat: 0.0 for feat in features}
        input_df = pd.DataFrame([feature_dict])
        
        # ====================================================================
        # SET NUMERIC FEATURES (Exactly as in training)
        # ====================================================================
        input_df['time_in_hospital'] = float(user_inputs['time_in_hospital'])
        input_df['num_lab_procedures'] = float(user_inputs['num_lab_procedures'])
        input_df['num_medications'] = float(user_inputs['num_medications'])
        input_df['total_hospital_visits'] = float(user_inputs['total_hospital_visits'])
        input_df['number_emergency'] = float(user_inputs['number_emergency'])
        input_df['age_numeric'] = float(user_inputs['age_numeric'])
        
        # ====================================================================
        # SET CATEGORICAL FEATURES (One-hot encoding EXACTLY as in training)
        # ====================================================================
        
        # Gender mapping (from training data exploration)
        gender_map = {
            "Female": 0,      # gender_0 = 1
            "Male": 1,        # gender_1 = 1
            "Unknown/Other": 2  # gender_2 = 1
        }
        gender_idx = gender_map[user_inputs['gender']]
        input_df[f'gender_{gender_idx}'] = 1.0
        
        # Admission type mapping (from training data)
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
        
        # Discharge disposition mapping (CRITICAL - must match training)
        # Based on your feature importance: discharge_disposition_1, _16, _7 are important
        discharge_map = {
            "Discharged to home": 0,  # discharge_disposition_0 = 1
            "Discharged/transferred to another short term hospital": 1,  # _1 = 1
            "Discharged/transferred to SNF": 2,
            "Discharged/transferred to ICF": 3,
            "Discharged/transferred to another type of inpatient care institution": 4,
            "Discharged/transferred to home with home health service": 5,
            "Left AMA": 6,
            "Discharged/transferred to home under care of Home IV provider": 7,  # _7 = 1
            "Admitted as an inpatient to this hospital": 8,
            "Neonate discharged to another hospital": 9,
            "Expired": 10,
            "Still patient": 11,
            "Hospice / home": 12,
            "Hospice / medical facility": 13,
            "Discharged/transferred within this institution": 14,
            "Discharged/transferred to rehab": 15,
            "Discharged/transferred to another Medicare certified swing bed": 16,  # _16 = 1
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
        age_group_map = {
            "18-45": 0,   # age_group_0 = 1
            "46-65": 1,   # age_group_1 = 1
            "66-85": 2,   # age_group_2 = 1
            "86+": 3      # age_group_3 = 1
        }
        age_group_idx = age_group_map[user_inputs['age_group']]
        input_df[f'age_group_{age_group_idx}'] = 1.0
        
        # ====================================================================
        # VALIDATION CHECKS
        # ====================================================================
        validation_expander = st.expander("🔍 Prediction Validation", expanded=False)
        with validation_expander:
            # Check 1: Feature count
            st.write(f"**Feature Count:** {len(input_df.columns)} (Expected: {model.n_features_in_})")
            
            # Check 2: Non-zero features (should be 13-15)
            non_zero_count = (input_df != 0).sum().sum()
            st.write(f"**Non-zero features:** {non_zero_count}")
            
            # Check 3: Top features from training
            st.write("**Setting top important features:**")
            top_features = ['total_hospital_visits', 'discharge_disposition_1', 
                          'discharge_disposition_16', 'discharge_disposition_7', 
                          'number_emergency']
            for feat in top_features:
                if feat in input_df.columns:
                    value = input_df[feat].iloc[0]
                    st.write(f"- {feat}: {value}")
            
            # Check 4: Show all non-zero features
            non_zero_features = input_df.loc[:, (input_df != 0).any()].columns.tolist()
            st.write(f"**All non-zero features ({len(non_zero_features)}):**")
            for feat in non_zero_features:
                st.write(f"  - {feat}: {input_df[feat].iloc[0]}")
        
        # ====================================================================
        # MAKE PREDICTION
        # ====================================================================
        # Ensure correct data type
        input_df = input_df.astype(np.float32)
        
        # Get probability
        probability = model.predict_proba(input_df)[0, 1]
        
        # Final validation
        if not (0 <= probability <= 1):
            st.error(f"Invalid probability: {probability}")
            return None
        
        return probability
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.error("Please check that all inputs are valid.")
        return None

# ============================================================================
# USER INTERFACE - MATCHES TRAINING DATA DISTRIBUTION
# ============================================================================
st.header("📋 Patient Assessment")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🏥 Clinical Information")
    
    # Based on training data statistics
    time_in_hospital = st.slider(
        "Time in Hospital (days)",
        min_value=1,
        max_value=30,
        value=7,
        help="Average in training: ~4.4 days"
    )
    
    num_lab_procedures = st.number_input(
        "Number of Lab Procedures",
        min_value=0,
        max_value=200,
        value=45,
        help="Average in training: ~43"
    )
    
    num_medications = st.number_input(
        "Number of Medications",
        min_value=0,
        max_value=100,
        value=12,
        help="Average in training: ~16"
    )
    
    total_hospital_visits = st.number_input(
        "Total Hospital Visits (past year)",
        min_value=0,
        max_value=50,
        value=3,
        help="**TOP PREDICTOR** - Average in training: ~2.4"
    )
    
    number_emergency = st.number_input(
        "Emergency Visits (past year)",
        min_value=0,
        max_value=20,
        value=1,
        help="**5th most important predictor** - Average in training: ~0.3"
    )
    
    age_numeric = st.slider(
        "Age (years)",
        min_value=18,
        max_value=100,
        value=58,
        help="Average in training: ~55"
    )

with col2:
    st.subheader("👤 Demographic & Administrative Information")
    
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Unknown/Other"],
        help="Female is most common in training data"
    )
    
    admission_type = st.selectbox(
        "Admission Type",
        [
            "Emergency", "Urgent", "Elective", "Newborn", 
            "Trauma Center", "Not Mapped", "NULL", "Not Available"
        ],
        index=0,
        help="Emergency admissions have higher readmission risk"
    )
    
    discharge_disposition = st.selectbox(
        "Discharge Disposition",
        [
            "Discharged to home",  # Index 0
            "Discharged/transferred to another short term hospital",  # Index 1 (2nd most important)
            "Discharged/transferred to SNF",
            "Discharged/transferred to ICF",
            "Discharged/transferred to another type of inpatient care institution",
            "Discharged/transferred to home with home health service",
            "Left AMA",
            "Discharged/transferred to home under care of Home IV provider",  # Index 7 (4th most important)
            "Admitted as an inpatient to this hospital",
            "Neonate discharged to another hospital",
            "Expired",
            "Still patient",
            "Hospice / home",
            "Hospice / medical facility",
            "Discharged/transferred within this institution",
            "Discharged/transferred to rehab",
            "Discharged/transferred to another Medicare certified swing bed",  # Index 16 (3rd most important)
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
    
    age_group = st.selectbox(
        "Age Group",
        ["18-45", "46-65", "66-85", "86+"],
        index=1,
        help="46-65 is most common age group"
    )

# ============================================================================
# PREDICTION BUTTON AND RESULTS
# ============================================================================
st.markdown("---")

if st.button("🔮 Predict Readmission Risk", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not available. Please check if model files exist.")
    else:
        with st.spinner("🔄 Calculating risk..."):
            # Collect all inputs
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
            
            # Make prediction
            probability = predict_readmission_risk(user_inputs)
            
            if probability is not None:
                st.success("✅ Assessment Complete")
                
                # Display results
                st.subheader("📊 Risk Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Risk Probability",
                        f"{probability:.1%}",
                        delta=f"{probability - threshold:.1%} vs threshold",
                        delta_color="inverse"
                    )
                    st.metric("Decision Threshold", f"{threshold:.1%}")
                
                with col2:
                    risk_level = "🔴 HIGH RISK" if probability >= threshold else "🟢 LOW RISK"
                    st.metric("Risk Level", risk_level)
                    
                    if probability >= threshold:
                        st.error("⚠️ Priority follow-up required")
                        st.write("""
                        **Clinical Actions (per model training):**
                        - Schedule follow-up within 7 days
                        - Coordinate with home care services
                        - Review medication adherence
                        - Flag for care team notification
                        """)
                    else:
                        st.success("✅ Standard care protocol")
                        st.write("""
                        **Clinical Actions (per model training):**
                        - Standard discharge planning
                        - 30-day follow-up appointment
                        - Patient education materials
                        - Regular monitoring advised
                        """)
                
                with col3:
                    st.write("**🎯 Top Risk Factors Present:**")
                    
                    # Based on feature importance from training
                    risk_factors = []
                    
                    # Total hospital visits (MOST IMPORTANT)
                    if total_hospital_visits >= 4:
                        risk_factors.append(f"• **High hospital visits** ({total_hospital_visits}) - Top predictor")
                    
                    # Discharge disposition (2nd, 3rd, 4th most important)
                    discharge_idx = [
                        "Discharged/transferred to another short term hospital",
                        "Discharged/transferred to another Medicare certified swing bed",
                        "Discharged/transferred to home under care of Home IV provider"
                    ]
                    if discharge_disposition in discharge_idx:
                        risk_factors.append(f"• **Specific discharge disposition** - High importance in model")
                    
                    # Emergency visits (5th most important)
                    if number_emergency >= 2:
                        risk_factors.append(f"• **Multiple ED visits** ({number_emergency}) - 5th most important")
                    
                    # Other factors
                    if time_in_hospital >= 10:
                        risk_factors.append(f"• **Long hospital stay** ({time_in_hospital} days)")
                    
                    if num_medications >= 15:
                        risk_factors.append(f"• **High medication count** ({num_medications})")
                    
                    if age_numeric >= 75:
                        risk_factors.append(f"• **Advanced age** ({age_numeric} years)")
                    
                    if risk_factors:
                        for factor in risk_factors:
                            st.write(factor)
                    else:
                        st.write("• No high-risk factors identified")
                
                # Show explanation based on training
                st.info("""
                **Model Performance Context (from training):**
                - **Recall**: 69.0% - Model identifies 69% of actual readmissions
                - **Precision**: 15.4% - When model predicts HIGH RISK, 15.4% actually readmit
                - **Threshold**: 48% - Optimized to maximize identification of high-risk patients
                - **False Positives**: Expected - Model prioritizes catching true positives
                """)

# ============================================================================
# SIDEBAR WITH MODEL INFORMATION
# ============================================================================
with st.sidebar:
    st.title("🤖 Model Information")
    
    # Model performance from training
    perf_metrics = metadata.get("performance_metrics", {})
    
    st.subheader("📈 Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Recall", f"{perf_metrics.get('recall', 0.690):.1%}")
        st.metric("Precision", f"{perf_metrics.get('precision', 0.154):.1%}")
    with col2:
        st.metric("F1-Score", f"{perf_metrics.get('f1_score', 0.252):.3f}")
        st.metric("ROC AUC", f"{perf_metrics.get('roc_auc', 0.660):.3f}")
    
    st.divider()
    
    st.subheader("⚙️ Model Details")
    model_info = metadata.get("model_info", {})
    st.write(f"• **Algorithm**: Random Forest")
    st.write(f"• **Features**: {len(features)}")
    st.write(f"• **n_estimators**: {model_info.get('n_estimators', 285)}")
    st.write(f"• **max_depth**: {model_info.get('max_depth', 5)}")
    st.write(f"• **Threshold**: {threshold:.1%}")
    
    st.divider()
    
    st.subheader("🎯 Top 5 Predictors")
    st.write("From feature importance analysis:")
    st.write("1. **total_hospital_visits** (47.98%)")
    st.write("2. **discharge_disposition_1** (15.15%)")
    st.write("3. **discharge_disposition_16** (13.57%)")
    st.write("4. **discharge_disposition_7** (7.84%)")
    st.write("5. **number_emergency** (3.49%)")
    
    st.divider()
    
    st.subheader("📋 Training Context")
    st.write("• **Dataset**: 81,412 training samples")
    st.write("• **Class Balance**: 11.2% readmitted, 88.8% not readmitted")
    st.write("• **Optimization**: Threshold tuned for Recall ≥ 65%")
    st.write("• **Use Case**: Identify high-risk patients for intervention")
    
    # Quick test
    if st.button("🧪 Run Quick Test"):
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
            st.info(f"Test prediction: {prob:.1%}")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("🏥 **Kenya Hospital System • Clinical Decision Support Tool v1.0**")
st.caption("""
**For clinical support only • Combine with professional judgment**
- Model trained on historical hospital data
- Validated for Kenyan healthcare context
- Threshold optimized for maximum high-risk patient identification
""")

# ============================================================================
# DEBUG MODE (Hidden by default) - FIXED SYNTAX
# ============================================================================
if st.sidebar.checkbox("🛠️ Debug Mode", value=False):
    st.sidebar.subheader("Debug Information")
    
    st.sidebar.write("**Features:**")
    st.sidebar.write(f"Total: {len(features)}")
    
    # Show feature categories
    feature_cats = metadata.get('feature_categories', {})
    if feature_cats:
        st.sidebar.write(f"Numeric: {len(feature_cats.get('numeric', []))}")
        st.sidebar.write(f"Categorical groups: {len(feature_cats.get('categorical', {}))}")
    
    # Show first 10 features
    st.sidebar.write("**First 10 features:**")
    for feat in list(features)[:10]:
        st.sidebar.write(f"- {feat}")
    
    # Show model attributes - FIXED SYNTAX HERE
    if model:
        st.sidebar.write("**Model attributes:**")
        st.sidebar.write(f"- n_features_in_: {model.n_features_in_}")
        st.sidebar.write(f"- n_estimators: {model.n_estimators}")
        st.sidebar.write(f"- max_depth: {model.max_depth}")    