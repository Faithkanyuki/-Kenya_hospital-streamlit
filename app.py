#import os
#os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np
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
st.write("Clinical tool for predicting patient readmission risk")

# Clear cache button
if st.button("🔄 Clear Cache and Reload Model"):
    st.cache_resource.clear()
    st.rerun()

# System info
with st.expander("System Information", expanded=True):
    st.write(f"**Python version:** {sys.version.split()[0]}")
    try:
        import sklearn
        st.write(f"**scikit-learn version:** {sklearn.__version__}")
    except ImportError:
        st.write("**scikit-learn:** Not available")
    st.write(f"**pandas version:** {pd.__version__}")
    st.write(f"**numpy version:** {np.__version__}")
    
    # List all files
    st.write("**Files in directory:**")
    files = os.listdir('.')
    for file in files:
        st.write(f"- {file}")

# COMPATIBILITY LAYER FOR MODEL LOADING
def patch_sklearn_model(model):
    """Patch sklearn model for compatibility with older versions"""
    
    # Patch 1: Add missing attributes to Random Forest
    if hasattr(model, 'estimators_'):
        for tree in model.estimators_:
            # These attributes were added in newer sklearn versions
            if not hasattr(tree, 'monotonic_cst'):
                tree.monotonic_cst = None
            if not hasattr(tree, 'missing_values_in_feature_mask'):
                tree.missing_values_in_feature_mask = None
            if not hasattr(tree, 'missing_go_to_left'):
                tree.missing_go_to_left = None
    
    # Patch 2: Ensure feature attributes exist
    if not hasattr(model, 'feature_names_in_'):
        model.feature_names_in_ = None
    
    if not hasattr(model, 'n_features_in_'):
        model.n_features_in_ = None
    
    return model

# Load model with robust error handling
@st.cache_resource
def load_model_and_data():
    """Load model, features, and metadata with compatibility patches"""
    try:
        st.write("📦 Loading model files...")
        
        # Check if files exist
        required_files = [
            "hospital_rf_20260121_streamlit.joblib",
            "model_features.joblib",
            "hospital_metadata_20260121.pkl"
        ]
        
        for file in required_files:
            if not os.path.exists(file):
                st.error(f"❌ Missing file: {file}")
                return None, [], {}
            else:
                st.write(f"✅ Found: {file} ({os.path.getsize(file)/1024:.1f} KB)")
        
        # Load with joblib
        import joblib
        
        # Load features and metadata first
        features = joblib.load("model_features.joblib")
        metadata = joblib.load("hospital_metadata_20260121.pkl")
        
        st.write(f"✅ Loaded {len(features)} features")
        
        # Try to load model with multiple strategies
        st.write("🔧 Loading model with compatibility mode...")
        
        try:
            # Strategy 1: Direct load with joblib
            model = joblib.load("hospital_rf_20260121_streamlit.joblib")
            st.write("✅ Model loaded with standard joblib")
        except Exception as e1:
            st.warning(f"Standard load failed: {str(e1)[:100]}...")
            
            # Strategy 2: Try with pickle directly
            try:
                import pickle
                with open("hospital_rf_20260121_streamlit.joblib", 'rb') as f:
                    model = pickle.load(f)
                st.write("✅ Model loaded with pickle")
            except Exception as e2:
                st.error(f"Pickle load also failed: {str(e2)[:100]}...")
                return None, [], {}
        
        # Apply compatibility patches
        st.write("🔧 Applying compatibility patches...")
        model = patch_sklearn_model(model)
        
        # Verify the model
        st.write("🔍 Verifying model...")
        
        # Check model attributes
        model_attrs = {
            'Has estimators_': hasattr(model, 'estimators_'),
            'Has n_estimators': hasattr(model, 'n_estimators'),
            'Has predict_proba': hasattr(model, 'predict_proba'),
            'Has classes_': hasattr(model, 'classes_')
        }
        
        for attr, value in model_attrs.items():
            st.write(f"  - {attr}: {'✅' if value else '❌'}")
        
        # Set feature names if missing
        if not hasattr(model, 'feature_names_in_') or model.feature_names_in_ is None:
            model.feature_names_in_ = np.array(features)
            st.write("✅ Added feature_names_in_ attribute")
        
        if not hasattr(model, 'n_features_in_') or model.n_features_in_ is None:
            model.n_features_in_ = len(features)
            st.write(f"✅ Set n_features_in_ = {len(features)}")
        
        # Simple test prediction
        st.write("🧪 Running test prediction...")
        try:
            # Create simple test data
            test_data = pd.DataFrame({feat: [0.0] for feat in features})
            
            # Set some values for common features
            common_features = ['time_in_hospital', 'num_lab_procedures', 'num_medications', 
                              'total_hospital_visits', 'number_emergency', 'age_numeric']
            
            for feat in common_features:
                if feat in features:
                    test_data.at[0, feat] = 1.0
            
            # Ensure correct data type
            test_data = test_data.astype(np.float32)
            
            # Make prediction
            proba = model.predict_proba(test_data)
            
            if len(proba.shape) == 2 and proba.shape[1] >= 2:
                test_prob = proba[0, 1]
                st.write(f"✅ Test prediction: {test_prob:.4f} ({test_prob*100:.1f}%)")
            else:
                st.warning(f"Unexpected probability shape: {proba.shape}")
                test_prob = 0.5
                
        except Exception as e:
            st.warning(f"Test prediction issue: {str(e)[:100]}...")
            test_prob = 0.5
        
        st.success(f"✅ Model ready: {len(features)} features")
        
        return model, features, metadata
        
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None, [], {}

# Load the model
model, features, metadata = load_model_and_data()

# Set threshold
if metadata and "model_info" in metadata:
    threshold = metadata["model_info"].get("optimal_threshold", 0.48)
else:
    threshold = 0.48

# SIMPLIFIED PREDICTION FUNCTION
def predict_readmission_simple(user_inputs, model, features):
    """Simplified prediction function"""
    
    if model is None:
        st.error("Model not loaded")
        return None
    
    try:
        # Create DataFrame with all zeros
        input_data = pd.DataFrame({feat: [0.0] for feat in features})
        
        # Map user inputs to features
        # This is a simplified mapping - you need to adapt based on your actual feature names
        
        # Direct numeric mappings
        numeric_fields = {
            'time_in_hospital': 'time_in_hospital',
            'num_lab_procedures': 'num_lab_procedures', 
            'num_medications': 'num_medications',
            'total_hospital_visits': 'total_hospital_visits',
            'number_emergency': 'number_emergency',
            'age_numeric': 'age_numeric'
        }
        
        for input_key, feature_name in numeric_fields.items():
            if feature_name in features:
                input_data.at[0, feature_name] = float(user_inputs[input_key])
        
        # Handle medications changed
        if 'num_medications_changed' in features:
            input_data.at[0, 'num_medications_changed'] = 1.0 if user_inputs['num_medications_changed'] == "Yes" else 0.0
        
        # Convert to float32 for compatibility
        input_data = input_data.astype(np.float32)
        
        # Make prediction
        probabilities = model.predict_proba(input_data)
        
        if probabilities.shape[1] >= 2:
            risk_probability = probabilities[0, 1]
            return float(risk_probability)
        else:
            return float(probabilities[0, 0])
            
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# UI - SIMPLIFIED VERSION
st.header("Patient Assessment")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Information")
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 30, 7)
    num_lab_procedures = st.number_input("Number of Lab Procedures", 0, 200, 45)
    num_medications = st.number_input("Number of Medications", 0, 100, 12)
    num_medications_changed = st.selectbox("Medications Changed?", ["No", "Yes"])
    total_hospital_visits = st.number_input("Total Hospital Visits (past year)", 0, 50, 3)
    number_emergency = st.number_input("Emergency Visits (past year)", 0, 20, 1)
    age_numeric = st.slider("Age", 18, 100, 58)

with col2:
    st.subheader("Demographic Information")
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Other"])
    
    # Simplified admission types for demo
    admission_type = st.selectbox("Admission Type", [
        "Emergency", "Urgent", "Elective", "Other"
    ])
    
    # Simplified discharge types
    discharge_disposition = st.selectbox("Discharge Disposition", [
        "Discharged to home",
        "Transferred to facility", 
        "Other"
    ])
    
    age_group = st.selectbox("Age Group", ["18-45", "46-65", "66-85", "86+"])

# Predict button
st.markdown("---")
if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
    if model is None:
        st.error("⚠️ Model not loaded. Check if all required files are uploaded.")
        st.info("Required files:")
        st.code("""
        - hospital_rf_20260121_streamlit.joblib
        - model_features.joblib  
        - hospital_metadata_20260121.pkl
        """)
    else:
        with st.spinner("Calculating risk..."):
            user_inputs = {
                'time_in_hospital': time_in_hospital,
                'num_lab_procedures': num_lab_procedures,
                'num_medications': num_medications,
                'num_medications_changed': num_medications_changed,
                'total_hospital_visits': total_hospital_visits,
                'number_emergency': number_emergency,
                'age_numeric': age_numeric,
                'gender': gender,
                'admission_type': admission_type,
                'discharge_disposition': discharge_disposition,
                'age_group': age_group
            }
            
            prob = predict_readmission_simple(user_inputs, model, features)
            
            if prob is not None:
                # Display results
                st.success(f"✅ Assessment Complete: {prob:.1%} risk")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Risk Probability", f"{prob:.1%}")
                    st.metric("Threshold", f"{threshold:.1%}")
                    
                with col2:
                    if prob >= threshold:
                        st.error("**HIGH RISK** - Priority follow-up needed")
                    else:
                        st.success("**LOW RISK** - Standard care")
                
                # Show debug info
                with st.expander("📊 Model Details"):
                    if hasattr(model, 'n_estimators'):
                        st.write(f"**Model:** Random Forest ({model.n_estimators} trees)")
                    st.write(f"**Features used:** {len(features)}")
                    st.write(f"**Probability:** {prob:.4f}")

# Sidebar
with st.sidebar:
    st.title("About")
    st.write("**Clinical Decision Support Tool**")
    st.write("Version 1.0")
    st.write("For Kenya Hospital System")
    
    if metadata:
        st.divider()
        st.write("**Model Performance:**")
        if "performance_metrics" in metadata:
            metrics = metadata["performance_metrics"]
            st.write(f"Recall: {metrics.get('recall', 0.69):.1%}")
            st.write(f"AUC: {metrics.get('roc_auc', 0.66):.3f}")

st.markdown("---")
st.caption("Kenya Hospital System • Clinical Decision Support Tool")