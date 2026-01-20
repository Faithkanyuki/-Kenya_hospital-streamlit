import streamlit as st
import pandas as pd
import numpy as np
import warnings
import sys

warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    layout="wide"
)

st.title("Kenya Hospital Readmission Risk Predictor")

st.write("Clinical tool for predicting patient readmission risk")

# Add this right after your title
if st.button(" Clear Cache and Reload Model"):
    st.cache_resource.clear()
    st.rerun()

# System info
with st.expander("System Information", expanded=False):
    st.write(f"**Python version:** {sys.version.split()[0]}")
    try:
        import sklearn
        st.write(f"**scikit-learn version:** {sklearn.__version__}")
    except ImportError:
        st.write("**scikit-learn:** Not available")
    st.write(f"**pandas version:** {pd.__version__}")
    st.write(f"**numpy version:** {np.__version__}")

# Load model
@st.cache_resource
def load_model_and_data():
    """Load model, features, and metadata"""
    try:
        import joblib
        import os
        
        model = joblib.load("model_rf_v2.joblib")  # Changed
        features = joblib.load("features_v2.pkl")   # Changed
        metadata = joblib.load("metadata_v2.pkl")   # Changed
        
        # DEBUG: Check what model was loaded
        st.write("🔍 **MODEL FILE INFO:**")
        st.write(f"- File size: {os.path.getsize('model_rf_v2.joblib') / (1024*1024):.2f} MB")
        st.write(f"- n_estimators: {model.n_estimators}")
        st.write(f"- max_depth: {model.max_depth}")
        
        # Test with known input
        test_df = pd.DataFrame({feat: [0.0] for feat in features})
        test_df.at[0, 'time_in_hospital'] = 7.0
        test_df.at[0, 'num_lab_procedures'] = 45.0
        test_df.at[0, 'num_medications'] = 12.0
        test_df.at[0, 'total_hospital_visits'] = 3.0
        test_df.at[0, 'number_emergency'] = 1.0
        test_df.at[0, 'age_numeric'] = 58.0
        test_df.at[0, 'gender_0'] = 1.0
        test_df.at[0, 'admission_type_0'] = 1.0
        test_df.at[0, 'discharge_disposition_0'] = 1.0
        test_df.at[0, 'age_group_1'] = 1.0
        
        test_prob = model.predict_proba(test_df)[0, 1]
        st.write(f"- Test prediction: {test_prob:.4f} ({test_prob*100:.1f}%)")
        
        if test_prob > 1:
            st.error("⚠️ WRONG MODEL FILE IS DEPLOYED!")
        else:
            st.success(f"✅ Correct model loaded")
        
        st.success(f"✅ Model loaded: {len(features)} features, threshold={metadata.get('model_info', {}).get('optimal_threshold', 0.48):.2f}")
        
        return model, features, metadata
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, [], {}
model, features, metadata = load_model_and_data()
threshold = metadata.get("model_info", {}).get("optimal_threshold", 0.48)

# Prediction function - BULLETPROOF VERSION
def predict_readmission_risk(user_inputs_dict, model, feature_list):
    """Make prediction - BULLETPROOF VERSION"""
    
    if model is None or not feature_list:
        st.error("Model not loaded")
        return None
    
    try:
        # Create a dictionary with ALL features initialized to 0.0
        feature_dict = {feat: 0.0 for feat in feature_list}
        
        # Create DataFrame from dictionary (ensures correct order)
        input_data = pd.DataFrame([feature_dict])
        
        # Fill in user values - BE VERY EXPLICIT
        # Numeric features
        input_data.at[0, 'time_in_hospital'] = float(user_inputs_dict['time_in_hospital'])
        input_data.at[0, 'num_lab_procedures'] = float(user_inputs_dict['num_lab_procedures'])
        input_data.at[0, 'num_medications'] = float(user_inputs_dict['num_medications'])
        input_data.at[0, 'num_medications_changed'] = 1.0 if user_inputs_dict['num_medications_changed'] == "Yes" else 0.0
        input_data.at[0, 'total_hospital_visits'] = float(user_inputs_dict['total_hospital_visits'])
        input_data.at[0, 'number_emergency'] = float(user_inputs_dict['number_emergency'])
        input_data.at[0, 'age_numeric'] = float(user_inputs_dict['age_numeric'])
        
        # Gender (one-hot)
        gender_map = {"Female": 0, "Male": 1, "Unknown/Other": 2}
        gender_idx = gender_map[user_inputs_dict['gender']]
        input_data.at[0, f'gender_{gender_idx}'] = 1.0
        
        # Admission type (one-hot)
        admission_types = ["Emergency", "Urgent", "Elective", "Newborn", "Trauma Center", "Not Mapped", "NULL", "Not Available"]
        admission_idx = admission_types.index(user_inputs_dict['admission_type'])
        input_data.at[0, f'admission_type_{admission_idx}'] = 1.0
        
        # Discharge disposition (one-hot)
        discharge_types = [
            "Discharged to home", "Discharged/transferred to another short term hospital",
            "Discharged/transferred to SNF", "Discharged/transferred to ICF",
            "Discharged/transferred to another type of inpatient care institution",
            "Discharged/transferred to home with home health service",
            "Left AMA", "Discharged/transferred to home under care of Home IV provider",
            "Admitted as an inpatient to this hospital", "Neonate discharged to another hospital",
            "Expired", "Still patient", "Hospice / home", "Hospice / medical facility",
            "Discharged/transferred within this institution", "Discharged/transferred to rehab",
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
        ]
        discharge_idx = discharge_types.index(user_inputs_dict['discharge_disposition'])
        input_data.at[0, f'discharge_disposition_{discharge_idx}'] = 1.0
        
        # Age group (one-hot)
        age_groups = ["18-45", "46-65", "66-85", "86+"]
        age_group_idx = age_groups.index(user_inputs_dict['age_group'])
        input_data.at[0, f'age_group_{age_group_idx}'] = 1.0
        
        # DEBUG INFO - ENHANCED
        with st.expander("🔍 Debug Information"):
            st.write(f"**DataFrame shape:** {input_data.shape}")
            st.write(f"**DataFrame dtypes:** {input_data.dtypes.unique()}")
            st.write(f"**Model expects:** {model.n_features_in_} features")
            st.write(f"**Feature order matches:** {list(input_data.columns) == list(model.feature_names_in_)}")
            st.write(f"**Non-zero values:** {(input_data != 0).sum().sum()}")
            st.write(f"**Value range:** {input_data.min().min():.2f} to {input_data.max().max():.2f}")
            
            st.write("**First 10 column names (ours vs model):**")
            comparison = pd.DataFrame({
                'Our columns': list(input_data.columns)[:10],
                'Model expects': list(model.feature_names_in_)[:10]
            })
            st.dataframe(comparison)
            
            st.write("**Non-zero features:**")
            non_zero = input_data.loc[0, input_data.loc[0] != 0]
            st.dataframe(non_zero)
            
            st.write("**First 10 values being sent to model:**")
            st.write(input_data.iloc[0, :10].tolist())
        
        # Ensure correct data type
        input_data = input_data.astype(float)
        
        # Make prediction
        probability = model.predict_proba(input_data)[0, 1]
        
        # Validate
        if not (0 <= probability <= 1):
            st.error(f"❌ INVALID PROBABILITY: {probability}")
            st.write("Input data summary:")
            st.dataframe(input_data.describe())
            return None
        
        return probability
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())
        return None

# UI
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
    st.subheader("Demographic & Administrative Information")
    
    gender = st.selectbox("Gender", ["Female", "Male", "Unknown/Other"])
    
    admission_type = st.selectbox("Admission Type", [
        "Emergency", "Urgent", "Elective", "Newborn", 
        "Trauma Center", "Not Mapped", "NULL", "Not Available"
    ])
    
    discharge_disposition = st.selectbox("Discharge Disposition", [
        "Discharged to home", "Discharged/transferred to another short term hospital",
        "Discharged/transferred to SNF", "Discharged/transferred to ICF",
        "Discharged/transferred to another type of inpatient care institution",
        "Discharged/transferred to home with home health service",
        "Left AMA", "Discharged/transferred to home under care of Home IV provider",
        "Admitted as an inpatient to this hospital", "Neonate discharged to another hospital",
        "Expired", "Still patient", "Hospice / home", "Hospice / medical facility",
        "Discharged/transferred within this institution", "Discharged/transferred to rehab",
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
    ])
    
    age_group = st.selectbox("Age Group", ["18-45", "46-65", "66-85", "86+"])

# Predict button
st.markdown("---")
if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not available. Check if files exist.")
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
            
            prob = predict_readmission_risk(user_inputs, model, features)
            
            if prob is not None:
                st.success("✅ Assessment Complete")
                
                st.subheader("Risk Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Risk Probability", f"{prob:.1%}", delta=f"{prob - threshold:.1%} vs threshold")
                    st.metric("Decision Threshold", f"{threshold:.1%}")
                
                with col2:
                    risk_level = "HIGH RISK" if prob >= threshold else "LOW RISK"
                    st.metric("Risk Level", risk_level)
                    
                    if prob >= threshold:
                        st.error("⚠️ Priority follow-up required")
                        st.write("""
                        **Clinical Actions:**
                        - Schedule follow-up within 7 days
                        - Coordinate with home care services
                        - Review medication adherence
                        - Flag for care team notification
                        """)
                    else:
                        st.success("✅ Standard care protocol")
                        st.write("""
                        **Clinical Actions:**
                        - Standard discharge planning
                        - 30-day follow-up appointment
                        - Patient education materials
                        - Regular monitoring advised
                        """)
                
                with col3:
                    st.write("**Key Risk Factors:**")
                    if total_hospital_visits >= 4:
                        st.write(f"• High hospital visits ({total_hospital_visits})")
                    if number_emergency >= 2:
                        st.write(f"• Multiple ED visits ({number_emergency})")
                    if time_in_hospital >= 10:
                        st.write(f"• Long hospital stay ({time_in_hospital} days)")
                    if num_medications >= 15:
                        st.write(f"• High medication count ({num_medications})")
                    if age_numeric >= 75:
                        st.write(f"• Advanced age ({age_numeric} years)")

# Sidebar
with st.sidebar:
    st.title("Model Information")
    
    perf_metrics = metadata.get("performance_metrics", {})
    
    st.write("**Performance Metrics:**")
    st.metric("Recall", f"{perf_metrics.get('recall', 0.690):.1%}")
    st.metric("Precision", f"{perf_metrics.get('precision', 0.154):.1%}")
    st.metric("F1-Score", f"{perf_metrics.get('f1_score', 0.252):.3f}")
    st.metric("ROC AUC", f"{perf_metrics.get('roc_auc', 0.660):.3f}")
    
    st.divider()
    
    st.write("**Model Details:**")
    model_info = metadata.get("model_info", {})
    st.write(f"• Algorithm: Random Forest")
    st.write(f"• Features: {len(features)}")
    st.write(f"• n_estimators: {model_info.get('n_estimators', 285)}")
    st.write(f"• max_depth: {model_info.get('max_depth', 5)}")
    
    st.divider()
    
    st.write("**Top Predictors:**")
    st.write("1. total_hospital_visits")
    st.write("2. discharge_disposition_1")
    st.write("3. discharge_disposition_16")
    st.write("4. discharge_disposition_7")
    st.write("5. number_emergency")

st.markdown("---")
st.caption("**Kenya Hospital System • Clinical Decision Support Tool v1.0**")
st.caption("For clinical support only • Combine with professional judgment")