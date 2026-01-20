import streamlit as st
import pandas as pd
import numpy as np
import warnings
import sys

# ===== CRITICAL FIXES =====
warnings.filterwarnings('ignore')

# Monkey patch to prevent monotonic_cst errors BEFORE any sklearn imports
try:
    import sklearn.tree._classes as tree_classes
    # Add dummy monotonic_cst attribute to prevent errors
    if not hasattr(tree_classes.DecisionTreeClassifier, 'monotonic_cst'):
        tree_classes.DecisionTreeClassifier.monotonic_cst = None
except ImportError:
    # sklearn not installed yet, that's ok
    pass
except Exception:
    # Other errors, also ok
    pass
# ===== END CRITICAL FIXES =====

# Page setup
st.set_page_config(
    page_title="Kenya Hospital Readmission Predictor",
    layout="wide"
)

# Title
st.title("Kenya Hospital Readmission Risk Predictor")
st.write("Clinical tool for predicting patient readmission risk")

# Debug information
with st.expander("System Information", expanded=False):
    st.write(f"**Python version:** {sys.version.split()[0]}")
    # Try to import sklearn, but don't fail if it's not available
    try:
        import sklearn
        st.write(f"**scikit-learn version:** {sklearn.__version__}")
    except ImportError:
        st.write("**scikit-learn:** Not available (check requirements.txt)")
    
    st.write(f"**pandas version:** {pd.__version__}")
    st.write(f"**numpy version:** {np.__version__}")
    

# ===== FIXED: load_model_and_data function =====
@st.cache_resource
def load_model_and_data():
    """Load model, features, and metadata - FIXED for monotonic_cst"""
    try:
        # Use joblib (best for scikit-learn models)
        import joblib
        
        # Load model
        model = joblib.load("random_forest_model.joblib")
        
        # CRITICAL FIX: Patch any trees that have monotonic_cst
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if hasattr(estimator, 'monotonic_cst'):
                    try:
                        # Remove the attribute
                        delattr(estimator, 'monotonic_cst')
                    except:
                        try:
                            # If can't delete, set to None
                            estimator.monotonic_cst = None
                        except:
                            pass
        
        st.success("Model loaded and patched successfully")
        
    except Exception as e:
        st.error(f"Error with joblib: {e}")
        
        # Try fallback with pickle
        try:
            import pickle
            
            with open("random_forest_model.joblib", "rb") as f:
                model = pickle.load(f)
            
            # Apply same patch
            if hasattr(model, 'estimators_'):
                for estimator in model.estimators_:
                    if hasattr(estimator, 'monotonic_cst'):
                        try:
                            delattr(estimator, 'monotonic_cst')
                        except:
                            try:
                                estimator.monotonic_cst = None
                            except:
                                pass
            
            st.success("Model loaded with pickle")
            
        except Exception as e2:
            st.error(f"All loading methods failed: {e2}")
            model = None
    
    # Load features
    features = []
    try:
        import joblib
        features = joblib.load("feature_names.pkl")
        st.success(f"Loaded {len(features)} features")
    except Exception as e:
        st.error(f"Error loading features: {e}")
        features = []
    
    # Load metadata
    metadata = {}
    try:
        import joblib
        metadata = joblib.load("model_metadata.pkl")
        st.success("Metadata loaded successfully")
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        metadata = {}
    
    return model, features, metadata
# ===== END FIXED FUNCTION =====

# Load everything
model, features, metadata = load_model_and_data()

# Get threshold from metadata - YOUR 0.48 THRESHOLD
threshold = metadata.get("model_info", {}).get("optimal_threshold", 0.48)

# Prediction function
def predict_readmission_risk(data_dict, all_features):
    """Make prediction with all 48 features - FIXED VERSION"""
    try:
        if model is None or not all_features:
            st.error("Model or features not loaded properly")
            return None
        
        # Create dataframe with ALL features initialized to 0.0 (important!)
        df = pd.DataFrame({feat: [0.0] for feat in all_features})
        
        # Update with user input - ensure we're setting values correctly
        for key, value in data_dict.items():
            if key in df.columns:
                # Use .at for single value assignment
                df.at[0, key] = float(value)
        
        # DEBUG: Show what we're sending to the model
        with st.expander("🔍 Debug Info - Click to see prediction inputs"):
            st.write("**Input DataFrame shape:**", df.shape)
            st.write("**Expected features:**", len(all_features))
            st.write("**Non-zero input features:**")
            non_zero_features = df.loc[0, df.loc[0] != 0]
            st.dataframe(non_zero_features)
            st.write("**Value ranges:**")
            st.write(f"- Min value: {df.min().min()}")
            st.write(f"- Max value: {df.max().max()}")
            st.write(f"- Total features with value=1: {(df == 1).sum().sum()}")
            st.write(f"- Total features with value>0: {(df > 0).sum().sum()}")
        
        # Make prediction
        prob = model.predict_proba(df)[0, 1]
        
        # Validate probability is in correct range [0, 1]
        if prob < 0 or prob > 1:
            st.error(f"⚠️ INVALID PROBABILITY: {prob}")
            st.error("This indicates a serious data preprocessing issue!")
            st.write("**Debugging information:**")
            st.write(f"- Input data shape: {df.shape}")
            st.write(f"- Model expects {model.n_features_in_} features")
            st.write("- First 10 values:", df.iloc[0, :10].tolist())
            return None
        
        return prob
    
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        return None

# Helper function to convert categorical selections to one-hot encoding
def prepare_features(user_inputs):
    """Convert user selections to one-hot encoded features - FIXED VERSION"""
    features_dict = {}
    
    # NUMERIC FEATURES - ensure they're floats, not strings
    numeric_map = {
        'time_in_hospital': float(user_inputs['time_in_hospital']),
        'num_lab_procedures': float(user_inputs['num_lab_procedures']),
        'num_medications': float(user_inputs['num_medications']),
        'num_medications_changed': float(1 if user_inputs['num_medications_changed'] == "Yes" else 0),
        'total_hospital_visits': float(user_inputs['total_hospital_visits']),
        'number_emergency': float(user_inputs['number_emergency']),
        'age_numeric': float(user_inputs['age_numeric'])
    }
    features_dict.update(numeric_map)
    
    # Gender one-hot encoding (gender_0, gender_1, gender_2)
    gender_options = ["Female", "Male", "Unknown/Other"]
    for i, option in enumerate(gender_options):
        features_dict[f'gender_{i}'] = 1.0 if user_inputs['gender'] == option else 0.0
    
    # Admission type one-hot encoding (admission_type_0 to admission_type_7)
    admission_options = [
        "Emergency", "Urgent", "Elective", "Newborn", 
        "Trauma Center", "Not Mapped", "NULL", "Not Available"
    ]
    for i, option in enumerate(admission_options):
        features_dict[f'admission_type_{i}'] = 1.0 if user_inputs['admission_type'] == option else 0.0
    
    # Discharge disposition one-hot encoding (discharge_disposition_0 to discharge_disposition_25)
    discharge_options = [
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
    
    # Initialize all discharge_disposition features to 0
    for i in range(26):
        features_dict[f'discharge_disposition_{i}'] = 0.0
    
    # Set only the selected one to 1
    try:
        selected_idx = discharge_options.index(user_inputs['discharge_disposition'])
        features_dict[f'discharge_disposition_{selected_idx}'] = 1.0
    except ValueError:
        st.warning(f"Unknown discharge disposition: {user_inputs['discharge_disposition']}")
    
    # Age group one-hot encoding (age_group_0 to age_group_3)
    age_group_options = ["18-45", "46-65", "66-85", "86+"]
    for i, option in enumerate(age_group_options):
        features_dict[f'age_group_{i}'] = 1.0 if user_inputs['age_group'] == option else 0.0
    
    return features_dict
# User interface
st.header("Patient Assessment")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Clinical Information")
    
    # YOUR NUMERIC FEATURES
    time_in_hospital = st.slider("Time in Hospital (days)", 1, 30, 7, key="time_in_hospital")
    num_lab_procedures = st.number_input("Number of Lab Procedures", 0, 200, 45, key="num_lab_procedures")
    num_medications = st.number_input("Number of Medications", 0, 100, 12, key="num_medications")
    num_medications_changed = st.selectbox("Medications Changed?", ["No", "Yes"], key="num_medications_changed")
    total_hospital_visits = st.number_input("Total Hospital Visits (past year)", 0, 50, 3, key="total_hospital_visits")
    number_emergency = st.number_input("Emergency Visits (past year)", 0, 20, 1, key="number_emergency")
    age_numeric = st.slider("Age", 18, 100, 58, key="age_numeric")

with col2:
    st.subheader("Demographic & Administrative Information")
    
    # Gender (YOUR: gender_0, gender_1, gender_2)
    gender = st.selectbox(
        "Gender",
        ["Female", "Male", "Unknown/Other"],
        key="gender"
    )
    
    # Admission Type (YOUR: admission_type_0 to admission_type_7)
    admission_type = st.selectbox(
        "Admission Type",
        [
            "Emergency", "Urgent", "Elective", "Newborn", 
            "Trauma Center", "Not Mapped", "NULL", "Not Available"
        ],
        key="admission_type"
    )
    
    # Discharge Disposition (YOUR: discharge_disposition_0 to discharge_disposition_25)
    discharge_disposition = st.selectbox(
        "Discharge Disposition",
        [
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
        ],
        key="discharge_disposition"
    )
    
    # Age Group (YOUR: age_group_0 to age_group_3)
    age_group = st.selectbox(
        "Age Group",
        ["18-45", "46-65", "66-85", "86+"],
        key="age_group"
    )

# Predict button
st.markdown("---")
if st.button("Predict Readmission Risk", type="primary", use_container_width=True):
    if model is None:
        st.error("Model not available. Please check if model files exist.")
    else:
        with st.spinner("Calculating risk..."):
            # Collect all user inputs
            user_inputs = {
                'time_in_hospital': float(time_in_hospital),
                'num_lab_procedures': float(num_lab_procedures),
                'num_medications': float(num_medications),
                'num_medications_changed': 1 if num_medications_changed == "Yes" else 0,
                'total_hospital_visits': float(total_hospital_visits),
                'number_emergency': float(number_emergency),
                'age_numeric': float(age_numeric),
                'gender': gender,
                'admission_type': admission_type,
                'discharge_disposition': discharge_disposition,
                'age_group': age_group
            }
            
            # Convert to one-hot encoded features
            features_dict = prepare_features(user_inputs)
            
            # Get prediction
            prob = predict_readmission_risk(features_dict, features)
            
            if prob is not None:
                st.success("Assessment Complete")
                
                # Show results
                st.subheader("Risk Assessment Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Risk Probability", 
                        f"{prob:.1%}",
                        delta=f"{prob - threshold:.1%} vs threshold"
                    )
                    st.metric("Decision Threshold", f"{threshold:.1%}")
                
                with col2:
                    risk_level = "HIGH RISK" if prob >= threshold else "LOW RISK"
                    st.metric("Risk Level", risk_level)
                    
                    if prob >= threshold:
                        st.error("Priority follow-up required")
                        st.write("""
                        **Clinical Actions:**
                        - Schedule follow-up within 7 days
                        - Coordinate with home care services
                        - Review medication adherence
                        - Flag for care team notification
                        """)
                    else:
                        st.success("Standard care protocol")
                        st.write("""
                        **Clinical Actions:**
                        - Standard discharge planning
                        - 30-day follow-up appointment
                        - Patient education materials
                        - Regular monitoring advised
                        """)
                
                with col3:
                    st.write("**Key Risk Factors Identified:**")
                    if total_hospital_visits >= 4:
                        st.write(f"- High hospital visits ({total_hospital_visits})")
                    if number_emergency >= 2:
                        st.write(f"- Multiple ED visits ({number_emergency})")
                    if time_in_hospital >= 10:
                        st.write(f"- Long hospital stay ({time_in_hospital} days)")
                    if num_medications >= 15:
                        st.write(f"- High medication count ({num_medications})")
                    if age_numeric >= 75:
                        st.write(f"- Advanced age ({age_numeric} years)")
                    
                    # Show actual probabilities
                    st.write(f"**Probability:** {prob:.2%}")
                    st.write(f"**Threshold:** {threshold:.2%}")

# Sidebar information
with st.sidebar:
    st.title("Model Information")
    
    # Performance metrics from YOUR model
    perf_metrics = metadata.get("performance_metrics", {})
    
    st.write("**Model Performance Metrics:**")
    st.metric("Recall", f"{perf_metrics.get('recall', 0.690):.1%}")
    st.metric("Precision", f"{perf_metrics.get('precision', 0.154):.1%}")
    st.metric("F1-Score", f"{perf_metrics.get('f1_score', 0.252):.3f}")
    st.metric("ROC AUC", f"{perf_metrics.get('roc_auc', 0.660):.3f}")
    st.metric("Optimal Threshold", f"{threshold:.3f}")
    
    st.divider()
    
    st.write("**Model Details:**")
    model_info = metadata.get("model_info", {})
    st.write(f"- **Algorithm:** Random Forest")
    st.write(f"- **Features used:** {len(features)}")
    st.write(f"- **n_estimators:** {model_info.get('n_estimators', 285)}")
    st.write(f"- **max_depth:** {model_info.get('max_depth', 5)}")
    
    st.divider()
    
    st.write("**Top Predictors:**")
    # Show top features from YOUR model
    top_features_list = [
        "total_hospital_visits",
        "discharge_disposition_1", 
        "discharge_disposition_16",
        "discharge_disposition_7",
        "number_emergency"
    ]
    for i, feat in enumerate(top_features_list, 1):
        st.write(f"{i}. {feat}")
    
    st.divider()
    
    st.write("**Clinical Guidelines:**")
    st.write("- **High risk:** ≥48% probability")
    st.write("- **Target recall:** ≥65%")
    st.write("- **Monitoring:** Review after 30 days")
    st.write("- **Tool:** Clinical decision support only")

# Footer
st.markdown("---")
footer_col1, footer_col2 = st.columns([2, 1])

with footer_col1:
    st.caption("**Developed for Kenya Hospital System • Clinical Decision Support Tool v1.0**")
    st.caption(f"Model trained on {model_info.get('save_date', 'Unknown date')}")

with footer_col2:
    st.caption("For clinical support only")
    st.caption("Always combine with professional judgment")
