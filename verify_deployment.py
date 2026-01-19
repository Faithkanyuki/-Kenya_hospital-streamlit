#!/usr/bin/env python
# Verification script for Kenya Hospital Readmission Predictor deployment
import os
import sys
import joblib

print("="*60)
print("DEPLOYMENT VERIFICATION SCRIPT")
print("="*60)

# Check required files
required_files = [
    "random_forest_model.joblib",
    "feature_names.pkl", 
    "model_metadata.pkl",
    "requirements.txt",
    "app.py"
]

all_good = True
for file in required_files:
    if os.path.exists(file):
        size = os.path.getsize(file) / 1024
        print(f"OK  {file:30} ({size:.1f} KB)")
    else:
        print(f"ERR {file:30} [MISSING]")
        all_good = False

print("-"*60)

if all_good:
    # Test loading the model
    try:
        model = joblib.load("random_forest_model.joblib")
        features = joblib.load("feature_names.pkl")
        metadata = joblib.load("model_metadata.pkl")
        
        print("Model loading test: SUCCESS")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Features: {len(features)}")
        
        # Get threshold
        threshold = metadata.get("model_info", {}).get("optimal_threshold", 0.48)
        print(f"  Optimal threshold: {threshold}")
        
        # Test prediction with dummy data
        import pandas as pd
        
        dummy_input = pd.DataFrame({feat: [0] for feat in features})
        prob = model.predict_proba(dummy_input)
        
        print(f"Prediction test: SUCCESS")
        print(f"  Probability shape: {prob.shape}")
        
    except Exception as e:
        print(f"Loading test failed: {e}")
        all_good = False

print("-"*60)

if all_good:
    print("ALL CHECKS PASSED!")
    print("Your deployment is ready for Streamlit Cloud.")
    print("")
    print("Next steps:")
    print("1. Upload this folder to GitHub")
    print("2. Go to share.streamlit.io")
    print("3. Connect your repository")
    print("4. Deploy!")
else:
    print("SOME CHECKS FAILED")
    print("Please fix the issues above before deployment.")

print("="*60)
