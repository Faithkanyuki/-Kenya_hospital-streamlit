# Kenya Hospital Readmission Predictor

Streamlit app for predicting 30-day hospital readmission risk.

## Files:
- app.py - Main Streamlit application
- random_forest_model.joblib - Trained Random Forest model
- feature_names.pkl - Feature names (48 features)
- model_metadata.pkl - Model metadata and threshold (0.48)
- requirements.txt - Python dependencies
- runtime.txt - Python 3.9.18

## Local Testing:
pip install -r requirements.txt
streamlit run app.py

## Model Performance:
- Recall: 69.0%
- Precision: 15.4%
- Optimal Threshold: 0.48
- F1-Score: 0.252
