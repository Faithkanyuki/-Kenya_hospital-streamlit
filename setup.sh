#!/bin/bash

# Install all required packages
pip install --upgrade pip
pip install streamlit==1.28.0
pip install pandas==1.5.3
pip install numpy==1.24.4
pip install scikit-learn==1.3.2
pip install joblib==1.3.2

echo "All packages installed successfully"
