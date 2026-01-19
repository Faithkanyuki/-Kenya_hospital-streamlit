#!/bin/bash

# Create requirements.txt with all packages and versions
cat > requirements.txt << 'EOF'
streamlit==1.28.0
pandas==1.5.3
numpy==1.24.4
scikit-learn==1.3.2
joblib==1.3.2
rich>=10.14.0,<14
typing-extensions>=3.10.0,<5
EOF

# Install all packages from requirements.txt
pip install --upgrade pip
pip install -r requirements.txt

echo "All packages installed successfully"
