#!/bin/bash

# 1. Update and Install System Dependencies
echo "Updating system and installing base tools..."
sudo apt-get update && sudo apt-get upgrade -y
sudo apt-get install -y \
    build-essential \
    python3-pip \
    python3-venv \
    unzip \
    curl \
    libopenblas-dev \
    libhdf5-dev \
    libnetcdf-dev

# 2. Install AWS CLI (ARM64 version for Graviton4)
echo "Installing AWS CLI v2 for ARM64..."
curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
unzip -q awscliv2.zip
sudo ./aws/install
rm -rf aws awscliv2.zip

# 3. Install CBC Solver (Coin-OR) - Required for explore_cluster.py
echo "Installing CBC Solver..."
sudo apt-get install -y coinor-cbc

# 4. Setup Python Virtual Environment
echo "Creating Virtual Environment 'tt_env'..."
cd /home/ubuntu
python3 -m venv tt_env
source tt_env/bin/activate

# 5. Install Python Libraries (2026 Verified Versions)
echo "Installing Python Stack (Optimized for Pandas 3.0 / NumPy 2.2)..."
pip install --upgrade pip
pip install \
    numpy==2.2.0 \
    pandas==3.0.1 \
    xarray==2026.4.0 \
    fsspec==2026.3.0 \
    s3fs==2026.3.0 \
    awswrangler==3.16.0\
    scipy==1.15.0 \
    scikit-learn==1.8.0 \
    pymoo==0.6.1.6 \
    pyomo==6.10.0 \
    awswrangler==3.16.0 \
    boto3==1.37.0 \
    botocore==1.37.0 \
    joblib==1.4.0 \
    requests==2.32.0 \
    python-dotenv==1.0.1

# 6. Final Sanity Check
echo "------------------------------------------------"
echo "INSTALLATION COMPLETE - VERIFICATION:"
aws --version
cbc -version | head -n 1
python -c "import pandas as pd; import pymoo; print(f'Python Environment: OK (Pandas {pd.__version__})')"
echo "------------------------------------------------"
echo "To start working, run: source ~/tt_env/bin/activate"