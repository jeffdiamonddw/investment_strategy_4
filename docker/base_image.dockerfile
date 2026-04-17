# Use an official Python ARM64 base (matching your Graviton4 hardware)
FROM python:3.11-slim-bookworm

# 1. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 2. Install System Dependencies & CBC Solver
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    unzip \
    curl \
    libopenblas-dev \
    libhdf5-dev \
    libnetcdf-dev \
    coinor-cbc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 3. Install AWS CLI v2 (ARM64 version)
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf aws awscliv2.zip

# 4. Install Python Libraries
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    numpy==2.2.0 \
    pandas==3.0.1 \
    xarray==2026.4.0 \
    fsspec==2026.3.0 \
    s3fs==2026.3.0 \
    awswrangler==3.16.0 \
    scipy==1.15.0 \
    scikit-learn==1.8.0 \
    pymoo==0.6.1.6 \
    pyomo==6.10.0 \
    boto3==1.37.0 \
    botocore==1.37.0 \
    joblib==1.4.0 \
    requests==2.32.0 \
    python-dotenv==1.0.1

# 5. Create Directories and Set Workspace
# We create /share specifically for your local volume mounting
RUN mkdir -p /app /share
WORKDIR /app

# Code and CMD are removed to keep the base image generic