#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# --- CONFIGURATION ---
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID="129861351772"
REPO_NAME="optimization"
IMAGE_TAG="latest"
DOCKERFILE_PATH="docker/pareto_navigator.dockerfile"
FULL_REPO_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo "🚀 Starting deployment for ${REPO_NAME}..."

# 1. Ensure the ECR repository exists
echo "🔍 Checking ECR repository..."
aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION} > /dev/null 2>&1 || echo "✅ Repository already exists."

# 2. Authenticate Docker to ECR
echo "🔑 Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${FULL_REPO_URL}

# 3. Build the ARM64 image
# Note: We use '.' as the context so 'COPY . .' in your Dockerfile sees the whole project
echo "🛠️ Building ARM64 image from ${DOCKERFILE_PATH}..."
docker buildx build \
  --platform linux/arm64 \
  -t ${REPO_NAME}:${IMAGE_TAG} \
  -f ${DOCKERFILE_PATH} \
  --load .

# 4. Tag the image for ECR
echo "🏷️ Tagging image..."
docker tag ${REPO_NAME}:${IMAGE_TAG} ${FULL_REPO_URL}:${IMAGE_TAG}

# 5. Push to ECR
echo "📤 Pushing to ECR..."
docker push ${FULL_REPO_URL}:${IMAGE_TAG}

echo "---"
echo "✅ Success! Optimization image is now at:"
echo "${FULL_REPO_URL}:${IMAGE_TAG}"