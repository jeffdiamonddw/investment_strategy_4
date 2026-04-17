#!/bin/bash

# --- CONFIGURATION ---
AWS_REGION="us-west-2"
AWS_ACCOUNT_ID="129861351772"
REPO_NAME="investment-base-image"
IMAGE_TAG="latest"
FULL_REPO_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${REPO_NAME}"

echo "🚀 Starting deployment for ${REPO_NAME}..."

# 1. Ensure the ECR repository exists (silently ignores if already exists)
aws ecr create-repository --repository-name ${REPO_NAME} --region ${AWS_REGION} > /dev/null 2>&1 || true

# 2. Authenticate Docker to ECR
echo "🔑 Authenticating with ECR..."
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${FULL_REPO_URL}

# 3. Build for ARM64 (Graviton4 compatibility)
# Using buildx ensures that even if you build on Intel, it works on c8g instances
echo "🛠️ Building ARM64 image..."
# -f specifies the path to the Dockerfile
# . (the dot) specifies the 'context' (where your code is)
docker buildx build \
  --platform linux/arm64 \
  -t ${REPO_NAME}:${IMAGE_TAG} \
  -f docker/base_image.dockerfile \
  --load .

# 4. Tag the image for ECR
echo "🏷️ Tagging image..."
docker tag ${REPO_NAME}:${IMAGE_TAG} ${FULL_REPO_URL}:${IMAGE_TAG}

# 5. Push to ECR
echo "📤 Pushing to ECR..."
docker push ${FULL_REPO_URL}:${IMAGE_TAG}

echo "✅ Success! Image is now at ${FULL_REPO_URL}:${IMAGE_TAG}"