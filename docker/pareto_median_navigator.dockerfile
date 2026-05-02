# 1. Use the base image you just pushed/built
# Ensure this matches your AWS Account ID
FROM 129861351772.dkr.ecr.us-east-1.amazonaws.com/investment-base-image:latest

# 2. Set the working directory (already created in base)
WORKDIR /app

# 3. Copy your project code into the container
# This copies everything from the root context into /app
COPY . .

# 4. Set project-specific environment variables
# Ensure your logic can find modules in the root
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# 5. Optimization for high-core Graviton4 (Spot Instances)
# Prevents libraries like OpenBLAS from creating too many threads
ENV OMP_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1

RUN pip install psutil

# 6. Execute your specific parallel dry run
CMD ["python3", "python_scripts/robust_median_evaluator.py"]