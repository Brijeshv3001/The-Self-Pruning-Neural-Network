# =================================================================
# File: Dockerfile
# Project: The Self-Pruning Neural Network
# Description: Multi-stage Docker build
# =================================================================

# Stage 1: Build base environment
FROM python:3.10-slim AS builder
WORKDIR /app
# Install core Python dependencies system-wide to avoid venv overhead in container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime image configuration
FROM python:3.10-slim AS runtime
WORKDIR /app
# Migrate over the constructed library environment from Builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Integrate the functional codebase entirely
COPY . .

# Expose standard FastAPI application port
EXPOSE 8000

# Fire the Uvicorn ASGI server bridging FastAPI instances
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
