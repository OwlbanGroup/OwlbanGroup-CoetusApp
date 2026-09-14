# Use NVIDIA CUDA base image for Blackwell support
FROM nvidia/cuda:12.4-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python, pip, and curl (for the container healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run as an unprivileged user (the payroll SQLite DB lives in /app)
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Run the application (bind all interfaces so the container runtime routes)
ENV HOST=0.0.0.0

# Liveness: the app exposes /health. start-period is generous because the
# ResNet50 weights download and model load can take a while on first boot.
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/health || exit 1

CMD ["python3", "app.py"]
