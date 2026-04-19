FROM python:3.10-slim

# Install minimal build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# STEP 1: Force-install CPU-only Torch
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 3: Setup appuser WITH a home directory
# This fixes the "Permission denied: /home/appuser" error
RUN groupadd -r appuser && useradd -r -g appuser -m -d /home/appuser appuser

# Create directories and grant full ownership to appuser
RUN mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app /home/appuser && \
    chmod -R 775 /app/data /app/models

USER appuser
# Set HOME env variable so Streamlit knows where to write its metrics
ENV HOME=/home/appuser

# Copy source code
COPY src/ ./src/
COPY .env.example .

EXPOSE 8080

# The Entrypoint
CMD streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true