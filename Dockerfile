FROM python:3.10-slim

# Install minimal build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# STEP 1: Force-install CPU-only Torch immediately
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# STEP 2: Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# STEP 3: Setup non-root user and persistent directories
RUN groupadd -r appuser && useradd -r -g appuser appuser
# Create directories and ensure appuser owns them for SQLite/Model caching
RUN mkdir -p /app/data /app/models && \
    chown -R appuser:appuser /app/data /app/models && \
    chmod -R 775 /app/data /app/models

USER appuser

# Copy source code
COPY src/ ./src/
COPY .env.example .

# Expose the port Railway expects
EXPOSE 8080

# The Final Entrypoint: Uses $PORT so Railway can find the app
CMD streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true