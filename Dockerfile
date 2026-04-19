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

# STEP 3: Setup non-root user and copy code
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN mkdir -p data models && chown -R appuser:appuser /app
USER appuser

COPY src/ ./src/
COPY .env.example .

EXPOSE 8080

# Run the dashboard (One-Click Setup)
CMD streamlit run src/dashboard/app.py --server.port $PORT --server.address 0.0.0.0