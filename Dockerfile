# To run FastAPI separately, use:
# docker exec <container> uvicorn src.api.main:app --host 0.0.0.0 --port 8000

FROM python:3.11-slim

# Install system dependencies (including gcc for compiling FAISS if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create empty directories for persistence
RUN mkdir -p data models

# Copy application files
COPY src/ ./src/
# Ensure empty placeholders have .gitkeep or exist
RUN touch data/.gitkeep models/.gitkeep

# Expose ports: 8000 for FastAPI API, 8501 for Streamlit Dashboard
EXPOSE 8000
EXPOSE 8501

# Default launch command runs the Streamlit UI dashboard
CMD ["streamlit", "run", "src/dashboard/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]