FROM python:3.11-slim

WORKDIR /app

# Install critical system compilation assets
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Cache dependency layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

# Copy startup scripts
COPY start.sh .
RUN chmod +x start.sh

# Expose mandatory Hugging Face traffic port
EXPOSE 7860

# Execute orchestration wrapper
CMD ["./start.sh"]