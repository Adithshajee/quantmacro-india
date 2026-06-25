FROM python:3.11-slim

# Required by HF Spaces: create user with UID 1000
RUN useradd -m -u 1000 user

# Install system dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Redirect ALL caches to /tmp (only writable dir on HF Spaces)
ENV HF_HOME=/tmp/hf_cache
ENV TRANSFORMERS_CACHE=/tmp/hf_cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/tmp/sentence_transformers
ENV XDG_CACHE_HOME=/tmp/cache

WORKDIR /home/user/app

# Install dependencies first (better Docker layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user src/ ./src/
COPY --chown=user data/ ./data/
COPY --chown=user models/ ./models/
COPY --chown=user start.sh ./start.sh

# Create writable dirs for runtime
RUN mkdir -p /tmp/hf_cache /tmp/sentence_transformers /tmp/cache && \
    chmod +x start.sh

USER user
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH

EXPOSE 7860

CMD ["./start.sh"]