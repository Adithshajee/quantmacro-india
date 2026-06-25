#!/bin/bash

# 1. Start the FastAPI backend application in the background
echo "Launching FastAPI backend execution network on port 8000..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# 2. Give the microservice a brief moment to initialize
sleep 3

# 3. Start the primary Streamlit interface on the mandatory HF port
echo "Launching Streamlit UI orchestration layer on Hugging Face port 7860..."
streamlit run src/dashboard/app.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless true
