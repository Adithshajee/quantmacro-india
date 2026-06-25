#!/bin/bash
# Start FastAPI backend on port 8000 (internal)
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 &

# Give FastAPI 4 seconds to start before Streamlit tries to connect
sleep 4

# Start Streamlit on port 7860 (HF Spaces requires this port)
streamlit run src/dashboard/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true
