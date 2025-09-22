#!/usr/bin/env bash
set -e

# Allow overriding port via $PORT (Azure sets this). We'll bind FastAPI to $PORT if present.
API_PORT=${PORT:-8000}
STREAMLIT_PORT=8501

echo "Starting Streamlit on ${STREAMLIT_PORT} ..."
streamlit run app.py --server.port ${STREAMLIT_PORT} --server.address 0.0.0.0 &

echo "Starting FastAPI (uvicorn) on ${API_PORT} ..."
exec uvicorn api:app --host 0.0.0.0 --port ${API_PORT}
