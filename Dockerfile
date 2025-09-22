# Multi-service container: Streamlit UI (port 8501) + FastAPI (port 8000)
FROM python:3.11-slim

# System deps for geopandas (choropleth) and proj
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
    libgeos-dev libproj-dev proj-bin gdal-bin \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

# Expose both ports (Azure will typically route to $PORT; we proxy via startup.sh)
EXPOSE 8000 8501

# Default startup launches both services
CMD ["bash", "startup.sh"]
