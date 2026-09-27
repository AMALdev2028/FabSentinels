# FabSentinel dashboard (+ Yield Planner page) for Koyeb / any Docker host.
FROM python:3.11-slim

# libgomp1: OpenMP runtime that XGBoost needs; not in the slim image.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Koyeb sets $PORT; 8000 is the default it expects.
ENV PORT=8000
EXPOSE 8000
CMD streamlit run src/dashboard.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
