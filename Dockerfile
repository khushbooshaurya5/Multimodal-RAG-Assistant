# Multimodal RAG Assistant - CPU image (GPU: swap base for an nvidia/cuda image
# and install a CUDA torch wheel).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models_cache

# ffmpeg is used for MP3/M4A decoding when soundfile cannot decode a file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Install a CPU-only torch first so the generic requirements do not pull CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY . .

RUN mkdir -p data/raw data/processed data/index /models_cache

EXPOSE 8501 8080

# Default: Streamlit UI. Override with e.g.
#   docker run ... uvicorn src.api.server:app --host 0.0.0.0 --port 8080
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
