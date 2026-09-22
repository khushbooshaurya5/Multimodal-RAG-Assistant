# Multimodal RAG Assistant - CPU image.
# Works locally (docker run -p 8501:8501) and as a Hugging Face Docker Space (app_port 8501).
# GPU: swap the base for an nvidia/cuda image and install a CUDA torch wheel.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg is used for MP3/M4A decoding when soundfile cannot decode a file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces run the container as uid 1000; everything writable lives in its home.
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    MRAG_DATA_DIR=/home/user/data \
    MRAG_INDEX_DIR=/home/user/data/index

WORKDIR /app

COPY requirements.txt .
# CPU-only torch first so the generic requirements do not pull CUDA wheels.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p /home/user/data/raw /home/user/data/processed /home/user/data/index /home/user/.cache/huggingface \
    && chown -R user:user /home/user /app

USER user

EXPOSE 8501 8080

# Default: Streamlit UI. Override with e.g.
#   docker run ... uvicorn src.api.server:app --host 0.0.0.0 --port 8080
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
