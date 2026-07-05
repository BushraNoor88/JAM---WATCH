# JAM//WATCH -- RF spectrum anomaly detection service
# Single-container build: FastAPI backend + enterprise dashboard, served on one port.
#
# Build:  docker build -t jamwatch .
# Run:    docker run -p 8000:8000 jamwatch
# Then open http://localhost:8000/dashboard

FROM python:3.11-slim

WORKDIR /app

# Keep Python output unbuffered so logs show up immediately in `docker logs`
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Install PyTorch from the CPU-only wheel index -- this is what keeps the
# image a few hundred MB instead of several GB. There's no GPU in this
# container, so the CUDA-enabled build would be pure dead weight.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# Only the files actually needed at inference time -- training scripts,
# the raw training dataset, and generated plots are intentionally left out
# (see .dockerignore). This keeps the image lean and the intent clear:
# this container serves a trained model, it doesn't train one.
COPY api.py autoencoder_model.py spectrogram_pipeline.py signal_generator.py ./
COPY jamwatch_autoencoder.pt threshold.txt jamwatch_eval_results.npz ./
COPY jamwatch_enterprise_dashboard.html ./

EXPOSE 8000

# Basic container health check -- lets `docker ps` and orchestrators
# (Kubernetes, Compose) know when the API is actually ready to serve traffic.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
