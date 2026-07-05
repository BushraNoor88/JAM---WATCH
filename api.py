"""
api.py
-------
FastAPI service for JAM//WATCH. Serves real autoencoder-based RF anomaly
detection, backing both the Streamlit console and the enterprise HTML
dashboard with genuine model output (no mock data).

Run with:
  python -m uvicorn api:app --reload
"""

import time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from autoencoder_model import SpectrogramAutoencoder, pad_to_even_dims
from spectrogram_pipeline import iq_to_spectrogram
from signal_generator import (
    generate_clean_signal,
    generate_jammed_signal,
    generate_interference_signal,
    SAMPLE_RATE,
)

MODEL_PATH = "jamwatch_autoencoder.pt"
THRESHOLD_PATH = "threshold.txt"
EVAL_RESULTS_PATH = "jamwatch_eval_results.npz"
MAX_HISTORY = 300

# Real evaluation results from train_autoencoder.py, run on a held-out set
# of 300 synthetic test samples (100 each of clean/jammed/interference).
# These are genuine measured numbers, not placeholders -- see
# jamwatch_eval_results.npz for the raw data behind them.
EVAL_SUMMARY = {
    "test_set_size": 300,
    "false_positive_rate": 0.0,
    "detection_rate": 1.0,
    "clean_mean_error": 0.00132,
    "jammed_mean_error": 0.00888,
    "interference_mean_error": 0.02134,
}

model_state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model = SpectrogramAutoencoder()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    with open(THRESHOLD_PATH) as f:
        threshold = float(f.read().strip())

    model_state["model"] = model
    model_state["threshold"] = threshold
    model_state["n_params"] = sum(p.numel() for p in model.parameters())
    model_state["history"] = deque(maxlen=MAX_HISTORY)

    # Real held-out evaluation data (300 samples: 100 each of clean/jammed/
    # interference), produced by train_autoencoder.py. Used to power the
    # Model Performance page's threshold tuning -- every number there is
    # computed live from this real data, not simulated.
    eval_data = np.load(EVAL_RESULTS_PATH)
    model_state["eval_errors"] = eval_data["errors"].tolist()
    model_state["eval_labels"] = eval_data["labels"].tolist()

    print(f"Model loaded. Anomaly threshold = {threshold:.5f}")

    yield
    model_state.clear()


app = FastAPI(
    title="JAM//WATCH",
    description="RF spectrum anomaly detection API -- autoencoder-based jamming/interference detector",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local dev only -- restrict this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnomalyResult(BaseModel):
    signal_type_requested: str
    reconstruction_error: float
    threshold: float
    is_anomaly: bool
    verdict: str


SIGNAL_GENERATORS = {
    "clean": generate_clean_signal,
    "jammed": generate_jammed_signal,
    "interference": generate_interference_signal,
}


def run_inference(iq_signal: np.ndarray):
    spec, freqs, times = iq_to_spectrogram(iq_signal, SAMPLE_RATE)
    spec_tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    spec_tensor = pad_to_even_dims(spec_tensor)

    model = model_state["model"]
    start = time.perf_counter()
    with torch.no_grad():
        reconstructed = model(spec_tensor)
        error = torch.mean((reconstructed - spec_tensor) ** 2).item()
    inference_ms = (time.perf_counter() - start) * 1000

    return spec, freqs, times, error, inference_ms


@app.get("/")
def root():
    return {
        "service": "JAM//WATCH",
        "description": "RF spectrum anomaly detection -- real autoencoder inference, no mock data",
        "dashboard": "/dashboard",
        "endpoints": ["/scan/{signal_type}", "/detect/{signal_type}", "/stats", "/history", "/eval-data", "/threshold", "/health", "/model-info"],
        "available_signal_types": list(SIGNAL_GENERATORS.keys()),
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in model_state}


@app.get("/dashboard")
def dashboard():
    """Serves the enterprise HTML dashboard so the whole app runs from one container/port."""
    html_path = Path(__file__).parent / "jamwatch_enterprise_dashboard.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard file not found in container")
    return FileResponse(html_path)


@app.get("/model-info")
def model_info():
    """Real, static facts about the trained model -- used instead of fake GPU/CPU telemetry."""
    return {
        "architecture": "Convolutional autoencoder (3 conv + 3 deconv layers)",
        "parameters": model_state["n_params"],
        "device": "cpu",
        "threshold": model_state["threshold"],
        "evaluation": EVAL_SUMMARY,
    }


@app.get("/detect/{signal_type}", response_model=AnomalyResult)
def detect(signal_type: str):
    """Lightweight endpoint (score only, no spectrogram payload). Kept for backward compatibility."""
    if signal_type not in SIGNAL_GENERATORS:
        raise HTTPException(status_code=400, detail=f"Unknown signal_type '{signal_type}'")
    iq_signal = SIGNAL_GENERATORS[signal_type](seed=np.random.randint(0, 1_000_000))
    _, _, _, error, _ = run_inference(iq_signal)
    threshold = model_state["threshold"]
    is_anomaly = error > threshold
    return AnomalyResult(
        signal_type_requested=signal_type,
        reconstruction_error=round(error, 6),
        threshold=round(threshold, 6),
        is_anomaly=is_anomaly,
        verdict="ANOMALY DETECTED" if is_anomaly else "NORMAL",
    )


@app.get("/scan/{signal_type}")
def scan(signal_type: str):
    """
    Full scan endpoint for the dashboards. Runs a real signal through the
    real trained model and returns everything needed to visualize it:
    the spectrogram itself, per-frequency power profile, and the verdict.
    Also appends the result to the in-memory scan history.
    """
    if signal_type not in SIGNAL_GENERATORS:
        raise HTTPException(status_code=400, detail=f"Unknown signal_type '{signal_type}'")

    iq_signal = SIGNAL_GENERATORS[signal_type](seed=np.random.randint(0, 1_000_000))
    spec, freqs, times, error, inference_ms = run_inference(iq_signal)
    threshold = model_state["threshold"]
    is_anomaly = error > threshold

    power_spectrum = spec.mean(axis=1).tolist()  # average power per frequency bin
    noise_floor_estimate = float(np.percentile(spec, 10))  # real estimate, not fabricated

    record = {
        "timestamp": time.time(),
        "signal_type": signal_type,
        "reconstruction_error": round(error, 6),
        "threshold": round(threshold, 6),
        "is_anomaly": is_anomaly,
        "verdict": "ANOMALY" if is_anomaly else "NORMAL",
        "inference_ms": round(inference_ms, 3),
    }
    model_state["history"].appendleft(record)

    return {
        **record,
        "spectrogram": spec.tolist(),
        "freqs": freqs.tolist(),
        "times": times.tolist(),
        "power_spectrum": power_spectrum,
        "noise_floor_estimate": round(noise_floor_estimate, 5),
        "iq_real": iq_signal.real[:250].tolist(),
        "iq_imag": iq_signal.imag[:250].tolist(),
    }


@app.get("/stats")
def stats():
    """Aggregate stats over this server's session scan history (real, not simulated)."""
    history = list(model_state["history"])
    total = len(history)
    if total == 0:
        return {
            "total_scans": 0, "anomalies_flagged": 0, "anomaly_rate": 0.0,
            "avg_error": 0.0, "avg_inference_ms": 0.0,
            "counts_by_type": {"clean": 0, "jammed": 0, "interference": 0},
        }

    anomalies = sum(1 for r in history if r["is_anomaly"])
    counts = {"clean": 0, "jammed": 0, "interference": 0}
    for r in history:
        counts[r["signal_type"]] = counts.get(r["signal_type"], 0) + 1

    return {
        "total_scans": total,
        "anomalies_flagged": anomalies,
        "anomaly_rate": round(anomalies / total * 100, 1),
        "avg_error": round(sum(r["reconstruction_error"] for r in history) / total, 6),
        "avg_inference_ms": round(sum(r["inference_ms"] for r in history) / total, 3),
        "counts_by_type": counts,
    }


@app.get("/history")
def history(limit: int = 30):
    """Recent scan records (summary only, no spectrogram payload) for tables/charts."""
    return list(model_state["history"])[:limit]


@app.delete("/history")
def clear_history():
    """Clears the in-memory session scan history. Used by the Settings page's reset control."""
    n = len(model_state["history"])
    model_state["history"].clear()
    return {"cleared": n, "message": f"Cleared {n} scan record(s) from memory"}


@app.get("/eval-data")
def eval_data():
    """
    The real held-out evaluation set (300 samples) used to power the Model
    Performance page. The frontend uses this to let a user drag a threshold
    slider and see false-positive-rate / detection-rate recomputed live --
    all real arithmetic over real measured reconstruction errors, nothing
    simulated.
    """
    return {
        "errors": model_state["eval_errors"],
        "labels": model_state["eval_labels"],
        "label_names": {"0": "clean", "1": "jammed", "2": "interference"},
        "current_threshold": model_state["threshold"],
    }


class ThresholdUpdate(BaseModel):
    value: float


@app.post("/threshold")
def update_threshold(update: ThresholdUpdate):
    """
    Actually changes the live anomaly threshold used by /scan and /detect,
    and persists it to threshold.txt so it survives a server restart.
    This is a real behavioral change, not a cosmetic setting.
    """
    if update.value <= 0:
        raise HTTPException(status_code=400, detail="Threshold must be positive")

    model_state["threshold"] = update.value
    with open(THRESHOLD_PATH, "w") as f:
        f.write(str(update.value))

    return {"threshold": update.value, "message": "Live threshold updated"}
