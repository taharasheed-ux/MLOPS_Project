"""
FastAPI Inference Service for the Drift-Aware MLOps Pipeline.

Exposes:
- POST /predict       : Make predictions on raw data.
- POST /reload-model  : Hot-swap to the latest model version.
- GET /metrics        : Prometheus metrics for monitoring.
"""

import pandas as pd
from typing import Any, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import make_asgi_app, Counter, Histogram, Gauge

from src.train import load_model
from src.feature_encoding import EncoderPipeline
from src.utils import get_path, setup_logging

logger = setup_logging("api", log_file="api.log")

app = FastAPI(title="Drift-Aware MLOps API", version="1.0.0")

# ── Prometheus Metrics ──────────────────────────────────────────────

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Total prediction requests made"
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Latency of prediction requests"
)
MODEL_VERSION = Gauge(
    "current_model_version", "Currently loaded model version number"
)
DRIFT_SEVERITY = Gauge(
    "drift_severity", "Latest global drift severity score"
)
RETRAIN_EVENTS = Counter(
    "retrain_events_total", "Total policy-triggered retrain events"
)


# ── Global State ────────────────────────────────────────────────────
# We load the model and encoder lazily or at startup.

class AppState:
    model = None
    encoder = None
    version_int = 1

state = AppState()


def load_artifacts():
    """Load the latest model and encoder from disk."""
    try:
        # Load encoder
        encoder_path = get_path("models_dir") / "encoder_pipeline.pkl"
        if not encoder_path.exists():
            logger.warning(f"Encoder not found at {encoder_path}")
            return False

        encoder = EncoderPipeline.load(encoder_path)
        
        # Load the latest model (we'll just glob the highest version)
        models_dir = get_path("models_dir")
        model_files = list(models_dir.glob("xgb_model_v*.pkl"))
        
        if not model_files:
            logger.warning("No model files found.")
            return False
            
        # Parse version numbers and find the max
        # e.g., xgb_model_v2.pkl -> 2
        def extract_version(p):
            try:
                return int(p.stem.split("_v")[-1])
            except ValueError:
                return 0
                
        latest_file = max(model_files, key=extract_version)
        version_int = extract_version(latest_file)
        
        state.model = load_model(latest_file)
        state.encoder = encoder
        state.version_int = version_int
        
        MODEL_VERSION.set(version_int)
        logger.info(f"Loaded artifacts: Model v{version_int}, Encoder {encoder_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        return False


@app.on_event("startup")
def startup_event():
    logger.info("Starting up API...")
    load_artifacts()


# ── Pydantic Models ─────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    records: List[dict[str, Any]]


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_version: str


# ── Endpoints ───────────────────────────────────────────────────────

@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "model_loaded": state.model is not None}


@app.post("/reload-model")
def reload_model_endpoint():
    """Hot-swap the model and encoder in memory (triggered after retrain)."""
    success = load_artifacts()
    if success:
        return {"status": "success", "message": f"Loaded model v{state.version_int}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to load model artifacts")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Make predictions on raw data."""
    if state.model is None or state.encoder is None:
        success = load_artifacts()
        if not success:
            raise HTTPException(status_code=503, detail="Model artifacts not available")

    PREDICTION_REQUESTS.inc(len(request.records))

    # Convert to DataFrame
    if not request.records:
        raise HTTPException(status_code=400, detail="At least one record is required")

    df = pd.DataFrame(request.records)

    with PREDICTION_LATENCY.time():
        # Encode features
        try:
            required_columns = state.encoder.get_feature_names()
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"Missing required columns: {missing}")
            encoded_df = state.encoder.transform(df, include_target=False)
            X = encoded_df[required_columns]
        except Exception as e:
            logger.error(f"Encoding error: {e}")
            raise HTTPException(status_code=400, detail=f"Data encoding failed: {e}")

        # Predict
        try:
            preds = state.model.predict(X)
            probs = state.model.predict_proba(X)[:, 1]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    return PredictionResponse(
        predictions=preds.tolist(),
        probabilities=probs.tolist(),
        model_version=f"v{state.version_int}",
    )


@app.post("/internal/update-drift-metrics")
def update_drift_metrics(severity: float, retrain_triggered: bool):
    """
    Internal endpoint to update Prometheus gauges from the background pipeline.
    This acts as a bridge so Prometheus can scrape the metrics.
    """
    DRIFT_SEVERITY.set(severity)
    if retrain_triggered:
        RETRAIN_EVENTS.inc()
    return {"status": "metrics updated"}
