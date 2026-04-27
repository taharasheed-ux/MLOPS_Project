"""
Tests for the FastAPI Inference Service.
"""

import pytest
from fastapi.testclient import TestClient
from api.app import app, state
from src.train import load_model, run_training_pipeline
from src.utils import get_path

# Create a test client
client = TestClient(app)

@pytest.fixture(scope="module", autouse=True)
def setup_api_state():
    """Ensure tests run with a valid loaded model."""
    from src.feature_encoding import EncoderPipeline
    from src.data_processing import load_data, clean_data, split_data, save_processed_data
    
    # Just in case artifacts don't exist in the test environment yet:
    # Check if models/encoder_pipeline.pkl exists, if not, wait or throw an error.
    # In regular pipeline they should be created by Milestone 2 test or training scripts.
    models_dir = get_path("models_dir")
    model_paths = list(models_dir.glob("xgb_model_v*.pkl"))
    encoder_path = models_dir / "encoder_pipeline.pkl"
    
    if not model_paths or not encoder_path.exists():
        pytest.skip("Models not trained yet. Run `python -m src.train` first to test the API.")

    # Trigger initialization (like app startup)
    from api.app import load_artifacts
    assert load_artifacts() is True

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "prediction_requests_total" in response.text
    assert "current_model_version" in response.text

def test_predict_endpoint():
    payload = {
        "records": [
            {
                "age": 39,
                "workclass": "State-gov",
                "fnlwgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        ]
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "probabilities" in data
    assert "model_version" in data
    assert len(data["predictions"]) == 1
    assert data["predictions"][0] in [0, 1]

def test_predict_endpoint_missing_fields():
    payload = {
        "records": [
            {
                "age": 39
                # Missing all other fields
            }
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422 # Pydantic Validation Error
