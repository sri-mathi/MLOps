from fastapi.testclient import TestClient
import pytest
from src.api.main import app, HISTORY_CACHE

@pytest.fixture
def client():
    """Context managed TestClient to trigger FastAPI startup event handlers."""
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    """Verify that the health check endpoint returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "model_loaded" in data

def test_predict_validation_error(client):
    """Verify that sending invalid data returns 422 validation error."""
    # Sending negative pm2_5 should fail Pydantic gt=0 validation
    payload = {
        "city": "Chennai",
        "timestamp": "2026-05-18T12:00:00",
        "pm2_5": -10.0,
        "pm10": 20.0,
        "nitrogen_dioxide": 15.0,
        "sulphur_dioxide": 5.0,
        "carbon_monoxide": 0.5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_success(client):
    """Verify that a valid payload returns a successful prediction and updates history cache."""
    # Reset cache for test predictability
    HISTORY_CACHE["Chennai"] = []
    
    payload = {
        "city": "Chennai",
        "timestamp": "2026-05-18T12:00:00",
        "pm2_5": 25.5,
        "pm10": 45.0,
        "nitrogen_dioxide": 20.0,
        "sulphur_dioxide": 8.0,
        "carbon_monoxide": 0.6
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["city"] == "Chennai"
    assert "is_anomaly" in data
    assert "anomaly_score" in data
    assert "status" in data
    
    # Verify the history cache for Chennai was updated
    assert len(HISTORY_CACHE["Chennai"]) == 1
    assert HISTORY_CACHE["Chennai"][0]["pm2_5"] == 25.5
