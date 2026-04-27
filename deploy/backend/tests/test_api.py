import pytest
from fastapi.testclient import TestClient
from main import app, model_state

client = TestClient(app)

@pytest.fixture(autouse=True)
def setup_state():
    # Setup some test state
    model_state['device'] = 'cpu'

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "device": "cpu"}

def test_readiness_check():
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}

def test_predict_slip_valid():
    response = client.post("/predict", json={
        "mw": 6.5,
        "strk": 180.0,
        "dip": 45.0,
        "rake": 90.0,
        "lat": 0.0,
        "lon": 0.0,
        "dep": 10.0,
        "nx": 50,
        "nz": 50,
        "dx": 1.0,
        "dz": 1.0,
        "random_seed": 42,
        "apply_dz": False
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "slip_map_2d" in data
    # Matrix dimensions check
    assert len(data["slip_map_2d"]) == 50
    assert len(data["slip_map_2d"][0]) == 50

def test_predict_slip_invalid_mw():
    response = client.post("/predict", json={
        "mw": 11.0, # Greater than max 10.0
        "strk": 180.0,
    })
    assert response.status_code == 422 # Validation Error
