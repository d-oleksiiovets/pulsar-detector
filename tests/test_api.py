from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid_input():
    payload = {
        "mean_profile": 140.5, "std_profile": 55.7,
        "kurtosis_profile": -0.2, "skewness_profile": 0.3,
        "mean_dm_snr": 3.2, "std_dm_snr": 19.1,
        "kurtosis_dm_snr": 7.9, "skewness_dm_snr": 74.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert 0.0 <= body["probability"] <= 1.0

def test_predict_invalid_threshold():
    payload = {"mean_profile": 1, "std_profile": 1, "kurtosis_profile": 1,
               "skewness_profile": 1, "mean_dm_snr": 1, "std_dm_snr": 1,
               "kurtosis_dm_snr": 1, "skewness_dm_snr": 1}
    response = client.post("/predict?threshold=1.5", json=payload)
    assert response.status_code == 400

def test_predict_missing_field():
    response = client.post("/predict", json={"mean_profile": 1})
    assert response.status_code == 422