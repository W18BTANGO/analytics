from fastapi.testclient import TestClient
import pytest
from app.main import app  # Assuming your FastAPI app is in the 'main.py' file

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_predict(client):
    data = {
        "x": [1, 2, 3, 4],
        "y": [2, 4, 6, 8],
        "predict": [5, 6]
    }
    response = client.post("/predict", json=data)
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], list)
    assert all(isinstance(i, float) for i in response.json()["prediction"])

def test_average_price_by_suburb(client):
    data = {
        "prices": [300000, 400000, 500000, 600000],
        "suburbs": ["Suburb1", "Suburb2", "Suburb1", "Suburb2"]
    }
    response = client.post("/average-price-by-suburb", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "average_prices" in result
    assert result["average_prices"]["Suburb1"] == 400000
    assert result["average_prices"]["Suburb2"] == 500000

def test_median_price_by_suburb(client):
    data = {
        "prices": [300000, 400000, 500000, 600000],
        "suburbs": ["Suburb1", "Suburb2", "Suburb1", "Suburb2"]
    }
    response = client.post("/median-price-by-suburb", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "median_prices" in result
    assert result["median_prices"]["Suburb1"] == 400000
    assert result["median_prices"]["Suburb2"] == 500000

def test_highest(client):
    data = {"prices": [300000, 500000, 450000]}
    response = client.get("/highest", params=data)
    print(response.json())
    assert response.status_code == 200
    result = response.json()
    assert "highest_value" in result
    assert result["highest_value"] == 500000


def test_lowest(client):
    data = {"prices": [300000, 500000, 450000]}
    response = client.get("/lowest", params=data)
    assert response.status_code == 200
    result = response.json()
    assert "lowest_value" in result
    assert result["lowest_value"] == 300000

def test_median(client):
    data = {"prices": [300000, 500000, 450000]}
    response = client.get("/median", params=data)
    assert response.status_code == 200
    result = response.json()
    assert "median" in result
    # This will assert the median calculation is correct (should be 450000)
    assert result["median"] == 450000

def test_highest_no_prices(client):
    # Test when no prices are provided in the query
    response = client.get("/highest", params={})
    assert response.status_code == 400
    assert response.json() == {"detail": "400: No prices provided."}

def test_lowest_no_prices(client):
    # Test when no prices are provided in the query
    response = client.get("/lowest", params={})
    assert response.status_code == 400
    assert response.json() == {"detail": "400: No prices provided."}

def test_median_no_prices(client):
    # Test when no prices are provided in the query
    response = client.get("/median", params={})
    print(response.json()['detail'])
    assert response.status_code == 400
    assert response.json() == {"detail": "400: No prices provided."}
